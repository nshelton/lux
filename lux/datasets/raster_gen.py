"""Pure-NumPy analytic ray-caster structured-light renderer.

A fast alternative backend to the Mitsuba path tracer (:mod:`lux.datasets.mitsuba_gen`),
with the *same* interface (``load_geometry``, ``render_ground_truth``,
``render_capture``) so the dataset scripts can swap between them. It renders the
same scene files (parsed to analytic primitives by
:func:`lux.datasets.scene_loader.load_scene_primitives`).

Why a ray-caster makes the ground truth exact *by construction*: we cast exactly
``lux.geometry.camera_rays`` (Z=1, pixel-index convention), so a hit point is
``depth · ray`` and ``depth`` is its Z. :func:`lux.datasets.correspondence.projector_subpixel`
back-projects ``depth · camera_rays`` and projects into the projector — recovering
the identical 3D point, hence the identical projector coordinate. The capture
samples the pattern at that same coordinate, so ``gt_proj`` equals the capture's
illuminating projector pixel to machine precision.

Frames/units match mitsuba_gen: camera at the origin looking down +Z, +X right,
+Y down, metres; projector posed by the rig (``X_proj = R·X_cam + t``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..geometry import Rig, camera_rays, project
from ..render import RenderConfig, add_sensor_noise, sample_pattern
from .correspondence import projector_subpixel
from .scene_loader import Box, Plane, Sphere, Wavy, load_scene_primitives

_EPS = 1e-6


def load_geometry(scene: str = "blocks") -> list:
    """Parse a scene file (built-in name or path) into analytic primitives."""
    return load_scene_primitives(scene)


def default_geometry() -> list:
    return load_geometry("blocks")


# --------------------------------------------------------------------------
# Ray-primitive intersection (origin-generalised, vectorised over all pixels)
# --------------------------------------------------------------------------
# Each intersector takes a ray origin ``o`` (3,) and per-pixel directions ``d``
# (..., 3) and returns (t, normal, hit) where a hit point is ``o + t·d``. Used
# with ``o = 0`` for the camera and ``o = projector centre`` for shadow rays.

def _isect_plane(o, d, p: Plane):
    dz = d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (p.z - o[2]) / dz
    x = o[0] + t * d[..., 0]
    y = o[1] + t * d[..., 1]
    hit = np.isfinite(t) & (t > _EPS) & (np.abs(x) <= p.sx) & (np.abs(y) <= p.sy)
    n = np.zeros(d.shape)
    n[..., 2] = -1.0
    return t, n, hit


def _isect_sphere(o, d, p: Sphere):
    oc = o - p.center
    a = np.sum(d * d, axis=-1)
    b = 2.0 * np.sum(d * oc, axis=-1)
    cc = float(oc @ oc) - p.radius ** 2
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.maximum(disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - sq) / (2 * a)
    hit = (disc > 0) & (t > _EPS)
    pt = o + t[..., None] * d
    n = (pt - p.center) / p.radius
    return t, n, hit


def _isect_box(o, d, p: Box):
    # Slab test in the box frame: for an oriented box, rotate the ray into box
    # coordinates (columns of R are the box axes, so world->box is `v @ R`); t is
    # unchanged by the rigid transform, so depth stays exact.
    oc = o - p.center
    if p.R is not None:
        oc = oc @ p.R
        d = d @ p.R
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (-p.half - oc) * inv
        t2 = (p.half - oc) * inv
    tmin = np.minimum(t1, t2)          # (..., 3)
    tmax = np.maximum(t1, t2)
    t_near = np.max(tmin, axis=-1)
    t_far = np.min(tmax, axis=-1)
    hit = (t_near <= t_far) & (t_far > _EPS)
    t = np.where(t_near > _EPS, t_near, t_far)
    # Face normal: the slab axis that produced t_near, pointing back along the ray.
    axis = np.argmax(tmin, axis=-1)                      # (...)
    dg = np.take_along_axis(d, axis[..., None], axis=-1)[..., 0]
    n = np.zeros(d.shape)
    np.put_along_axis(n, axis[..., None], -np.sign(dg)[..., None], axis=-1)
    if p.R is not None:
        n = n @ p.R.T                                    # box-frame normal -> world
    return t, n, hit


def _wavy_f(p: Wavy, x, y):
    """Heightfield value ONLY (no partials). ``f = cz + amp·sin(thx)·sin(thy)`` needs just the
    two sines, vs the four transcendentals (sin+cos of both) :func:`_wavy_fields` computes for the
    gradient. The bracket + bisection only need ``f``, so this halves their transcendental cost
    with bit-identical ``f``; Newton + the final normal still use :func:`_wavy_fields`."""
    thx = 2 * np.pi * p.fx * (x - p.cx) / p.ex + p.px
    thy = 2 * np.pi * p.fy * (y - p.cy) / p.ey + p.py
    return p.cz + p.amp * np.sin(thx) * np.sin(thy)


def _wavy_fields(p: Wavy, x, y):
    """Heightfield f and its x/y partials at world (x, y)."""
    thx = 2 * np.pi * p.fx * (x - p.cx) / p.ex + p.px
    thy = 2 * np.pi * p.fy * (y - p.cy) / p.ey + p.py
    sx, cx = np.sin(thx), np.cos(thx)
    sy, cy = np.sin(thy), np.cos(thy)
    f = p.cz + p.amp * sx * sy
    fx = p.amp * (2 * np.pi * p.fx / p.ex) * cx * sy
    fy = p.amp * (2 * np.pi * p.fy / p.ey) * sx * cy
    return f, fx, fy


def _isect_wavy(o, d, p: Wavy):
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]

    def g(t):  # residual z(t) - f(x(t), y(t)); broadcasts over any trailing t axis
        return (o[2] + t * dz) - _wavy_f(p, o[0] + t * dx, o[1] + t * dy)

    # z(t) = o.z + t·d.z sweeps the bounded band [cz-amp, cz+amp]; the root lives
    # in that t-interval (z linear, f bounded), with g(t_lo)<=0<=g(t_hi).
    with np.errstate(divide="ignore", invalid="ignore"):
        tA = (p.cz - p.amp - o[2]) / dz
        tB = (p.cz + p.amp - o[2]) / dz
    t_lo = np.minimum(tA, tB)
    t_hi = np.maximum(tA, tB)

    # Sample the band to bracket the *nearest* root (first sign change). Done in pixel
    # CHUNKS: a full-frame (H, W, N) sample stack at N=64 is ~13 GB (x~12 temporaries inside
    # `_wavy_fields`) -- it dominated render memory and OOM'd parallel workers. Chunking caps the
    # (chunk, N) intermediates to a few hundred MB with BIT-IDENTICAL output; the bisection/Newton
    # below run on the cheap (H, W) lo/hi, so they (and the GT-exactness) are untouched.
    N = 64
    s = np.linspace(0.0, 1.0, N)
    shp = t_lo.shape
    fl_lo, fl_hi = np.ravel(t_lo), np.ravel(t_hi)
    fdx, fdy, fdz = np.ravel(dx), np.ravel(dy), np.ravel(dz)
    # origin may be a scalar centre (camera/projector) or a per-pixel field (origin-generalised
    # shadow rays); broadcast either to the flat pixel axis.
    fox, foy, foz = (np.broadcast_to(o[k], shp).reshape(-1) for k in range(3))
    M = fl_lo.shape[0]
    lo = np.empty(M, t_lo.dtype); hi = np.empty(M, t_lo.dtype); has = np.empty(M, bool)
    CHUNK = 65536
    for c0 in range(0, M, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, M))
        tl = fl_lo[sl][:, None]
        ts = tl + (fl_hi[sl][:, None] - tl) * s                  # (m, N)
        fc = _wavy_f(p, fox[sl][:, None] + ts * fdx[sl][:, None],
                     foy[sl][:, None] + ts * fdy[sl][:, None])
        gs = (foz[sl][:, None] + ts * fdz[sl][:, None]) - fc
        change = (gs[:, :-1] <= 0) & (gs[:, 1:] > 0)
        idx = np.argmax(change, axis=-1)
        has[sl] = change.any(axis=-1)
        lo[sl] = np.take_along_axis(ts, idx[:, None], axis=-1)[:, 0]
        hi[sl] = np.take_along_axis(ts, (idx + 1)[:, None], axis=-1)[:, 0]
    lo, hi, has = lo.reshape(shp), hi.reshape(shp), has.reshape(shp)

    # Bisection (robust), then a couple of Newton polishing steps.
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        neg = g(mid) <= 0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, mid, hi)
    t = 0.5 * (lo + hi)
    for _ in range(3):
        f, fx, fy = _wavy_fields(p, o[0] + t * dx, o[1] + t * dy)
        gt = (o[2] + t * dz) - f
        gp = dz - (fx * dx + fy * dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = t - gt / gp
        t = np.clip(t, lo, hi)

    x = o[0] + t * d[..., 0]
    y = o[1] + t * d[..., 1]
    _, fx, fy = _wavy_fields(p, x, y)
    in_ext = (np.abs(x - p.cx) <= p.ex / 2) & (np.abs(y - p.cy) <= p.ey / 2)
    hit = has & np.isfinite(t) & (t > _EPS) & in_ext
    n = np.stack([-fx, -fy, np.ones_like(fx)], axis=-1)
    return t, n, hit


_ISECT = {Plane: _isect_plane, Sphere: _isect_sphere, Box: _isect_box, Wavy: _isect_wavy}


def _intersect(o, d, prim):
    return _ISECT[type(prim)](o, d, prim)


# --------------------------------------------------------------------------
# Procedural albedo textures (ray-caster only; Mitsuba ignores the field)
# --------------------------------------------------------------------------
# Evaluated in world space at the hit points, so no UV mapping is needed and the
# pattern is continuous across faces. Returns a multiplier in [1-contrast, 1]
# applied to the primitive's base reflectance.

def _eval_texture(tex: dict, p: np.ndarray) -> np.ndarray:
    kind = tex.get("type", "noise")
    scale = float(tex.get("scale", 0.05))      # feature size, metres
    contrast = float(tex.get("contrast", 0.5))
    if kind == "checker":
        q = np.floor(p / scale).astype(int)
        t01 = ((q[..., 0] + q[..., 1] + q[..., 2]) % 2).astype(float)
    elif kind == "stripes":
        dirv = np.asarray(tex.get("dir", [1.0, 0.0, 0.0]), float)
        dirv = dirv / np.linalg.norm(dirv)
        t01 = 0.5 + 0.5 * np.sin(2 * np.pi * (p @ dirv) / scale
                                 + float(tex.get("phase", 0.0)))
    elif kind == "noise":
        # Value-noise-ish: a seeded sum of random-direction 3D sinusoids,
        # squashed to [0, 1]. Reproducible from the JSON alone.
        rng = np.random.default_rng(int(tex.get("seed", 0)))
        acc = np.zeros(p.shape[:-1])
        K = 6
        for _ in range(K):
            k = rng.normal(size=3)
            k *= (2 * np.pi / scale) * rng.uniform(0.5, 2.0) / np.linalg.norm(k)
            acc += np.sin(p @ k + rng.uniform(0, 2 * np.pi))
        t01 = 0.5 + 0.5 * np.tanh(acc / np.sqrt(K / 2))
    else:
        raise ValueError(f"unknown texture type {kind!r}; "
                         f"supported: checker, stripes, noise")
    return 1.0 - contrast * t01


# --------------------------------------------------------------------------
# Screen-space culling: intersect each primitive only over its projected footprint
# --------------------------------------------------------------------------
_CUBE = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)        # 8 unit-cube corners
                  for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def _prim_world_corners(prim):
    """The primitive's world-space AABB corners (conservative bound), or None if effectively
    unbounded (Plane/Wavy are finite rectangles but typically frame-filling -- bounded here too)."""
    if isinstance(prim, Sphere):
        return prim.center + prim.radius * _CUBE
    if isinstance(prim, Box):
        loc = prim.half * _CUBE                              # box-frame corner offsets
        if prim.R is not None:
            loc = loc @ prim.R.T                             # box-frame -> world
        return prim.center + loc
    if isinstance(prim, Wavy):
        half = np.array([prim.ex / 2.0, prim.ey / 2.0, prim.amp])
        return np.array([prim.cx, prim.cy, prim.cz]) + half * _CUBE
    if isinstance(prim, Plane):
        return np.array([0.0, 0.0, prim.z]) + np.array([prim.sx, prim.sy, 0.0]) * _CUBE
    return None


def _screen_window(prim, dev, R, C, hw, margin: int = 2):
    """Conservative pixel window ``(slice_y, slice_x)`` of where ``prim`` can project into device
    ``dev`` (intrinsics) at world pose (``R`` rows = device axes, ``C`` = centre). Returns the FULL
    frame when the prim is unbounded or straddles the image plane (projection unreliable -> never
    cull a possible hit), or ``None`` when it projects entirely off-frame (skip the prim). A convex
    primitive lies inside its AABB, whose image lies inside the bbox of the projected corners, so the
    window is a guaranteed superset of every hit pixel (+margin for the index/round convention)."""
    H, W = hw
    full = (slice(0, H), slice(0, W))
    corners = _prim_world_corners(prim)
    if corners is None:
        return full
    pd = (corners - C) @ R.T                                 # device frame (8, 3)
    if float(np.min(pd[:, 2])) <= 1e-6:                      # any corner behind/at image plane
        return full
    u = dev.fx * pd[:, 0] / pd[:, 2] + dev.cx
    v = dev.fy * pd[:, 1] / pd[:, 2] + dev.cy
    x0 = max(0, int(np.floor(u.min())) - margin); x1 = min(W, int(np.ceil(u.max())) + margin + 1)
    y0 = max(0, int(np.floor(v.min())) - margin); y1 = min(H, int(np.ceil(v.max())) + margin + 1)
    if x1 <= x0 or y1 <= y0:
        return None                                          # entirely off-frame
    if (x1 - x0) * (y1 - y0) > 0.9 * H * W:                  # ~full anyway: skip slicing overhead
        return full
    return (slice(y0, y1), slice(x0, x1))


# --------------------------------------------------------------------------
# G-buffer
# --------------------------------------------------------------------------
@dataclass
class GBuffer:
    depth: np.ndarray    # (H, W) z-depth, NaN off-surface
    normal: np.ndarray   # (H, W, 3) unit, camera-facing
    albedo: np.ndarray   # (H, W, 3)
    obj_id: np.ndarray   # (H, W) int, -1 background
    mask: np.ndarray     # (H, W) bool


def build_gbuffer(rig: Rig, prims: list) -> GBuffer:
    # Cast the exact camera rays, rotated into the world frame and emitted from the
    # camera centre, against the world-space primitives. The hit param t equals the
    # camera-frame Z (because d_world = R_cam^T·camera_ray preserves the Z=1 ray
    # parametrisation), so depth == t — keeping gt_proj exact for any camera pose.
    cam_rays = camera_rays(rig.camera)                   # (H, W, 3), Z == 1 (camera frame)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d_world = cam_rays @ rig.R_cam                    # -> world directions
    o = rig.C_cam                                         # camera centre in world
    H, W = cam_rays.shape[:2]
    best_t = np.full((H, W), np.inf)
    normal = np.zeros((H, W, 3))
    albedo = np.zeros((H, W, 3))
    obj_id = np.full((H, W), -1, dtype=int)
    for k, prim in enumerate(prims):
        # cull to the primitive's projected pixel window: intersect only those rays, scatter back.
        win = _screen_window(prim, rig.camera, rig.R_cam, o, (H, W))
        if win is None:
            continue                                         # projects entirely off-frame
        sy, sx = win
        dW = d_world[sy, sx]
        t, n, hit = _intersect(o, dW, prim)
        closer = hit & (t < best_t[sy, sx])
        best_t[sy, sx] = np.where(closer, t, best_t[sy, sx])
        normal[sy, sx] = np.where(closer[..., None], n, normal[sy, sx])
        if getattr(prim, "texture", None) is not None:
            # Evaluate the texture ONLY at this object's winning pixels (within the window):
            # full-frame eval was ~all discarded and ran on off-surface points (overflow spam).
            mult = np.ones(closer.shape)
            if closer.any():
                pt = o + t[closer][..., None] * dW[closer]   # (n_closer, 3) hit points
                mult[closer] = _eval_texture(prim.texture, pt)
            alb = np.asarray(prim.reflectance) * mult[..., None]
        else:
            alb = prim.reflectance
        albedo[sy, sx] = np.where(closer[..., None], alb, albedo[sy, sx])
        obj_id[sy, sx] = np.where(closer, k, obj_id[sy, sx])
    mask = np.isfinite(best_t)
    depth = np.where(mask, best_t, np.nan)
    # Orient world-space normals to face the camera (against the view rays), unit-length.
    flip = np.sum(normal * d_world, axis=-1) > 0
    normal = np.where(flip[..., None], -normal, normal)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, nrm, out=np.zeros_like(normal), where=nrm > 0)
    return GBuffer(depth=depth, normal=normal, albedo=albedo, obj_id=obj_id, mask=mask)


def _projector_depth(rig: Rig, prims: list) -> np.ndarray:
    """Projector-frame Z of the nearest surface along each projector pixel ray.

    Cast the projector's rays (origin = projector centre, directions rotated into
    the camera frame) through the same intersectors; ``inf`` where nothing is hit.
    Used as a shadow map.
    """
    centre = rig.C_proj                                  # projector centre in world
    proj_rays = camera_rays(rig.projector)               # (Hp, Wp, 3) in projector frame
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        d_world = proj_rays @ rig.R_proj                  # -> world directions
        hw = proj_rays.shape[:2]
        best_t = np.full(hw, np.inf)
        for prim in prims:
            win = _screen_window(prim, rig.projector, rig.R_proj, centre, hw)
            if win is None:
                continue
            sy, sx = win
            t, _, hit = _intersect(centre, d_world[sy, sx], prim)
            closer = hit & (t < best_t[sy, sx])
            best_t[sy, sx] = np.where(closer, t, best_t[sy, sx])
    # As for the camera, t along the world ray equals the projector-frame Z.
    return np.where(np.isfinite(best_t), best_t, np.inf)


# Single-entry memo so render_ground_truth and every render_capture in a run share
# the (expensive) G-buffer + projector depth without recomputing per pattern.
_CACHE: dict = {}


def _scene_buffers(rig: Rig, prims: list):
    key = (id(rig), id(prims))
    cached = _CACHE.get("k")
    if cached is None or _CACHE["k"] != key:
        _CACHE.clear()
        _CACHE["k"] = key
        _CACHE["gb"] = build_gbuffer(rig, prims)
        _CACHE["pd"] = _projector_depth(rig, prims)
        _CACHE["ref"] = (rig, prims)                     # keep objects alive
    return _CACHE["gb"], _CACHE["pd"]


# --------------------------------------------------------------------------
# Rendering (same interface as mitsuba_gen)
# --------------------------------------------------------------------------
def _print_render_stats(label, build_s, render_s, h, w, extra):
    print(f"  [raster] {label:<16s} {w}x{h} "
          f"build {build_s:5.3f}s shade {render_s:6.4f}s  {extra}")


def render_ground_truth(rig: Rig, geometry=None, spp: int = 0, *,
                        label: str | None = None, **kw):
    """Exact GT: camera-frame Z-depth (H, W) and world hit positions (H, W, 3).

    NaN where no surface is hit. ``spp`` is ignored (geometry is analytic/exact),
    kept for interface parity with the Mitsuba backend.
    """
    prims = geometry if geometry is not None else default_geometry()
    t0 = time.perf_counter()
    gb, _ = _scene_buffers(rig, prims)
    t1 = time.perf_counter()
    pos = gb.depth[..., None] * camera_rays(rig.camera)
    pos = np.where(gb.mask[..., None], pos, np.nan)
    if label is not None:
        h, w = gb.depth.shape
        zr = (f"{np.nanmin(gb.depth):.3f}-{np.nanmax(gb.depth):.3f}"
              if gb.mask.any() else "n/a")
        _print_render_stats(label, t1 - t0, 0.0, h, w,
                            f"hit={gb.mask.mean() * 100:4.0f}%  z={zr}")
    return gb.depth, pos


def projector_visible(rig: Rig, geometry=None, cast_shadows: bool = True) -> np.ndarray:
    """Per camera pixel: can the projector reach its surface point?

    Tests each camera-visible point against the **analytic projector-depth raycast**
    (the same one the captures' shadows use), so a smooth/curved surface does not
    self-shadow — unlike a depth-map z-buffer, which quantises camera points into
    projector pixels and produces shadow acne at grazing angles. Used both for the
    captures' ``lit`` mask and to mark occluded ``gt_proj`` correspondences invalid.
    """
    prims = geometry if geometry is not None else default_geometry()
    gb, proj_depth = _scene_buffers(rig, prims)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        pts_proj = (gb.depth[..., None] * camera_rays(rig.camera)) @ rig.R.T + rig.t
        ideal_uv = project(rig.projector, pts_proj)
    vis = gb.mask & _in_projector_fov(ideal_uv, rig.projector)
    if cast_shadows:
        Dp = _nearest_sample(proj_depth, ideal_uv)
        vis = vis & (pts_proj[..., 2] <= Dp + _SHADOW_BIAS)
    return vis


def _sample_any(pattern: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear pattern sample: (H,W) for grayscale, (H,W,3) for colour patterns."""
    if pattern.ndim == 3:
        return np.stack([sample_pattern(pattern[..., c], u, v) for c in range(3)], axis=-1)
    return sample_pattern(pattern, u, v)


def render_capture(rig: Rig, pattern: np.ndarray, *, label: str | None = None,
                   optics=None, geometry=None, cfg: RenderConfig | None = None,
                   spp: int = 1, **kw) -> np.ndarray:
    """Render one pattern-lit capture in [0, 1] via normal-aware Lambertian shading.

    Returns grayscale ``(H, W)`` for a grayscale pattern, colour ``(H, W, 3)`` for a
    colour one. ``optics`` applies the projector lens (distortion handled by sampling
    the pattern at the same coordinate ``gt_proj`` records). ``spp`` is unused.
    """
    cfg = cfg or RenderConfig()
    prims = geometry if geometry is not None else default_geometry()
    proj_optics = optics.projector if optics is not None else None

    t0 = time.perf_counter()
    gb, proj_depth = _scene_buffers(rig, prims)
    t1 = time.perf_counter()

    rays = camera_rays(rig.camera)
    pts_cam = gb.depth[..., None] * rays                 # (H, W, 3), NaN off-surface
    lit = projector_visible(rig, prims, cast_shadows=cfg.cast_shadows)

    # Pattern lookup: the SAME coordinate gt_proj records (distortion-authored).
    uv = projector_subpixel(rig, gb.depth, proj_optics=proj_optics)
    u, v = uv[..., 0], uv[..., 1]
    proj_val = _sample_any(np.clip(pattern, 0, 1), np.nan_to_num(u, nan=-1.0),
                           np.nan_to_num(v, nan=-1.0))

    # Normal-aware Lambertian, evaluated in the world frame (normals are world):
    # albedo·(ambient + gain·lit·pattern·max(N·L,0)·falloff), L toward the projector.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        pts_world = pts_cam @ rig.R_cam + rig.C_cam      # camera-frame hit -> world
    L = rig.C_proj - pts_world
    dist = np.linalg.norm(L, axis=-1, keepdims=True)
    Lhat = np.divide(L, dist, out=np.zeros_like(L), where=dist > 0)
    ndotl = np.clip(np.sum(gb.normal * Lhat, axis=-1), 0.0, None)
    ref = np.nanmedian(dist[gb.mask]) if gb.mask.any() else 1.0
    falloff = (ref / np.maximum(dist[..., 0], 1e-6)) ** 2

    drive = cfg.gain * np.where(lit, 1.0, 0.0) * ndotl * falloff   # (H, W)
    if proj_val.ndim == 3:
        signal = gb.albedo * (cfg.ambient + drive[..., None] * proj_val)
    else:
        signal = gb.albedo * (cfg.ambient + (drive * proj_val)[..., None])

    if cfg.spec_strength > 0:
        # GGX (Cook-Torrance) microfacet specular + Fresnel-Schlick, white (not albedo-
        # tinted, dielectric). At grazing angles Fresnel spikes, so the projector pattern
        # reflects strongly toward the camera — blowing highlights to white or killing
        # contrast. That failure physics is otherwise absent from the (Lambertian) training
        # distribution; this is the likely lever for the grazing real-capture collapse.
        # Closed-form per pixel, stays in the fast path.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            Vv = rig.C_cam - pts_world                            # toward the camera
            vd = np.linalg.norm(Vv, axis=-1, keepdims=True)
            Vhat = np.divide(Vv, vd, out=np.zeros_like(Vv), where=vd > 0)
            Hh = Lhat + Vhat
            hd = np.linalg.norm(Hh, axis=-1, keepdims=True)
            Hhat = np.divide(Hh, hd, out=np.zeros_like(Hh), where=hd > 0)
            ndotv = np.clip(np.sum(gb.normal * Vhat, axis=-1), 0.0, None)
            ndoth = np.clip(np.sum(gb.normal * Hhat, axis=-1), 0.0, None)
            vdoth = np.clip(np.sum(Vhat * Hhat, axis=-1), 0.0, None)
            a = max(cfg.roughness, 0.04) ** 2                     # GGX alpha = roughness^2
            a2 = a * a
            D = a2 / (np.pi * (ndoth * ndoth * (a2 - 1.0) + 1.0) ** 2 + 1e-9)
            F = cfg.spec_f0 + (1.0 - cfg.spec_f0) * (1.0 - vdoth) ** 5   # Fresnel grazing spike
            k = a * 0.5                                           # Schlick-GGX geometry term
            g1l = ndotl / (ndotl * (1.0 - k) + k + 1e-6)
            g1v = ndotv / (ndotv * (1.0 - k) + k + 1e-6)
            spec_brdf = D * F * g1l * g1v / (4.0 * ndotl * ndotv + 1e-6)
            spec_drive = (cfg.spec_strength * cfg.gain * np.where(lit, 1.0, 0.0)
                          * falloff * ndotl * spec_brdf)          # (H, W)
        spec_drive = np.nan_to_num(spec_drive, nan=0.0, posinf=0.0, neginf=0.0)
        spec_pat = spec_drive[..., None] * (proj_val if proj_val.ndim == 3 else proj_val[..., None])
        signal = np.clip(signal + spec_pat, 0.0, 1.0)            # explicit saturation (whiteout)

    color = np.ndim(pattern) == 3
    out = signal if color else signal.mean(axis=2)

    # Capture post-effects: depth of field (uses the sharp depth, so applied in the
    # ideal image space where it aligns) -> lens distortion -> bloom -> sensor noise.
    cam_optics = optics.camera if optics is not None else None
    if cam_optics is not None and cam_optics.has_dof:
        from .optics import apply_depth_of_field
        out = apply_depth_of_field(out, gb.depth, cam_optics, rig.camera.fx)
    if cam_optics is not None and cam_optics.has_distortion:
        from .optics import apply_distortion
        out = apply_distortion(out, rig.camera.K, cam_optics.dist)
    if optics is not None and optics.bloom.active:
        from .optics import apply_bloom
        out = apply_bloom(out, optics.bloom)
    rng = np.random.default_rng(cfg.seed)
    out = np.clip(add_sensor_noise(out, cfg, rng), 0.0, 1.0)
    t2 = time.perf_counter()
    if label is not None:
        h, w = out.shape[:2]
        lum = out.mean(axis=2) if color else out
        tag = "rgb " if color else ""
        _print_render_stats(label, t1 - t0, t2 - t1, h, w,
                            f"{tag}lum[min/mean/max]={lum.min():.3f}/{lum.mean():.3f}/{lum.max():.3f}")
    return out


_SHADOW_BIAS = 3e-3   # metres; tolerance against projector-depth discretisation


def _in_projector_fov(uv, proj):
    u, v = uv[..., 0], uv[..., 1]
    return (np.isfinite(u) & (u >= 0) & (u <= proj.width - 1)
            & (v >= 0) & (v <= proj.height - 1))


def _nearest_sample(buf: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lookup of a (Hp, Wp) buffer at continuous (u, v); inf outside."""
    h, w = buf.shape
    u = np.rint(np.nan_to_num(uv[..., 0], nan=-1.0)).astype(int)
    v = np.rint(np.nan_to_num(uv[..., 1], nan=-1.0)).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(uv.shape[:2], np.inf)
    out[ok] = buf[np.clip(v[ok], 0, h - 1), np.clip(u[ok], 0, w - 1)]
    return out
