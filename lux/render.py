"""Forward renderer: turn (scene, projector patterns) into captured images.

This is the analytic structured-light simulator. For each camera pixel that
sees a surface point, we:

  1. Back-project to the 3D point using the GT depth and camera ray.
  2. Project that point into the projector to find which pattern texel lit it.
  3. Sample the pattern, modulate by surface albedo, add ambient + sensor noise.

It also resolves **occlusion / projector shadows**: a surface point that the
projector cannot see (because a nearer point along the same projector ray
blocks it) receives no pattern light, only ambient. This is the main source of
"invalid" pixels real decoders must cope with, so it is on by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Rig, camera_rays, project
from .scene import Scene


@dataclass
class RenderConfig:
    ambient: float = 0.05          # constant illumination floor, fraction of full scale
    gain: float = 0.9              # projector contribution scale
    read_noise: float = 0.01       # additive white Gaussian sigma (fraction of full scale)
    shot_noise: float = 0.0        # Poisson-like photon noise scale (0 disables)
    blue_noise: float = 0.0        # high-frequency (blue-spectrum) grain sigma (0 disables)
    cast_shadows: bool = True      # mask points the projector cannot see
    seed: int | None = 0           # RNG seed for noise; None = fresh entropy (random)


def blue_noise_field(shape, sigma: float, rng) -> np.ndarray:
    """A zero-mean blue-noise field of std ``sigma`` over ``shape`` (H, W[, C]).

    White Gaussian noise spectrally tilted toward high frequencies (amplitude
    weighted by radial frequency), so the grain is spatially decorrelated with no
    low-frequency clumping — perceptually cleaner than white noise. Channels are
    drawn independently.
    """
    if len(shape) == 3:
        return np.stack([blue_noise_field(shape[:2], sigma, rng) for _ in range(shape[2])], axis=-1)
    h, w = shape
    white = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)            # 0 at DC, grows with frequency
    bn = np.fft.ifft2(np.fft.fft2(white) * radius).real
    s = bn.std()
    return bn * (sigma / s) if s > 0 else bn


def add_sensor_noise(img: np.ndarray, cfg: "RenderConfig", rng) -> np.ndarray:
    """Add shot + read (white) + blue-noise grain to an image (not clipped)."""
    out = img
    if cfg.shot_noise > 0:
        out = out + rng.normal(0.0, cfg.shot_noise * np.sqrt(np.maximum(out, 0)), out.shape)
    if cfg.read_noise > 0:
        out = out + rng.normal(0.0, cfg.read_noise, out.shape)
    if cfg.blue_noise > 0:
        out = out + blue_noise_field(out.shape, cfg.blue_noise, rng)
    return out


@dataclass
class Capture:
    """Output of a render: the image stack plus GT bookkeeping for evaluation."""

    images: np.ndarray        # (N, H, W) float in [0, 1], N = number of patterns
    lit_mask: np.ndarray      # (H, W) bool: surface AND reached by projector
    gt_proj_col: np.ndarray   # (H, W) float: GT projector column per pixel (NaN off-surface)
    scene: Scene
    rig: Rig


def _projector_coords(scene: Scene, rig: Rig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each camera pixel to its projector pixel coords given GT depth."""
    rays = camera_rays(rig.camera)                 # (H, W, 3)
    pts_cam = scene.depth[..., None] * rays        # (H, W, 3)
    pts_proj = pts_cam @ rig.R.T + rig.t           # (H, W, 3)
    uv = project(rig.projector, pts_proj)          # (H, W, 2)
    u_p, v_p = uv[..., 0], uv[..., 1]
    in_fov = (
        scene.mask
        & np.isfinite(u_p)
        & (u_p >= 0) & (u_p <= rig.projector.width - 1)
        & (v_p >= 0) & (v_p <= rig.projector.height - 1)
    )
    return u_p, v_p, in_fov


def _shadow_mask(scene: Scene, rig: Rig, u_p: np.ndarray, v_p: np.ndarray, in_fov: np.ndarray) -> np.ndarray:
    """Z-buffer in the projector frame: keep only the nearest surface per projector texel.

    Distance to the projector is approximated by Z in the projector frame, which
    is monotonic enough for shadow ordering in this pinhole setup.
    """
    H, W = scene.depth.shape
    rays = camera_rays(rig.camera)
    pts_proj_z = (scene.depth[..., None] * rays @ rig.R.T + rig.t)[..., 2]

    reachable = np.zeros((H, W), dtype=bool)
    pu = np.round(u_p).astype(int)
    pv = np.round(v_p).astype(int)
    # Nearest projector-Z wins each projector texel.
    best = {}
    ys, xs = np.where(in_fov)
    for y, x in zip(ys.tolist(), xs.tolist()):
        key = (pv[y, x], pu[y, x])
        z = pts_proj_z[y, x]
        cur = best.get(key)
        if cur is None or z < cur[0]:
            best[key] = (z, y, x)
    for _, y, x in best.values():
        reachable[y, x] = True
    return reachable


def sample_pattern(pattern: np.ndarray, u_p: np.ndarray, v_p: np.ndarray) -> np.ndarray:
    """Bilinearly sample a projector pattern at (u_p, v_p), zero outside."""
    H, W = pattern.shape
    u0 = np.floor(u_p).astype(int)
    v0 = np.floor(v_p).astype(int)
    fu = u_p - u0
    fv = v_p - v0

    def at(uu, vv):
        ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        out = np.zeros_like(u_p, dtype=np.float64)
        out[ok] = pattern[np.clip(vv[ok], 0, H - 1), np.clip(uu[ok], 0, W - 1)]
        return out

    c = (
        at(u0, v0) * (1 - fu) * (1 - fv)
        + at(u0 + 1, v0) * fu * (1 - fv)
        + at(u0, v0 + 1) * (1 - fu) * fv
        + at(u0 + 1, v0 + 1) * fu * fv
    )
    return c


def render(scene: Scene, patterns: np.ndarray, rig: Rig, cfg: RenderConfig | None = None) -> Capture:
    """Render a stack of projector patterns into a camera image stack.

    Parameters
    ----------
    patterns : (N, Hp, Wp) float in [0, 1]
        Projector patterns to display, one per captured frame.
    """
    cfg = cfg or RenderConfig()
    rng = np.random.default_rng(cfg.seed)
    patterns = np.asarray(patterns, dtype=np.float64)
    if patterns.ndim == 2:
        patterns = patterns[None]

    u_p, v_p, in_fov = _projector_coords(scene, rig)
    lit = in_fov
    if cfg.cast_shadows:
        lit = lit & _shadow_mask(scene, rig, u_p, v_p, in_fov)

    H, W = scene.depth.shape
    images = np.empty((patterns.shape[0], H, W), dtype=np.float64)
    for i, pat in enumerate(patterns):
        proj_val = sample_pattern(pat, u_p, v_p)
        signal = scene.albedo * (cfg.ambient + cfg.gain * np.where(lit, proj_val, 0.0))
        images[i] = np.clip(add_sensor_noise(signal, cfg, rng), 0.0, 1.0)

    gt_col = np.where(lit, u_p, np.nan)
    return Capture(images=images, lit_mask=lit, gt_proj_col=gt_col, scene=scene, rig=rig)
