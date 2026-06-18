#!/usr/bin/env python3
"""Generate a randomized training dataset with the NumPy ray-caster.

Each sample draws a random scene (mixed oriented boxes / spheres over a random
background, procedural textures) and a random rig (posed camera + projector,
varied baseline/FOV plus probabilistic lens & sensor imperfections), writes
both as ``scene.json`` / ``rig.json`` into the sample folder, then renders the
standard artifact set (GT depth, gt_proj, captures) there — so every sample is
self-documenting and exactly re-renderable with
``gen_rasterizer_dataset.py --scene <dir>/scene.json --rig <dir>/rig.json``.

Samples are seeded individually (sample i = ``--seed + i``) and a per-sample
``sample.json`` manifest is written last, so runs are reproducible, resumable
(rerun skips completed samples), extendable (continue with a higher --seed),
and splittable (disjoint seed ranges for train/val/test).

    python scripts/gen_training_data.py --n 100 --patterns patterns/marray --jobs 4
    python scripts/gen_training_data.py --n 20 --seed 100000 --out renders/val --lean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.geometry import Intrinsics, look_at_basis  # noqa: E402
from lux.datasets.raster_gen import (  # noqa: E402
    load_geometry, render_capture, render_ground_truth, projector_visible,
)
from lux.datasets.correspondence import projector_subpixel  # noqa: E402
from lux.datasets.rig_loader import build_rig  # noqa: E402
from lux.datasets.optics import parse_optics  # noqa: E402
from lux.render import RenderConfig  # noqa: E402

from gen_rasterizer_dataset import render_pattern_dir  # noqa: E402


# --------------------------------------------------------------------------
# Parameter sampling
# --------------------------------------------------------------------------
def _log_uniform(rng, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_reflectance(rng: np.random.Generator, lo: float = 0.3, hi: float = 0.95):
    """Gray scalar half the time, otherwise a chroma-jittered [r, g, b] triple."""
    base = rng.uniform(lo, hi)
    if rng.random() < 0.5:
        return round(base, 3)
    return [round(float(np.clip(base * rng.uniform(0.65, 1.35), 0.05, 1.0)), 3)
            for _ in range(3)]


def _sample_texture(rng: np.random.Generator) -> dict | None:
    """A procedural albedo texture on ~60% of surfaces (None = uniform albedo)."""
    if rng.random() < 0.4:
        return None
    kind = str(rng.choice(["checker", "stripes", "noise"], p=[0.25, 0.25, 0.5]))
    tex = {"type": kind,
           "scale": round(_log_uniform(rng, 0.01, 0.15), 4),
           "contrast": round(rng.uniform(0.2, 0.8), 3)}
    if kind == "stripes":
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        tex["dir"] = [round(float(x), 3) for x in d]
        tex["phase"] = round(rng.uniform(0, 2 * np.pi), 3)
    elif kind == "noise":
        tex["seed"] = int(rng.integers(0, 2**31))
    return tex


def _with_surface(obj: dict, rng: np.random.Generator, lo: float = 0.3) -> dict:
    obj["reflectance"] = _sample_reflectance(rng, lo=lo)
    tex = _sample_texture(rng)
    if tex is not None:
        obj["texture"] = tex
    return obj


def _cam_axis(cam_blk: dict | None):
    """Camera centre and unit forward direction (defaults: origin-ish rig)."""
    if cam_blk is None:
        return np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])
    C = np.asarray(cam_blk["position"], float)
    fwd = np.asarray(cam_blk["look_at"], float) - C
    return C, fwd / np.linalg.norm(fwd)


def _sample_background(rng: np.random.Generator, cam_blk: dict | None,
                       bg_dist: float) -> list:
    """Background surface ``bg_dist`` metres down the view axis: flat plane,
    tilted wall (thin rotated box), wavy relief, or nothing (depth NaN).

    Extents scale with the distance so the surface fills the frustum at room
    scale (half-extent ~= bg_dist covers a 55-deg HFOV with margin)."""
    C, fwd = _cam_axis(cam_blk)
    bc = C + fwd * bg_dist                       # view axis at background depth
    z = round(float(bc[2]), 3)
    s = round(bg_dist, 2)
    kind = rng.choice(["plane", "tilted", "wavy", "none"], p=[0.45, 0.25, 0.2, 0.1])
    if kind == "plane":
        return [_with_surface({"type": "plane", "z": z, "scale": s}, rng, lo=0.6)]
    if kind == "tilted":
        return [_with_surface({"type": "box",
                               "center": [round(float(bc[0]), 3), round(float(bc[1]), 3), z],
                               "scale": [s, s, 0.01],
                               "rotation": [round(rng.uniform(-20, 20), 1),
                                            round(rng.uniform(-20, 20), 1),
                                            round(rng.uniform(0, 360), 1)]}, rng, lo=0.6)]
    if kind == "wavy":
        return [_with_surface({"type": "wavy",
                               "center": [round(float(bc[0]), 3), round(float(bc[1]), 3), z],
                               "extent": [round(1.7 * bg_dist, 2), round(1.3 * bg_dist, 2)],
                               "amplitude": round(rng.uniform(0.05, 0.25), 3),
                               "freq": [round(rng.uniform(1.0, 5.0), 2),
                                        round(rng.uniform(1.0, 5.0), 2)],
                               "phase": [round(rng.uniform(0, 2 * np.pi), 3),
                                         round(rng.uniform(0, 2 * np.pi), 3)]}, rng, lo=0.6)]
    return []


def _center_sampler(rng: np.random.Generator, cam_blk: dict | None,
                    dmin: float, dmax: float):
    """Object-centre sampler: a uniform pixel (8% border margin) and a uniform
    camera depth in [dmin, dmax], unprojected to world — every object is in
    view *by construction* and image coverage is uniform at every depth.

    Returns a ``draw()`` yielding ``(center_xyz, depth)``; the depth lets the
    caller scale object sizes so angular size stays distance-independent.
    """
    if cam_blk is None:
        def draw():
            d = rng.uniform(dmin, dmax)
            return [round(rng.uniform(-0.4, 0.4) * d, 3),
                    round(rng.uniform(-0.25, 0.25) * d, 3),
                    round(d - 1.0, 3)], d
        return draw
    K = Intrinsics.from_fov(cam_blk["width"], cam_blk["height"], cam_blk["hfov_deg"])
    C = np.asarray(cam_blk["position"], float)
    R = look_at_basis(np.asarray(cam_blk["look_at"], float) - C,
                      np.asarray(cam_blk["up"], float))

    def draw():
        d = rng.uniform(dmin, dmax)
        u = rng.uniform(0.08, 0.92) * K.width
        v = rng.uniform(0.08, 0.92) * K.height
        pc = np.array([(u - K.cx) / K.fx * d, (v - K.cy) / K.fy * d, d])
        w = R.T @ pc + C                                  # camera frame -> world
        return [round(float(x), 3) for x in w], d
    return draw


def _sample_hemi_scene(rng: np.random.Generator, n_objects: int | None = None) -> dict:
    """Origin-centred cluttered scene for **hemisphere-posed** rigs: a LARGE ground plane through the
    origin (fills the frame from any pose, like the planar set's slabs, so an independently-posed
    camera + projector both see it) with mixed boxes/spheres scattered near the origin on/above it.
    The big plane gives dense supervision + the eval-matching surface obliquity; the objects give the
    depth/occlusion edges. Mirrors the frustum-filling trick of ``sample_planar_scene``."""
    if n_objects is None:
        n_objects = int(rng.integers(4, 19))
    H = 25.0                                              # ground half-extent (overflows any frustum)
    if rng.random() < 0.7:                                # flat ground at z=0
        ground = _with_surface({"type": "plane", "z": 0.0, "scale": H}, rng, lo=0.5)
    else:                                                 # gently tilted big plane (thin rotated box)
        ground = _with_surface({"type": "box", "center": [0.0, 0.0, 0.0], "scale": [H, H, 0.002],
                                "rotation": [round(rng.uniform(-20, 20), 1),
                                             round(rng.uniform(-20, 20), 1),
                                             round(rng.uniform(0, 360), 1)]}, rng, lo=0.5)
    objects = [ground]
    R = 0.3                                               # object centres within +/-R of the origin
    for _ in range(n_objects):
        cx, cy = round(rng.uniform(-R, R), 3), round(rng.uniform(-R, R), 3)
        if rng.random() < 0.65:
            s = [round(_log_uniform(rng, 0.03, 0.16), 4) for _ in range(3)]
            obj = {"type": "box", "center": [cx, cy, round(rng.uniform(0.0, 0.35) + s[2], 3)],
                   "scale": s, "rotation": [round(rng.uniform(0, 360), 1) for _ in range(3)]}
        else:
            r = round(_log_uniform(rng, 0.03, 0.11), 4)
            obj = {"type": "sphere", "center": [cx, cy, round(rng.uniform(0.0, 0.35) + r, 3)], "radius": r}
        objects.append(_with_surface(obj, rng))
    return {"name": "hemi_clutter_scene", "objects": objects,
            "ambient": round(rng.uniform(0.05, 0.15), 3)}


def sample_scene(rng: np.random.Generator, n_objects: int | None = None,
                 cam_blk: dict | None = None, origin_centered: bool = False) -> dict:
    """A random interior-scale scene: background 2-5 m down the view axis, a
    variable count of mixed objects floating 1 m to background in front.

    Boxes (oriented, per-axis log-uniform half-extents spanning an order of
    magnitude -> cubes, slabs and skinny rods) and spheres; sparse to cluttered.
    Sizes scale with the sampled camera depth so angular (pixel) size stays
    distance-independent. ``cam_blk`` (the sampled rig's camera block) drives
    the frustum placement so every object lands in view.

    ``origin_centered=True`` (for hemisphere-posed rigs) instead builds an origin-centred big-ground-
    plane scene (:func:`_sample_hemi_scene`) so an independently-posed cam+proj both see the surface.
    """
    if origin_centered:
        return _sample_hemi_scene(rng, n_objects)
    if n_objects is None:
        n_objects = int(rng.integers(4, 19))
    bg_dist = round(rng.uniform(2.0, 5.0), 3)    # background distance from camera
    objects = _sample_background(rng, cam_blk, bg_dist)
    draw_center = _center_sampler(rng, cam_blk, 1.0, bg_dist - 0.15)
    for _ in range(n_objects):
        center, d = draw_center()
        if rng.random() < 0.65:
            obj = {"type": "box", "center": center,
                   "scale": [round(d * _log_uniform(rng, 0.015, 0.1), 4) for _ in range(3)],
                   "rotation": [round(rng.uniform(0, 360), 1) for _ in range(3)]}
        else:
            obj = {"type": "sphere", "center": center,
                   "radius": round(d * _log_uniform(rng, 0.02, 0.08), 4)}
        objects.append(_with_surface(obj, rng))
    return {"name": "random_scene",
            "objects": objects,
            "ambient": round(rng.uniform(0.05, 0.15), 3)}


# --------------------------------------------------------------------------
# Hemisphere pose sampling (shared with gen_planar_dataset, which imports these)
# --------------------------------------------------------------------------
def _tilt_deg(rng: np.random.Generator, max_tilt: float, grazing_frac: float) -> float:
    """Off-normal tilt in degrees. With prob ``grazing_frac`` draw from the cliff band
    ``[45, max_tilt]`` (oversampling grazing), else uniform ``[0, max_tilt]``."""
    if grazing_frac > 0.0 and rng.random() < grazing_frac:
        return float(rng.uniform(min(45.0, max_tilt), max_tilt))
    return float(rng.uniform(0.0, max_tilt))


def _hemisphere_dir(rng: np.random.Generator, max_tilt_deg: float,
                    grazing_frac: float = 0.0) -> np.ndarray:
    """A unit direction on the +Z hemisphere: azimuth uniform in [0, 2pi), tilt off the +Z
    normal per :func:`_tilt_deg` (uniform tilt, optionally grazing-oversampled)."""
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = np.radians(_tilt_deg(rng, max_tilt_deg, grazing_frac))
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def _rolled_up(rng: np.random.Generator, forward: np.ndarray) -> np.ndarray:
    """A world up-vector giving a uniformly random roll about ``forward`` (image-top direction)."""
    f = forward / np.linalg.norm(forward)
    ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(ref, f)
    right /= np.linalg.norm(right)
    up0 = np.cross(right, f)                      # right-handed up, perp to f
    a = rng.uniform(0.0, 2.0 * np.pi)
    return np.cos(a) * up0 + np.sin(a) * right


def _hemi_pose(rng: np.random.Generator, max_tilt: float, dmin: float, dmax: float,
               grazing_frac: float):
    """A camera/projector centre on the +Z hemisphere aimed at the origin, with random roll.
    Returns ``(centre, up)``; tilt oversamples grazing per ``grazing_frac``."""
    n = _hemisphere_dir(rng, max_tilt, grazing_frac)
    C = rng.uniform(dmin, dmax) * n
    return C, _rolled_up(rng, -C)


def sample_rig(rng: np.random.Generator, width: int, height: int,
               cam_distort: bool = False, independent_proj: bool = False,
               grazing_frac: float = 0.0, max_tilt: float = 0.0,
               dmin: float = 0.9, dmax: float = 2.0) -> dict:
    """A posed rig aimed near the origin, then probabilistic lens/sensor imperfections layered on.

    Default (``max_tilt=0``, no flags): the legacy near-frontal rig -- camera jittered around
    (0,0,-1m), projector at a small fixed baseline. With ``max_tilt>0`` the camera is posed on the
    hemisphere (tilt oversampled toward grazing per ``grazing_frac``); with ``independent_proj`` the
    projector is posed independently on the hemisphere too, so ``max(cam,proj)`` obliquity skews
    toward grazing the way the hemisphere eval does (instead of proj~cam). The scene is built in the
    camera frustum by the caller, so it follows whatever pose the camera takes."""
    up_p = None
    if independent_proj or grazing_frac > 0.0 or max_tilt > 0.0:
        cam_C, up = _hemi_pose(rng, max(max_tilt, 1.0), dmin, dmax, grazing_frac)
        if independent_proj:
            proj_C, up_p = _hemi_pose(rng, max(max_tilt, 1.0), dmin, dmax, grazing_frac)
        else:                                          # rigid baseline along the camera's right axis
            f = -cam_C / np.linalg.norm(cam_C)
            y = (up @ f) * f - up; y /= np.linalg.norm(y); right = np.cross(y, f)
            baseline = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0])
            proj_C = (cam_C + baseline * right + rng.uniform(-0.08, 0.08) * y
                      + rng.uniform(-0.06, 0.06) * f)
        target = [round(rng.uniform(-0.08, 0.08), 3), round(rng.uniform(-0.08, 0.08), 3), 0.0]
        cam_pos = [round(float(x), 4) for x in cam_C]
        proj_pos = [round(float(x), 4) for x in proj_C]
        up = [round(float(x), 4) for x in up]
        up_p = up if up_p is None else [round(float(x), 4) for x in up_p]
    else:                                              # legacy near-frontal rig (original draw order)
        cam_pos = [round(rng.uniform(-0.15, 0.15), 3), round(rng.uniform(-0.15, 0.15), 3),
                   round(-rng.uniform(0.85, 1.25), 3)]
        target = [round(rng.uniform(-0.08, 0.08), 3), round(rng.uniform(-0.08, 0.08), 3), 0.0]
        baseline = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0])
        proj_pos = [round(cam_pos[0] + baseline, 3), round(cam_pos[1] + rng.uniform(-0.12, 0.12), 3),
                    round(cam_pos[2] + rng.uniform(-0.1, 0.1), 3)]
        up = up_p = [0.0, -1.0, 0.0]
    camera = {"width": width, "height": height, "hfov_deg": round(rng.uniform(38.0, 55.0), 2),
              "position": cam_pos, "look_at": target, "up": up}
    projector = {"width": 1920, "height": 1080, "hfov_deg": round(rng.uniform(35.0, 50.0), 2),
                 "position": proj_pos, "look_at": target, "up": up_p}
    rig = {"name": "random_rig", "camera": camera, "projector": projector}
    return add_rig_imperfections(rig, rng, cam_distort=cam_distort)


def add_rig_imperfections(rig: dict, rng: np.random.Generator,
                          cam_distort: bool = False) -> dict:
    """Sprinkle probabilistic lens/sensor imperfections (domain randomization)
    onto a posed rig so a set spans clean -> ugly: depth of field, projector
    distortion, bloom, sensor noise (and opt-in camera distortion). Mutates and
    returns ``rig`` (``camera``/``projector`` blocks must already exist). Shared
    by every randomized generator so the imperfection mix stays consistent.

    Camera distortion is opt-in (``cam_distort``) because it warps the captures
    out of the ideal image space the GT lives in (undistort downstream to
    realign); everything else is pixel-aligned.
    """
    camera, projector = rig["camera"], rig["projector"]
    if rng.random() < 0.3:   # depth of field: focus somewhere in the scene's depth band
        camera["aperture_radius"] = round(rng.uniform(0.0, 0.02), 4)
        camera["focus_distance"] = round(_log_uniform(rng, 1.2, 4.0), 3)
    if rng.random() < 0.5:   # projector lens distortion (gt_proj stays exact: it
        projector["distortion"] = {                # records authored coordinates)
            "k1": round(rng.uniform(-0.12, 0.08), 4),
            "k2": round(rng.uniform(-0.02, 0.02), 4),
            "p1": round(rng.uniform(-0.004, 0.004), 4),
            "p2": round(rng.uniform(-0.004, 0.004), 4)}
    if cam_distort and rng.random() < 0.5:
        camera["distortion"] = {
            "k1": round(rng.uniform(-0.2, 0.05), 4),
            "k2": round(rng.uniform(-0.04, 0.04), 4),
            "p1": round(rng.uniform(-0.004, 0.004), 4),
            "p2": round(rng.uniform(-0.004, 0.004), 4)}
    if rng.random() < 0.3:
        rig["bloom"] = {"threshold": round(rng.uniform(0.6, 0.9), 3),
                        "intensity": round(rng.uniform(0.2, 1.0), 3),
                        "radius": round(rng.uniform(4.0, 12.0), 1)}
    if rng.random() < 0.8:   # sensor noise on most samples; seed omitted -> fresh grain
        rig["noise"] = {"read": round(rng.uniform(0.0, 0.03), 4),
                        "shot": round(rng.uniform(0.0, 0.05), 4),
                        "blue": round(rng.uniform(0.0, 0.03), 4)}
    if rng.random() < 0.5:   # glossy surface on ~half the samples: GGX specular puts the
        rig["material"] = {  # grazing-angle Fresnel whiteout / contrast-kill physics into
            "spec": round(rng.uniform(0.2, 1.8), 3),       # training (else absent -> the
            "roughness": round(rng.uniform(0.08, 0.6), 3),  # likely grazing real-capture cliff)
            "f0": round(rng.uniform(0.02, 0.10), 3)}
    return rig


# --------------------------------------------------------------------------
# Rendering (same artifact set as gen_rasterizer_dataset.py)
# --------------------------------------------------------------------------
def render_sample(scene_path: Path, rig_path: Path, pattern_dirs: list[str],
                  sdir: str, lean: bool = False) -> None:
    rig_spec = json.loads(rig_path.read_text())
    rig = build_rig(rig_spec)
    optics = parse_optics(rig_spec)
    nz = rig_spec.get("noise", {})
    mat = rig_spec.get("material", {})
    amb = json.loads(scene_path.read_text()).get("ambient", 0.05)
    cfg = RenderConfig(ambient=amb,
                       read_noise=float(nz.get("read", 0.0)),
                       shot_noise=float(nz.get("shot", 0.0)),
                       blue_noise=float(nz.get("blue", 0.0)),
                       spec_strength=float(mat.get("spec", 0.0)),
                       roughness=float(mat.get("roughness", 0.35)),
                       spec_f0=float(mat.get("f0", 0.04)),
                       seed=nz.get("seed", None))
    clean = RenderConfig(ambient=amb, read_noise=0.0, shot_noise=0.0, blue_noise=0.0)
    geo = load_geometry(str(scene_path))

    gt, _ = render_ground_truth(rig, geometry=geo, label="ground-truth")
    white = render_capture(rig, np.ones((rig.projector.height, rig.projector.width), np.float32),
                           geometry=geo, cfg=clean, label="white-ref")

    gt_proj = projector_subpixel(rig, gt, proj_optics=optics.projector)
    gt_proj = np.where(projector_visible(rig, geo)[..., None], gt_proj, np.nan)

    # float32: ~1e-4 px / ~0.1 um precision at this scale, half the disk of float64
    io.save_npy(os.path.join(sdir, "gt_depth.npy"), gt.astype(np.float32))
    io.save_npy(os.path.join(sdir, "gt_proj.npy"), gt_proj.astype(np.float32))
    io.save_image(os.path.join(sdir, "white.png"), white)
    if not lean:  # human-facing extras a training dataloader never reads
        albedo = np.clip(white / max(white.max(), 1e-6), 0, 1)
        io.save_image(os.path.join(sdir, "gt_proj.png"),
                      io.proj_to_rgb(gt_proj, rig.projector.width, rig.projector.height))
        io.save_image(os.path.join(sdir, "albedo.png"), albedo)
        pts, col = io.depth_to_points(gt, rig, albedo)
        io.save_ply(os.path.join(sdir, "gt_cloud.ply"), pts, col)

    # All sets share the cached G-buffer, so extra pattern sets only pay shading.
    for pd in pattern_dirs:
        render_pattern_dir(rig, geo, pd, sdir, optics=optics, cfg=cfg, montage=not lean)


def _render_one(i: int, args) -> str:
    """Render sample ``i`` (worker-safe). Returns a one-line status."""
    seed = args.seed + i
    sdir = os.path.join(args.out, f"sample_{seed:05d}")
    manifest = Path(sdir, "sample.json")
    if manifest.exists() and not args.overwrite:
        return f"sample_{seed:05d}: exists, skipped"

    rng = np.random.default_rng(seed)
    io.ensure_dir(sdir)
    scene_path, rig_path = Path(sdir, "scene.json"), Path(sdir, "rig.json")
    # Rig first: the scene sampler rejection-tests object centres against the
    # sampled camera frustum so nothing lands out of view.
    rig_spec = sample_rig(rng, args.width, args.height, cam_distort=args.cam_distort,
                          independent_proj=args.independent_proj, grazing_frac=args.grazing_frac,
                          max_tilt=args.max_tilt)
    hemi = args.max_tilt > 0.0 or args.independent_proj   # origin-centred scene for hemisphere rigs
    scene = sample_scene(rng, args.objects, cam_blk=rig_spec["camera"], origin_centered=hemi)
    rig_path.write_text(json.dumps(rig_spec, indent=2) + "\n")
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")

    print(f"=== sample seed {seed} -> ./{sdir}/ ===")
    render_sample(scene_path, rig_path, args.patterns, sdir, lean=args.lean)

    # Written last -> doubles as the completion marker for --overwrite-less resume.
    manifest.write_text(json.dumps({
        "seed": seed,
        "patterns": [Path(p).name for p in args.patterns],
        "camera": [args.width, args.height],
        "n_objects": len(scene["objects"]),
        "lean": args.lean,
    }, indent=2) + "\n")
    return f"sample_{seed:05d}: done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", nargs="+", default=["patterns/marray"],
                    help="one or more folders of PNG patterns; each set is rendered "
                         "into its own subfolder per sample (G-buffer is shared, so "
                         "extra sets are cheap)")
    ap.add_argument("--n", type=int, default=10, help="number of samples to render")
    ap.add_argument("--objects", type=int, default=None,
                    help="objects per scene (default: random 4-18 per sample)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed; sample i uses seed+i, so a run is reproducible "
                         "and extendable (--seed 0 --n 100 then --seed 100 --n 100); "
                         "use a disjoint seed range for a val/test split")
    ap.add_argument("--width", type=int, default=1920, help="camera width")
    ap.add_argument("--height", type=int, default=1080, help="camera height")
    ap.add_argument("--cam-distort", action="store_true",
                    help="also randomize camera lens distortion (warps captures out of "
                         "the ideal image space GT lives in; undistort downstream)")
    ap.add_argument("--max-tilt", type=float, default=0.0,
                    help="max rig tilt off the scene's frontal axis, degrees (0 = legacy "
                         "near-frontal rig; >0 = hemisphere-posed camera, covering oblique views)")
    ap.add_argument("--independent-proj", action="store_true",
                    help="pose the projector independently on the hemisphere (not rigidly offset "
                         "from the camera), so max(cam,proj) obliquity matches the hemisphere eval")
    ap.add_argument("--grazing-frac", type=float, default=0.0,
                    help="fraction of camera/projector poses drawn from the >=45 deg cliff band "
                         "(oversamples grazing; requires --max-tilt)")
    ap.add_argument("--lean", action="store_true",
                    help="skip human-facing extras (gt_cloud.ply, montage, gt_proj.png, "
                         "albedo.png) - keeps gt_depth/gt_proj/white/captures only")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render samples that already have a sample.json (default: "
                         "skip them, so an interrupted run resumes where it stopped)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (samples are independent)")
    ap.add_argument("--maxtasks", type=int, default=8,
                    help="recycle each worker process after this many samples. The NumPy "
                         "ray-caster's peak RSS is not returned to the OS, so a long-lived pool "
                         "worker ratchets to multi-GB and (xN jobs) OOMs the box; recycling resets it.")
    ap.add_argument("--out", default="renders/train")
    args = ap.parse_args()

    if args.jobs > 1:
        from multiprocessing import Pool
        from functools import partial
        # maxtasksperchild bounds memory: a recycled worker frees its NumPy high-water-mark
        # RSS back to the OS. imap_unordered streams (progress + low driver memory).
        with Pool(processes=args.jobs, maxtasksperchild=args.maxtasks) as pool:
            results = list(pool.imap_unordered(partial(_render_one, args=args), range(args.n)))
    else:
        results = [_render_one(i, args) for i in range(args.n)]

    done = sum(r.endswith("done") for r in results)
    skipped = len(results) - done
    print(f"\n{done} rendered, {skipped} skipped -> ./{args.out}/ "
          f"(seeds {args.seed}..{args.seed + args.n - 1})")


if __name__ == "__main__":
    main()
