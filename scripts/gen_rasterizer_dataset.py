#!/usr/bin/env python3
"""Render a ground-truth structured-light dataset with the NumPy ray-caster.

A fast analytic alternative to ``gen_mitsuba_dataset.py`` (no path tracer): it
casts the exact camera rays through scene primitives, so the ground-truth
projector subpixel is exact *by construction* with the rendered captures. Same
file-driven inputs and ``renders/`` output layout as the Mitsuba script:

  * ``--scene``    the geometry in front of the rig   (lux/datasets/scenes/*.json)
  * ``--rig``      the camera + projector parameters   (lux/datasets/rigs/*.json)
  * ``--patterns`` a folder of PNGs projected in order (bring your own)

    python scripts/gen_rasterizer_dataset.py --scene wavy --patterns patterns/graycode
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.datasets.raster_gen import (  # noqa: E402
    load_geometry, render_capture, render_ground_truth,
)
from lux.datasets.correspondence import projector_subpixel  # noqa: E402
from lux.datasets.scene_loader import list_scenes  # noqa: E402
from lux.datasets.rig_loader import load_rig_spec, build_rig, list_rigs  # noqa: E402
from lux.datasets.optics import parse_optics  # noqa: E402
from lux.render import RenderConfig  # noqa: E402


def render_pattern_dir(rig, geo, patterns_dir, sdir, optics=None, cfg=None):
    """Project a user-supplied folder of PNG patterns and save the captures.

    Patterns are projected in filename order; each capture is written as
    ``cap_<pattern-stem>.png``. Pattern images may be any resolution (the
    projector treats them as a texture across its FOV). ``optics`` applies the
    projector lens model (distortion) to each capture.
    """
    files = sorted(p for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg")
                   for p in Path(patterns_dir).glob(ext))
    if not files:
        raise SystemExit(f"no PNG/JPG patterns found in {patterns_dir!r}")

    set_name = Path(patterns_dir).name
    odir = io.ensure_dir(os.path.join(sdir, set_name))
    print(f"[{set_name}] projecting {len(files)} supplied patterns ...")
    caps = []
    for i, f in enumerate(files):
        # Load as RGB, but collapse to grayscale when the PNG carries no colour
        # so monochrome sets stay (H, W) and only true colour patterns go (H, W, 3).
        rgb = io.load_image(str(f), gray=False)
        pat = rgb[..., 0] if np.allclose(rgb[..., 0], rgb[..., 1]) and \
            np.allclose(rgb[..., 1], rgb[..., 2]) else rgb
        # Vary the noise seed per frame so the grain is independent across captures
        # (a fixed base seed stays reproducible but decorrelated; None = fully random).
        fcfg = cfg
        if cfg is not None:
            fcfg = replace(cfg, seed=None if cfg.seed is None else cfg.seed + i)
        cap = render_capture(rig, pat, geometry=geo, optics=optics, cfg=fcfg,
                             label=f"{set_name} {i + 1:02d}/{len(files)} {f.name}")
        io.save_image(os.path.join(odir, f"cap_{f.stem}.png"), cap)
        caps.append(cap)
    io.save_image(os.path.join(odir, "captures_montage.png"), io.montage(np.stack(caps)))
    print(f"wrote {len(caps)} captures to ./{odir}/cap_*.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", required=True,
                    help="folder of PNG patterns to project, in filename order; "
                         "each is rendered to cap_<filename>.png")
    ap.add_argument("--scene", default="blocks",
                    help=f"built-in scene name {list_scenes()} or a path to a .json scene file")
    ap.add_argument("--name", default=None,
                    help="output folder name under --out (defaults to raster_<scene>)")
    ap.add_argument("--rig", default="default",
                    help=f"built-in rig name {list_rigs()} or a path to a .json rig file")
    ap.add_argument("--out", default="renders")
    args = ap.parse_args()

    # The rig file owns all camera/projector parameters, including the optional
    # projector distortion; camera DoF/distortion are ignored by the ray-caster.
    rig_spec = load_rig_spec(args.rig)
    rig = build_rig(rig_spec)
    optics = parse_optics(rig_spec)
    cam, proj = rig.camera, rig.projector
    if "position" in rig_spec.get("camera", {}):
        print(f"rig '{args.rig}': camera {cam.width}x{cam.height}@{rig.C_cam.round(3).tolist()}  "
              f"projector {proj.width}x{proj.height}@{rig.C_proj.round(3).tolist()}")
    else:
        print(f"rig '{args.rig}': camera {cam.width}x{cam.height}  "
              f"projector {proj.width}x{proj.height}  baseline {rig_spec.get('baseline')}m")
    # Sensor noise lives in an optional rig-file "noise" block; the captures get it
    # (plus camera lens distortion), but the white-ref / GT stay clean and undistorted.
    nz = rig_spec.get("noise", {})
    # seed defaults to None -> noise is freshly random per frame; set "seed" in the
    # noise block for a reproducible (but still per-frame decorrelated) realization.
    cfg = RenderConfig(read_noise=float(nz.get("read", 0.0)),
                       shot_noise=float(nz.get("shot", 0.0)),
                       blue_noise=float(nz.get("blue", 0.0)),
                       seed=nz.get("seed", None))
    clean = RenderConfig(read_noise=0.0, shot_noise=0.0, blue_noise=0.0)
    if optics.projector.has_distortion:
        print(f"  projector distortion dist={optics.projector.dist}")
    if optics.camera.has_distortion:
        print(f"  camera distortion dist={optics.camera.dist}")
    if optics.camera.has_dof:
        print(f"  depth of field: aperture={optics.camera.aperture_radius} "
              f"focus={optics.camera.focus_distance}m")
    if optics.bloom.active:
        print(f"  bloom: threshold={optics.bloom.threshold} "
              f"intensity={optics.bloom.intensity} radius={optics.bloom.radius}px")
    if cfg.read_noise or cfg.shot_noise or cfg.blue_noise:
        print(f"  noise: read={cfg.read_noise} shot={cfg.shot_noise} blue={cfg.blue_noise}")

    geo = load_geometry(args.scene)
    scene_name = args.name or f"raster_{Path(args.scene).stem}"

    print(f"scene '{args.scene}' -> ./{args.out}/{scene_name}/")
    print("rendering ground truth + albedo ...")
    gt, _ = render_ground_truth(rig, geometry=geo, label="ground-truth")
    white = render_capture(rig, np.ones((rig.projector.height, rig.projector.width), np.float32),
                           geometry=geo, cfg=clean, label="white-ref")
    albedo = np.clip(white / max(white.max(), 1e-6), 0, 1)

    # The fundamental SL ground truth: exact projector subpixel (col, row) that lit
    # each camera pixel, from GT depth + calibration. Depth is the same information
    # triangulated, so we keep both (gt_depth feeds the cloud/viewer).
    gt_proj = projector_subpixel(rig, gt, proj_optics=optics.projector)

    sdir = io.ensure_dir(os.path.join(args.out, scene_name))
    io.save_npy(os.path.join(sdir, "gt_depth.npy"), gt)
    io.save_npy(os.path.join(sdir, "gt_proj.npy"), gt_proj)
    io.save_image(os.path.join(sdir, "gt_proj.png"),
                  io.proj_to_rgb(gt_proj, rig.projector.width, rig.projector.height))
    io.save_image(os.path.join(sdir, "white.png"), white)
    io.save_image(os.path.join(sdir, "albedo.png"), albedo)
    pts, col = io.depth_to_points(gt, rig, albedo)
    io.save_ply(os.path.join(sdir, "gt_cloud.ply"), pts, col)

    render_pattern_dir(rig, geo, args.patterns, sdir, optics=optics, cfg=cfg)
    print(f"\nwrote GT + captures to ./{args.out}/{scene_name}/ (viewable in the synthetic 3D tab)")


if __name__ == "__main__":
    main()
