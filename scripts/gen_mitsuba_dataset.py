#!/usr/bin/env python3
"""Render a ground-truth structured-light dataset with Mitsuba 3.

Projects a user-supplied sequence of patterns through a calibrated projector and
renders, per frame, the pattern-lit camera capture, plus the exact ground-truth
depth (position AOV). Everything is fed in by file:

  * ``--scene``    the geometry in front of the rig   (lux/datasets/scenes/*.json)
  * ``--rig``      the camera + projector parameters   (lux/datasets/rigs/*.json)
  * ``--patterns`` a folder of PNGs projected in order (bring your own)

Patterns are arbitrary, so there is no baked-in decoding strategy and no
scoring; the script just produces the rendered captures + GT in the ``renders/``
layout the synthetic 3D viewer reads.

    python scripts/gen_mitsuba_dataset.py --scene wavy --patterns ./my_patterns --spp 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.datasets.mitsuba_gen import (  # noqa: E402
    load_geometry, render_capture, render_ground_truth, projector_subpixel,
)
from lux.datasets.raster_gen import projector_visible  # noqa: E402
from lux.datasets.scene_loader import (  # noqa: E402
    list_scenes, load_scene_primitives, load_scene_ambient,
)
from lux.datasets.rig_loader import load_rig_spec, build_rig, list_rigs  # noqa: E402
from lux.datasets.optics import parse_optics  # noqa: E402


def render_pattern_dir(rig, geo, patterns_dir, sdir, spp, optics=None, ambient=0.04):
    """Project a user-supplied folder of PNG patterns and save the captures.

    Patterns are projected in filename order; each capture is written as
    ``cap_<pattern-stem>.png``. Pattern images may be any resolution (the
    projector treats them as a texture across its FOV). ``optics`` applies the
    camera + projector lens model (DoF + distortion) to each capture; ``ambient``
    sets the constant environment light.
    """
    files = sorted(p for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg")
                   for p in Path(patterns_dir).glob(ext))
    if not files:
        raise SystemExit(f"no PNG/JPG patterns found in {patterns_dir!r}")

    set_name = Path(patterns_dir).name
    odir = io.ensure_dir(os.path.join(sdir, set_name))
    print(f"[{set_name}] projecting {len(files)} supplied patterns (spp={spp}) ...")
    caps = []
    for i, f in enumerate(files):
        # Load as RGB, but collapse to grayscale when the PNG carries no colour
        # so monochrome sets stay (H, W) and only true colour patterns go (H, W, 3).
        rgb = io.load_image(str(f), gray=False)
        pat = rgb[..., 0] if np.allclose(rgb[..., 0], rgb[..., 1]) and \
            np.allclose(rgb[..., 1], rgb[..., 2]) else rgb
        cap = render_capture(rig, pat, geometry=geo, spp=spp, optics=optics, ambient=ambient,
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
                    help="output folder name under --out (defaults to the scene name)")
    ap.add_argument("--rig", default="default",
                    help=f"built-in rig name {list_rigs()} or a path to a .json rig file")
    ap.add_argument("--spp", type=int, default=24, help="samples/pixel for the pattern captures")
    ap.add_argument("--gt-spp", type=int, default=32,
                    help="samples/pixel for the GT depth pass; the position AOV is geometric, "
                         "so this only sets edge antialiasing and need not track --spp")
    ap.add_argument("--out", default="renders")
    args = ap.parse_args()

    # The rig file owns all camera/projector parameters, including the optional
    # lens model (depth of field + distortion); absent fields mean ideal pinhole.
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
    if optics.active:
        c, p = optics.camera, optics.projector
        print(f"  lens: cam aperture={c.aperture_radius} focus={c.focus_distance}m "
              f"dist={c.dist} | proj defocus={p.defocus_px}px dist={p.dist}")
    geo = load_geometry(args.scene)
    # Geometry source (--scene) and output folder (--name) are decoupled so a
    # scene file path still lands in a tidy folder; default keeps the
    # ``mitsuba_<scene>`` layout (e.g. blocks -> mitsuba_blocks).
    scene_name = args.name or f"mitsuba_{Path(args.scene).stem}"

    ambient = load_scene_ambient(args.scene)    # scene-defined ambient light level
    print(f"scene '{args.scene}' -> ./{args.out}/{scene_name}/  (ambient {ambient})")
    print("rendering ground truth + albedo ...")
    gt, _ = render_ground_truth(rig, geometry=geo, spp=args.gt_spp, label="ground-truth")
    white = render_capture(rig, np.ones((rig.projector.height, rig.projector.width), np.float32),
                           geometry=geo, spp=args.spp, ambient=ambient, label="white-ref")
    albedo = np.clip(white / max(white.max(), 1e-6), 0, 1)

    # The fundamental SL ground truth: exact projector subpixel (col, row) that lit
    # each camera pixel, from GT depth + calibration. Depth is the same information
    # triangulated, so we keep both (gt_depth feeds the cloud/viewer).
    gt_proj = projector_subpixel(rig, gt, proj_optics=optics.projector)
    # Occlusion is a property of the (shared) geometry; use the analytic raycaster
    # as the visibility oracle for both backends so it's acne-free on curved surfaces.
    gt_proj = np.where(projector_visible(rig, load_scene_primitives(args.scene))[..., None],
                       gt_proj, np.nan)

    sdir = io.ensure_dir(os.path.join(args.out, scene_name))
    io.save_npy(os.path.join(sdir, "gt_depth.npy"), gt)
    io.save_npy(os.path.join(sdir, "gt_proj.npy"), gt_proj)
    # Quick-look of the projector-column correspondence (normalised to [0,1]).
    io.save_image(os.path.join(sdir, "gt_proj.png"),
                  io.proj_to_rgb(gt_proj, rig.projector.width, rig.projector.height))
    # white = raw white-lit capture; albedo = white normalised so its peak is 1.0.
    io.save_image(os.path.join(sdir, "white.png"), white)
    io.save_image(os.path.join(sdir, "albedo.png"), albedo)
    pts, col = io.depth_to_points(gt, rig, albedo)
    io.save_ply(os.path.join(sdir, "gt_cloud.ply"), pts, col)

    render_pattern_dir(rig, geo, args.patterns, sdir, args.spp, optics=optics, ambient=ambient)
    print(f"\nwrote GT + captures to ./{args.out}/{scene_name}/ (viewable in the synthetic 3D tab)")


if __name__ == "__main__":
    main()
