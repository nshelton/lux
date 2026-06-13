#!/usr/bin/env python3
"""Generate a randomized *planar* training dataset with the NumPy ray-caster.

Every sample is a single textured plane viewed from a rig placed at a random
pose over the plane's hemisphere:

  * the scene is two big half-planes meeting at a crease (junction) through the
    world origin — one flat, the other tilted out of it by a small dihedral angle
    — each with its own random albedo texture / reflectance (set the dihedral to 0
    for a single flat plane); no spheres, no curves;
  * the camera sits anywhere on the +Z hemisphere of that plane (random azimuth
    and tilt up to ``--max-tilt`` off the normal, random viewing distance),
    aimed back at the plane centre, with a uniformly random **roll** about the
    optical axis (the "any z orientation");
  * the projector rides rigidly with the camera (baseline offset along the
    camera's own right axis), so the rig stays a calibrated pair at any pose.

Scene + rig are written as ``scene.json`` / ``rig.json`` into each sample folder
and the standard artifact set (GT depth, gt_proj, captures) is rendered there,
so every sample is self-documenting and re-renderable with
``gen_rasterizer_dataset.py --scene <dir>/scene.json --rig <dir>/rig.json``.

Seeding / resume / split semantics match ``gen_training_data.py`` (sample i uses
``--seed + i``; a per-sample ``sample.json`` written last marks completion).

    python scripts/gen_planar_dataset.py --n 100 --patterns patterns/marray --jobs 4
    python scripts/gen_planar_dataset.py --n 20 --seed 100000 --out renders/val --lean
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

from gen_training_data import (  # noqa: E402
    add_rig_imperfections, render_sample, _with_surface,
)


# --------------------------------------------------------------------------
# Pose / geometry sampling
# --------------------------------------------------------------------------
def _hemisphere_dir(rng: np.random.Generator, max_tilt_deg: float) -> np.ndarray:
    """A unit direction on the +Z hemisphere: azimuth uniform in [0, 2pi), tilt
    uniform in [0, max_tilt] off the +Z normal (uniform tilt, not area-uniform,
    so grazing views are sampled as densely as head-on ones)."""
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = np.radians(rng.uniform(0.0, max_tilt_deg))
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def _rolled_up(rng: np.random.Generator, forward: np.ndarray) -> np.ndarray:
    """A world up-vector giving a uniformly random roll about ``forward``. ``up``
    is the world direction that lands at image-top, so rolling it around the
    optical axis rolls the image (covers every in-image orientation)."""
    f = forward / np.linalg.norm(forward)
    ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(ref, f)
    right /= np.linalg.norm(right)
    up0 = np.cross(right, f)                      # right-handed up, perp to f
    a = rng.uniform(0.0, 2.0 * np.pi)
    return np.cos(a) * up0 + np.sin(a) * right


def sample_planar_scene(rng: np.random.Generator,
                        min_dihedral: float = 8.0,
                        max_dihedral: float = 30.0) -> dict:
    """Two big half-plane slabs meeting at a crease (junction) along the world Y
    axis: one flat at z=0 (the -x half), the other (the +x half) tilted out of it
    by a 'slightly tilted' dihedral angle, both through the origin. Each gets its
    own random albedo/texture, so two surfaces of differing appearance meet at the
    crease. The slabs are sized to overflow the frustum at any sampled pose, so the
    frame stays filled even at grazing tilt (extents are cheap for the ray-caster).

    Modeled as thin rotated *boxes*: the scene ``plane`` primitive is constant-Z and
    can't tilt, but a thin box rotated about Y is an arbitrarily-oriented plane (the
    same trick the ``tilted`` background uses). Setting the dihedral to 0 (e.g.
    ``--min-dihedral 0 --max-dihedral 0``) degrades to a single flat plane."""
    H, t = 40.0, 0.002                              # half-extent / tiny thickness
    theta = float(rng.uniform(min_dihedral, max_dihedral) * rng.choice([-1.0, 1.0]))
    b = np.radians(theta)
    # flat half-plane on -x: spans x in [-2H, 0], inner edge on the crease (x=0,z=0).
    flat = _with_surface({"type": "box", "center": [-H, 0.0, 0.0],
                          "scale": [H, H, t]}, rng, lo=0.45)
    # tilted half-plane on +x: rotate about Y by theta, centre placed so its inner
    # edge (local -x) lands back on the same crease line: c = (H cos b, 0, -H sin b).
    tilt = _with_surface({"type": "box",
                          "center": [round(H * np.cos(b), 4), 0.0, round(-H * np.sin(b), 4)],
                          "scale": [H, H, t],
                          "rotation": [0.0, round(theta, 3), 0.0]}, rng, lo=0.45)
    return {"name": "planar_junction_scene",
            "objects": [flat, tilt],
            "ambient": round(rng.uniform(0.04, 0.12), 3),
            "dihedral_deg": round(theta, 3)}


def sample_planar_rig(rng: np.random.Generator, width: int, height: int,
                      dmin: float, dmax: float, max_tilt: float,
                      cam_distort: bool = False) -> dict:
    """A posed rig over the plane's hemisphere: camera at a random hemisphere
    direction * random distance, looking at the plane centre with random roll;
    projector offset along the camera's right axis by a random baseline. Then
    the shared probabilistic lens/sensor imperfections are layered on."""
    n = _hemisphere_dir(rng, max_tilt)
    d = rng.uniform(dmin, dmax)
    cam_C = d * n                                  # plane centre is the origin
    forward = -cam_C                              # aim back at the plane centre
    up = _rolled_up(rng, forward)

    # camera world basis (rows: right, down, forward) -> offset the projector
    # along the camera's own right axis so the baseline is sensible at any pose.
    f = forward / np.linalg.norm(forward)
    y = (up @ f) * f - up
    y /= np.linalg.norm(y)
    right = np.cross(y, f)
    baseline = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0])
    proj_C = (cam_C + baseline * right
              + rng.uniform(-0.08, 0.08) * y          # small vertical jitter
              + rng.uniform(-0.06, 0.06) * f)         # small in/out jitter

    target = [0.0, 0.0, 0.0]
    camera = {"width": width, "height": height,
              "hfov_deg": round(rng.uniform(38.0, 55.0), 2),
              "position": [round(float(x), 4) for x in cam_C],
              "look_at": target, "up": [round(float(x), 4) for x in up]}
    projector = {"width": 1920, "height": 1080,
                 "hfov_deg": round(rng.uniform(35.0, 50.0), 2),
                 "position": [round(float(x), 4) for x in proj_C],
                 "look_at": target, "up": [round(float(x), 4) for x in up]}
    rig = {"name": "planar_rig", "camera": camera, "projector": projector}
    return add_rig_imperfections(rig, rng, cam_distort=cam_distort)


# --------------------------------------------------------------------------
# Per-sample driver (resume/skip semantics mirror gen_training_data.py)
# --------------------------------------------------------------------------
def _render_one(i: int, args) -> str:
    seed = args.seed + i
    sdir = os.path.join(args.out, f"sample_{seed:05d}")
    manifest = Path(sdir, "sample.json")
    if manifest.exists() and not args.overwrite:
        return f"sample_{seed:05d}: exists, skipped"

    rng = np.random.default_rng(seed)
    io.ensure_dir(sdir)
    scene_path, rig_path = Path(sdir, "scene.json"), Path(sdir, "rig.json")
    rig_spec = sample_planar_rig(rng, args.width, args.height, args.dmin, args.dmax,
                                 args.max_tilt, cam_distort=args.cam_distort)
    scene = sample_planar_scene(rng, args.min_dihedral, args.max_dihedral)
    rig_path.write_text(json.dumps(rig_spec, indent=2) + "\n")
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")

    print(f"=== planar sample seed {seed} -> ./{sdir}/ ===")
    render_sample(scene_path, rig_path, args.patterns, sdir, lean=args.lean)

    manifest.write_text(json.dumps({
        "seed": seed,
        "patterns": [Path(p).name for p in args.patterns],
        "camera": [args.width, args.height],
        "kind": "planar_junction",
        "dihedral_deg": scene["dihedral_deg"],
        "lean": args.lean,
    }, indent=2) + "\n")
    return f"sample_{seed:05d}: done"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", nargs="+", default=["patterns/marray"],
                    help="one or more folders of PNG patterns (each rendered into its "
                         "own subfolder per sample; G-buffer is shared so extras are cheap)")
    ap.add_argument("--n", type=int, default=10, help="number of samples to render")
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed; sample i uses seed+i (reproducible, extendable, "
                         "splittable via disjoint seed ranges)")
    ap.add_argument("--width", type=int, default=1920, help="camera width")
    ap.add_argument("--height", type=int, default=1080, help="camera height")
    ap.add_argument("--dmin", type=float, default=0.8, help="min camera distance to plane (m)")
    ap.add_argument("--dmax", type=float, default=3.0, help="max camera distance to plane (m)")
    ap.add_argument("--max-tilt", type=float, default=75.0,
                    help="max camera tilt off the plane normal, degrees (0 = head-on, "
                         "90 = grazing); samples uniformly in [0, max-tilt]")
    ap.add_argument("--min-dihedral", type=float, default=8.0,
                    help="min crease (dihedral) angle between the two planes, degrees")
    ap.add_argument("--max-dihedral", type=float, default=30.0,
                    help="max crease angle, degrees (set both min/max to 0 for a single "
                         "flat plane); the tilt sign is randomized (ridge or valley)")
    ap.add_argument("--cam-distort", action="store_true",
                    help="also randomize camera lens distortion (warps captures out of "
                         "the ideal image space GT lives in; undistort downstream)")
    ap.add_argument("--lean", action="store_true",
                    help="skip human-facing extras (gt_cloud.ply, montage, gt_proj.png, "
                         "albedo.png) - keeps gt_depth/gt_proj/white/captures only")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render samples that already have a sample.json "
                         "(default: skip, so an interrupted run resumes)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (samples are independent)")
    ap.add_argument("--out", default="renders/planar")
    args = ap.parse_args()

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(_render_one, range(args.n), [args] * args.n))
    else:
        results = [_render_one(i, args) for i in range(args.n)]

    done = sum(r.endswith("done") for r in results)
    skipped = len(results) - done
    print(f"\n{done} rendered, {skipped} skipped -> ./{args.out}/ "
          f"(seeds {args.seed}..{args.seed + args.n - 1})")


if __name__ == "__main__":
    main()
