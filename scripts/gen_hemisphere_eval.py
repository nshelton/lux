#!/usr/bin/env python3
"""Render the hemisphere-over-a-flat-plane evaluation set.

The projection-mapping deployment case: a single large plane, with the camera
and projector posed independently on the hemisphere above it — elevation
(obliquity from the plane normal) uniform in [0, --max-oblique], azimuth
uniform, distance log-uniform. Captures are clean (no noise / DoF / distortion)
so the sweep isolates *pose* generalization: anamorphic code compression at
grazing angles, scale variation, keystone.

Each sample folder gets the standard artifact set plus the pose angles in
``sample.json``; ``scripts/eval_hemisphere.py`` turns a checkpoint into
metric-vs-obliquity curves over this set.

    python scripts/gen_hemisphere_eval.py --n 160
    python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net.pt
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
from gen_training_data import render_sample  # noqa: E402


def _hemi_pos(rng: np.random.Generator, max_oblique: float, target: np.ndarray):
    """Position on the hemisphere over the plane z=0 (camera side is z<0),
    aimed at ``target``. Returns (position, theta_deg, phi_deg, r)."""
    theta = np.radians(rng.uniform(0.0, max_oblique))   # uniform in angle, not solid angle
    phi = rng.uniform(0.0, 2 * np.pi)
    r = float(np.exp(rng.uniform(np.log(1.5), np.log(3.0))))
    pos = target + r * np.array([np.sin(theta) * np.cos(phi),
                                 np.sin(theta) * np.sin(phi),
                                 -np.cos(theta)])
    return pos, float(np.degrees(theta)), float(np.degrees(phi)), r


def sample_hemi_rig(rng: np.random.Generator, max_oblique: float) -> tuple[dict, dict]:
    """A posed rig on the hemisphere; returns (rig_spec, pose_meta)."""
    target = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.0])
    cam_pos, tc, pc, rc = _hemi_pos(rng, max_oblique, target)
    proj_pos, tp, pp, rp = _hemi_pos(rng, max_oblique, target)
    # 'up' must not be parallel to the view direction at steep azimuths.
    def up_for(pos):
        fwd = target - pos
        fwd = fwd / np.linalg.norm(fwd)
        return [0.0, 0.0, -1.0] if abs(fwd[1]) > 0.9 else [0.0, -1.0, 0.0]
    rig = {
        "name": "hemisphere_eval",
        "camera": {"width": 1920, "height": 1080, "hfov_deg": 45.0,
                   "position": [round(float(x), 4) for x in cam_pos],
                   "look_at": [round(float(x), 4) for x in target],
                   "up": up_for(cam_pos)},
        "projector": {"width": 1920, "height": 1080, "hfov_deg": 40.0,
                      "position": [round(float(x), 4) for x in proj_pos],
                      "look_at": [round(float(x), 4) for x in target],
                      "up": up_for(proj_pos)},
    }
    meta = {"theta_cam_deg": round(tc, 2), "theta_proj_deg": round(tp, 2),
            "phi_cam_deg": round(pc, 2), "phi_proj_deg": round(pp, 2),
            "r_cam_m": round(rc, 3), "r_proj_m": round(rp, 3)}
    return rig, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--max-oblique", type=float, default=75.0,
                    help="max elevation from the plane normal, degrees")
    ap.add_argument("--seed", type=int, default=7_000_000)
    ap.add_argument("--patterns", nargs="+", default=["patterns/marray"])
    ap.add_argument("--out", default="evals/hemisphere/data")
    args = ap.parse_args()

    scene = {"name": "hemi_plane",
             "objects": [{"type": "plane", "z": 0.0, "scale": 6.0, "reflectance": 0.85}],
             "ambient": 0.08}

    for i in range(args.n):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        sdir = io.ensure_dir(os.path.join(args.out, f"sample_{seed}"))
        manifest = Path(sdir, "sample.json")
        if manifest.exists():
            continue
        rig, pose = sample_hemi_rig(rng, args.max_oblique)
        scene_path, rig_path = Path(sdir, "scene.json"), Path(sdir, "rig.json")
        scene_path.write_text(json.dumps(scene, indent=2) + "\n")
        rig_path.write_text(json.dumps(rig, indent=2) + "\n")
        print(f"=== hemi sample {i + 1}/{args.n} (seed {seed})  "
              f"cam {pose['theta_cam_deg']:.0f}°  proj {pose['theta_proj_deg']:.0f}° ===",
              flush=True)
        render_sample(scene_path, rig_path, args.patterns, sdir, lean=True)
        manifest.write_text(json.dumps({"seed": seed, **pose}, indent=2) + "\n")

    print(f"\nhemisphere eval set -> ./{args.out}/")


if __name__ == "__main__":
    main()
