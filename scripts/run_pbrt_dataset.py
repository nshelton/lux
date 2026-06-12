#!/usr/bin/env python3
"""Decode the PBRT Gray-coded phase-shift dataset and emit site artifacts.

For each scene: load the 16 captures, decode the Gray+sine hybrid into a
projector-column correspondence map, triangulate to a metric point cloud using
the rig parsed from the PBRT includes, and write everything the web viewer needs
to ``results_real/``.

    python scripts/run_pbrt_dataset.py \
        --root camera-4056-3040-projector-1920-1080 \
        --variant real-with-out-dispersion --downsample 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.datasets.pbrt_sl import (  # noqa: E402
    PBRTDataset, HybridGrayPhase, build_hybrid_calibration, parse_pbrt_rig,
)


def colormap(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize + JET colour an array to an RGB float image in [0,1]."""
    out = np.zeros(arr.shape + (3,), np.float64)
    if valid.any():
        lo, hi = np.nanpercentile(arr[valid], [1, 99])
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        bgr = cv2.applyColorMap((np.nan_to_num(norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = (bgr[..., ::-1].astype(np.float64) / 255.0)
    out[~valid] = 0.12
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="camera-4056-3040-projector-1920-1080")
    ap.add_argument("--variant", default="perspective",
                    help="perspective (pinhole, triangulates cleanly) or real-with[-out]-dispersion")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--scenes", nargs="*", default=None, help="subset of scene names")
    ap.add_argument("--amp", type=float, default=8.0, help="sine modulation threshold")
    ap.add_argument("--out", default="results_real")
    args = ap.parse_args()

    ds = PBRTDataset(root=args.root, variant=args.variant, downsample=args.downsample)
    calib = build_hybrid_calibration(ds.proj_pattern_dir)
    print(f"calibration: {calib.n_fringes} fringes, fringe_width={calib.fringe_width:.2f}px, "
          f"sign={calib.sign:+.0f}, offset={calib.offset:.3f}")
    rig = parse_pbrt_rig(ds.scene_root, downsample=args.downsample)
    print(f"rig: cam fx={rig.camera.fx:.0f} {rig.camera.width}x{rig.camera.height}, "
          f"proj fx={rig.proj_fx:.0f}, baseline={np.linalg.norm(rig.proj_pos-rig.cam_pos):.3f}m, "
          f"metric_scale={rig.metric_scale}")

    decoder = HybridGrayPhase(calib, amp_threshold=args.amp)
    scenes = args.scenes or ds.scenes()
    out_root = Path(args.out)
    io.ensure_dir(str(out_root))
    manifest = {"dataset": args.root, "variant": args.variant,
                "downsample": args.downsample,
                "metric_scale": rig.metric_scale,  # baseline calibrated to the reference sphere
                "scenes": []}

    for name in scenes:
        try:
            sine, gray, white = ds.load_captures(name)
        except Exception as e:
            print(f"  skip {name}: {e}")
            continue
        res = decoder.decode(sine, gray, white)
        depth = rig.triangulate(res.proj_col, res.valid)

        sdir = io.ensure_dir(str(out_root / name))
        albedo = np.clip(white / max(white.max(), 1e-6), 0, 1)
        pts, col = io.depth_to_points(depth, rig, albedo)
        io.save_ply(str(Path(sdir) / "cloud.ply"), pts, col)
        io.save_image(str(Path(sdir) / "white.png"), albedo)
        io.save_image(str(Path(sdir) / "column.png"), colormap(res.proj_col, res.valid))
        io.save_image(str(Path(sdir) / "phase.png"), colormap(res.phase, res.valid))
        io.save_image(str(Path(sdir) / "depth.png"), colormap(depth, np.isfinite(depth)))
        io.save_npy(str(Path(sdir) / "depth.npy"), depth)

        finite = np.isfinite(depth)
        entry = {
            "name": name,
            "n_points": int(len(pts)),
            "valid_pct": round(100 * float(res.valid.mean()), 1),
            "depth_min": round(float(np.nanmin(depth)), 4) if finite.any() else None,
            "depth_max": round(float(np.nanmax(depth)), 4) if finite.any() else None,
        }
        manifest["scenes"].append(entry)
        print(f"  {name:22s} pts={entry['n_points']:7d} valid={entry['valid_pct']:5.1f}%  "
              f"depth=[{entry['depth_min']},{entry['depth_max']}]m")

    io.save_json(str(out_root / "manifest.json"), manifest)
    print(f"\nwrote {len(manifest['scenes'])} scenes + manifest.json to ./{args.out}/")


if __name__ == "__main__":
    main()
