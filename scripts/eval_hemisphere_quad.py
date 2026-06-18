#!/usr/bin/env python3
"""Hemisphere per-band eval for a continuous-phase (quad) co-design checkpoint — the §9.5
render-validation table. Same metrics/binning as ``eval_hemisphere.py`` (bin-acc, med|du|,
med|du| within correct unwrap, p95, IoU, binned by max(cam,proj) obliquity) but decodes via the
CRT vote (``lux.codesign_infer.predict_quad``) instead of argmax-bin. Run the M-array+bin baseline
with the original ``eval_hemisphere.py`` on the same poses for the matched A/B.

    python scripts/eval_hemisphere_quad.py --ckpt checkpoints/codesign_quad.pt --pattern-set codesign_learned
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.codesign_infer import load_quad, predict_quad  # noqa: E402
from lux.proj_net import N_BINS_U  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/codesign_quad.pt")
    ap.add_argument("--data", default="evals/hemisphere/data")
    ap.add_argument("--pattern-set", default="codesign_learned")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, gen, proj_wh, pu, pv = load_quad(args.ckpt, device=args.device)
    dirs = sorted(d for d in Path(args.data).glob("sample_*") if (d / "sample.json").exists())
    if not dirs:
        raise SystemExit(f"no samples under {args.data!r}")
    binw = proj_wh[0] / N_BINS_U
    rows = []
    for i, d in enumerate(dirs):
        cap = d / args.pattern_set / args.frame
        if not cap.exists():
            continue
        img = io.load_image(str(cap), gray=True)
        gt = np.load(d / "gt_proj.npy")
        uv, conf = predict_quad(model, img, proj_wh, pu, pv, device=args.device, return_conf=True)
        both = np.isfinite(gt[..., 0]) & np.isfinite(uv[..., 0])
        union = np.isfinite(gt[..., 0]) | np.isfinite(uv[..., 0])
        pose = json.loads((d / "sample.json").read_text())
        r = {**pose, "name": d.name, "iou": both.sum() / max(union.sum(), 1)}
        if both.any():
            du = np.abs((uv[..., 0] - gt[..., 0])[both])
            correct = (uv[..., 0][both] // binw).clip(0, N_BINS_U - 1) == (gt[..., 0][both] // binw).clip(0, N_BINS_U - 1)
            r.update(bin_acc=float(correct.mean()), med_du=float(np.median(du)),
                     med_du_correct=float(np.median(du[correct])) if correct.any() else np.nan,
                     p95_du=float(np.percentile(du, 95)))
        else:
            r.update(bin_acc=0.0, med_du=np.nan, med_du_correct=np.nan, p95_du=np.nan)
        rows.append(r)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(dirs)}", flush=True)

    theta = np.array([max(r["theta_cam_deg"], r["theta_proj_deg"]) for r in rows])
    edges = [0, 15, 30, 45, 60, 75, 90]
    print(f"\n{args.ckpt}  [{args.pattern_set}]  on {len(rows)} hemisphere samples "
          f"(binned by max(cam,proj) obliquity):")
    print(f"{'oblique':>10s} {'n':>4s} {'bin acc':>8s} {'med|du|':>8s} {'med|du|✓bin':>11s} {'p95|du|':>8s} {'IoU':>6s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, t in zip(rows, theta) if lo <= t < hi]
        if not sel:
            continue
        agg = lambda k: float(np.nanmedian(np.array([s[k] for s in sel], float)))
        print(f"{lo:3d}-{hi:3d}deg {len(sel):4d} {agg('bin_acc')*100:7.1f}% {agg('med_du'):7.2f}px "
              f"{agg('med_du_correct'):10.2f}px {agg('p95_du'):7.1f}px {agg('iou'):6.3f}")

    out = Path(args.out or Path(args.data).parent / f"results_quad_{Path(args.ckpt).stem}")
    out.mkdir(parents=True, exist_ok=True)
    keys = ["name", "theta_cam_deg", "theta_proj_deg", "bin_acc", "med_du", "med_du_correct", "p95_du", "iou"]
    with open(out / "per_sample.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"per-sample csv -> {out}/per_sample.csv")


if __name__ == "__main__":
    main()
