#!/usr/bin/env python3
"""Evaluate a correspondence checkpoint on the hemisphere flat-plane set.

For every sample under --data: full-frame inference, then per-sample metrics
(u-bin accuracy, median |du|, median |du| within correct bins, validity IoU)
joined with the sample's pose angles. Writes ``per_sample.csv`` and prints the
metric-vs-obliquity table (binned by max of camera/projector elevation — the
axis along which the code becomes anamorphic). Saves a scatter plot when
matplotlib is available.

    python scripts/eval_hemisphere.py --ckpt checkpoints/proj_net.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.proj_net import load_checkpoint, predict_full, N_BINS_U  # noqa: E402


def eval_sample(model, proj_wh, d: Path, device: str, pattern_set: str, frame: str) -> dict:
    img = io.load_image(str(d / pattern_set / frame), gray=True)
    gt = np.load(d / "gt_proj.npy")
    pred = predict_full(model, img, proj_wh, device=device)
    both = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
    union = np.isfinite(gt[..., 0]) | np.isfinite(pred[..., 0])
    out = {"iou": both.sum() / max(union.sum(), 1),
           "valid_px": int(both.sum())}
    if both.any():
        du = np.abs((pred[..., 0] - gt[..., 0])[both])
        binw = proj_wh[0] / N_BINS_U
        correct = (pred[..., 0][both] // binw).clip(0, N_BINS_U - 1) == \
                  (gt[..., 0][both] // binw).clip(0, N_BINS_U - 1)
        out.update(bin_acc=float(correct.mean()),
                   med_du=float(np.median(du)),
                   med_du_correct=float(np.median(du[correct])) if correct.any() else np.nan,
                   p95_du=float(np.percentile(du, 95)))
    else:
        out.update(bin_acc=0.0, med_du=np.nan, med_du_correct=np.nan, p95_du=np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evals/hemisphere/data")
    ap.add_argument("--ckpt", default="checkpoints/proj_net.pt")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="results dir (default <data>/../results_<ckpt-stem>)")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model, proj_wh = load_checkpoint(args.ckpt, device=args.device)

    dirs = sorted(d for d in Path(args.data).glob("sample_*") if (d / "sample.json").exists())
    if not dirs:
        raise SystemExit(f"no finished samples under {args.data!r}")
    out_dir = Path(args.out or Path(args.data).parent / f"results_{Path(args.ckpt).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, d in enumerate(dirs):
        pose = json.loads((d / "sample.json").read_text())
        m = eval_sample(model, proj_wh, d, args.device, args.pattern_set, args.frame)
        rows.append({**pose, **m, "name": d.name})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(dirs)} evaluated", flush=True)

    keys = ["name", "theta_cam_deg", "theta_proj_deg", "r_cam_m", "r_proj_m",
            "bin_acc", "med_du", "med_du_correct", "p95_du", "iou", "valid_px"]
    with open(out_dir / "per_sample.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # Binned summary along max(theta_cam, theta_proj).
    theta = np.array([max(r["theta_cam_deg"], r["theta_proj_deg"]) for r in rows])
    edges = [0, 15, 30, 45, 60, 75, 90]
    print(f"\n{args.ckpt} on {len(rows)} hemisphere samples "
          f"(binned by max(cam, proj) obliquity):")
    print(f"{'oblique':>10s} {'n':>4s} {'bin acc':>8s} {'med|du|':>8s} "
          f"{'med|du|✓bin':>11s} {'p95|du|':>8s} {'IoU':>6s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r, t in zip(rows, theta) if lo <= t < hi]
        if not sel:
            continue
        def agg(k):
            v = np.array([s[k] for s in sel], float)
            return np.nanmedian(v)
        print(f"{lo:3d}-{hi:3d}deg {len(sel):4d} {agg('bin_acc')*100:7.1f}% "
              f"{agg('med_du'):7.2f}px {agg('med_du_correct'):10.2f}px "
              f"{agg('p95_du'):7.1f}px {agg('iou'):6.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        tp = [r["theta_proj_deg"] for r in rows]
        tc = [r["theta_cam_deg"] for r in rows]
        ba = [r["bin_acc"] * 100 for r in rows]
        sc = axes[0].scatter(tp, ba, c=tc, cmap="viridis", s=18)
        axes[0].set_xlabel("projector obliquity (deg)")
        axes[0].set_ylabel("u-bin accuracy (%)")
        fig.colorbar(sc, ax=axes[0], label="camera obliquity (deg)")
        md = [r["med_du"] for r in rows]
        sc2 = axes[1].scatter(tp, md, c=tc, cmap="viridis", s=18)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("projector obliquity (deg)")
        axes[1].set_ylabel("median |du| (px, log)")
        fig.colorbar(sc2, ax=axes[1], label="camera obliquity (deg)")
        fig.suptitle(f"{Path(args.ckpt).name} — hemisphere flat-plane sweep")
        fig.tight_layout()
        fig.savefig(out_dir / "obliquity_sweep.png", dpi=130)
        print(f"\nplot -> {out_dir}/obliquity_sweep.png")
    except ImportError:
        print("\n(matplotlib not installed — skipped plot)")
    print(f"per-sample csv -> {out_dir}/per_sample.csv")


if __name__ == "__main__":
    main()
