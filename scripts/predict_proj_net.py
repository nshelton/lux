#!/usr/bin/env python3
"""Run the trained correspondence net on a sample and score it against GT.

Writes ``pred_proj.npy`` (+ ``pred_proj.png`` quicklook) into the sample
folder and, when ``gt_proj.npy`` is present, prints subpixel error stats.

    python scripts/predict_proj_net.py renders/train/sample_00000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.proj_net import load_checkpoint, predict_full, predict_tiled  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="sample folder(s) with <pattern-set>/<frame>")
    ap.add_argument("--ckpt", default="checkpoints/proj_net.pt")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiled", action="store_true",
                    help="stitch full-frame inference from 256-px tiles (training crop "
                         "size); auto-enabled for attn checkpoints")
    ap.add_argument("--tile-overlap", type=int, default=128,
                    help="when tiling, overlap tiles by this many px + keep max-confidence "
                         "per pixel (removes seams). 0 = fast hard-stitch")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="mask predictions whose u-bin softmax confidence is below "
                         "this (e.g. 0.9 -> ~50%% coverage at ~98%% bin accuracy); "
                         "a pred_conf.png quicklook is always written")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model, proj_wh = load_checkpoint(args.ckpt, device=args.device)
    model.to(args.device)
    use_tiled = args.tiled or getattr(model, "arch", "conv") == "attn"

    for s in args.samples:
        d = Path(s)
        img = io.load_image(str(d / args.pattern_set / args.frame), gray=True)
        if use_tiled:
            pred, conf = predict_tiled(model, img, proj_wh, device=args.device,
                                       overlap=args.tile_overlap, return_conf=True)
        else:
            pred, conf = predict_full(model, img, proj_wh, device=args.device,
                                      return_conf=True)
        if args.min_conf > 0:
            pred = np.where(conf[..., None] >= args.min_conf, pred, np.nan)
        io.save_npy(str(d / "pred_proj.npy"), pred)
        io.save_image(str(d / "pred_proj.png"), io.proj_to_rgb(pred, *proj_wh))
        io.save_image(str(d / "pred_conf.png"), conf)

        line = f"{d.name}: pred_proj.npy written"
        gt_path = d / "gt_proj.npy"
        if gt_path.exists():
            gt = np.load(gt_path)
            gt_valid = np.isfinite(gt[..., 0])
            produced = np.isfinite(pred[..., 0])           # predicted-valid ∧ conf ≥ min-conf
            both = gt_valid & produced
            if both.any():
                from lux.proj_net import N_BINS_U
                du = np.abs((pred[..., 0] - gt[..., 0])[both])
                dv = np.abs((pred[..., 1] - gt[..., 1])[both])
                binw = proj_wh[0] / N_BINS_U
                prec = ((pred[..., 0][both] // binw) == (gt[..., 0][both] // binw)).mean()
                cov = both.sum() / max(gt_valid.sum(), 1)
                line += (f"  coverage {cov * 100:.1f}% of GT-valid"
                         f"  bin-precision {prec * 100:.2f}%"
                         f"  |du| median {np.median(du):.2f}px p95 {np.percentile(du, 95):.2f}px"
                         f"  |dv| median {np.median(dv):.2f}px")
        print(line)


if __name__ == "__main__":
    main()
