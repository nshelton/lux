#!/usr/bin/env python3
"""§9.5 clutter eval — its OWN table, separate from the hemisphere-plane obliquity sweep (obliquity
is ill-defined at an edge; blending would pollute the clean obliquity curve). Clutter answers
GEOMETRY: depth discontinuities, occlusion, shadow boundaries.

Aggregate accuracy is the wrong metric (dominated by the large flat regions, hides the edge
degradation that is the entire point). So stratify every pixel by **distance-to-discontinuity**
(the edge analog of the obliquity bins), where a discontinuity is a depth/occlusion edge OR a
projector-shadow boundary, and report per distance band:
  - bin-acc and med|du| (raw accuracy), AND
  - the abstention behavior: mean peak-margin confidence, and coverage + accuracy at a confidence
    threshold. The real test at edges is not "is it accurate" but "does the margin DROP on the
    contaminated pixels so the model abstains rather than confidently mis-decoding" -- risk-coverage
    vs distance-to-edge.

    python scripts/eval_clutter_quad.py --ckpt checkpoints/codesign_quad_rendered.pt \
        --data evals/hemisphere/data_learned_clutter --pattern-set codesign_learned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402  (already a dep via lux.io)
import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.codesign_infer import load_quad, predict_quad  # noqa: E402
from lux.proj_net import N_BINS_U  # noqa: E402

DIST_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]


def edge_distances(gt_proj, gt_depth, depth_step_m=0.05):
    """Distances (px) to the two discontinuity types SEPARATELY, since they abstain for different
    reasons: a DEPTH/occlusion edge blends two surfaces' carriers -> false consensus (risk:
    confidently wrong); a SHADOW boundary has no carrier signal -> low margin (risk: hallucinating
    a confident phase from ambient/noise). Returns (dist_depth, dist_shadow, lit_mask)."""
    surface = np.isfinite(gt_depth)                                   # geometry present
    lit = np.isfinite(gt_proj[..., 0])                               # projector-visible (on surface)
    k = np.ones((3, 3), np.uint8)
    sil = cv2.morphologyEx(surface.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0   # silhouette/occlusion
    d = np.nan_to_num(gt_depth.astype(np.float32), nan=0.0)
    gx = np.abs(d - np.roll(d, 1, 1)); gy = np.abs(d - np.roll(d, 1, 0))
    depth_edge = sil | (((gx + gy) > depth_step_m) & surface)         # silhouette + depth steps
    lit_grad = cv2.morphologyEx(lit.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    shadow_edge = lit_grad & surface & (~sil)                         # lit/unlit transition ON a surface
    dist_depth = cv2.distanceTransform((~depth_edge).astype(np.uint8), cv2.DIST_L2, 5)
    dist_shadow = cv2.distanceTransform((~shadow_edge).astype(np.uint8), cv2.DIST_L2, 5)
    return dist_depth, dist_shadow, lit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/codesign_quad_rendered.pt")
    ap.add_argument("--data", default="evals/hemisphere/data_learned_clutter")
    ap.add_argument("--pattern-set", default="codesign_learned")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--coverage", type=float, default=0.5, help="keep this global fraction by conf")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model, gen, proj_wh, pu, pv = load_quad(args.ckpt, device=args.device)
    binw = proj_wh[0] / N_BINS_U

    DD, DS, CORR, DU, CONF = [], [], [], [], []
    dirs = sorted(d for d in Path(args.data).glob("sample_*") if (d / "gt_proj.npy").exists())
    for i, d in enumerate(dirs):
        cap = d / args.pattern_set / args.frame
        if not cap.exists():
            continue
        img = io.load_image(str(cap), gray=True)
        gt = np.load(d / "gt_proj.npy").astype(np.float32)
        depth = np.load(d / "gt_depth.npy") if (d / "gt_depth.npy").exists() else np.full(img.shape, np.nan)
        dist_depth, dist_shadow, lit = edge_distances(gt, depth)
        uv, conf = predict_quad(model, img, proj_wh, pu, pv, device=args.device, return_conf=True)
        m = lit & np.isfinite(uv[..., 0])         # GT-lit pixels the model didn't abstain on
        if not m.any():
            continue
        corr = (uv[..., 0][m] // binw).clip(0, N_BINS_U - 1) == (gt[..., 0][m] // binw).clip(0, N_BINS_U - 1)
        DD.append(dist_depth[m]); DS.append(dist_shadow[m])
        CORR.append(corr); DU.append(np.abs(uv[..., 0] - gt[..., 0])[m]); CONF.append(conf[m])
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(dirs)}", flush=True)
    DD = np.concatenate(DD); DS = np.concatenate(DS)
    CORR = np.concatenate(CORR); DU = np.concatenate(DU); CONF = np.concatenate(CONF)
    tau = np.quantile(CONF, 1 - args.coverage)    # uncalibrated peak-margin -> use a quantile

    def table(dist, title, risk_note):
        print(f"\n  [{title}] stratified by distance to that edge type:")
        print(f"  {'dist(px)':>9} {'n':>9} {'bin-acc':>8} {'med|du|✓':>9} {'meanConf':>9} "
              f"{'cov@τ':>7} {'acc@τ':>7}")
        for lo, hi in DIST_BINS:
            sel = (dist >= lo) & (dist < hi)
            if sel.sum() < 50:
                continue
            c = CORR[sel]
            kept = sel & (CONF >= tau)
            ck = CORR[kept]
            macc = float(np.median(DU[sel][c])) if c.any() else np.nan
            print(f"  {lo:3.0f}-{hi if hi<1e8 else float('inf'):<4.0f} {sel.sum():9d} {c.mean()*100:7.1f}% "
                  f"{macc:8.2f}px {CONF[sel].mean():9.3f} {kept.sum()/sel.sum()*100:6.0f}% "
                  f"{ck.mean()*100 if ck.any() else 0:6.1f}%")
        print(f"    risk-coverage read: {risk_note}")

    print(f"\n{args.ckpt}  [{args.pattern_set}]  clutter geometry eval, {len(dirs)} scenes, "
          f"{len(DD)} px  (conf tau={tau:.3f} -> {args.coverage*100:.0f}% global coverage)")
    table(DD, "DEPTH / occlusion edges", "false consensus from two surfaces -> watch for acc@τ "
          "COLLAPSING (confidently wrong) rather than cov@τ dropping.")
    table(DS, "SHADOW boundaries", "no carrier signal -> margin SHOULD drop (conf low, cov@τ low) "
          "so acc@τ holds; if meanConf stays high in shadow it's hallucinating.")


if __name__ == "__main__":
    main()
