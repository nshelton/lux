#!/usr/bin/env python3
"""Overfit-one-batch INFORMATION-LIMIT test for the residual obliquity cliff.

Run at the TRAINING crop scale (256 px) so the result is not contaminated by the
full-frame inference artifact (docs/tiling_brief.md). Try to MEMORISE a fixed,
un-augmented batch of 256-px crops from a scene at a given obliquity. The best
achievable TRAIN bin-accuracy on a fixed batch measures input distinguishability.

Methodology (per review — the verdict is a FRACTION, not yes/no, and only means an
information limit once capacity and optimisation are ruled out):
  - SWEEP LR per obliquity and take the BEST ceiling -> rules out "bad LR caused the
    plateau" (optimisation confound).
  - >=2 BATCHES (crop seeds) at the deep angle -> rules out a one-batch fluke.
  - Frontal control must hit ~100% -> proves the head/capacity can represent it.
  - conv-init cross-check -> a codebook-aware model; if it can't fit it either, strong.
  - Synthetic + exact GT -> no label noise to confound the ceiling.
Interpretation: ceiling ~100% => information-SUFFICIENT (cliff is a training/
generalisation problem, fixable with oblique data). A partial ceiling (e.g. ~55%) is
the achievable decodable FRACTION; a low ceiling => information limit. A genuine fail
is a pattern SPEC: the ~20-px projector M-array window foreshortens to ~20*cos(tilt)
camera px; below the camera resolving floor (~4-5 px) different projector positions
alias -> the coarse scale of a multi-scale pattern must survive that compression.

    python scripts/overfit_oblique.py     # env: OVERFIT_STEPS, OVERFIT_K
"""
from __future__ import annotations
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lux import io  # noqa: E402
from lux.proj_net import ProjUNet, proj_loss, load_checkpoint  # noqa: E402

DATA = Path("evals/hemisphere/data")
CSV = Path("evals/hemisphere/results_conv_tiled/per_sample.csv")
CONV_CKPT = "checkpoints/proj_net_scratch.pt"
CROP = 256
K = int(os.environ.get("OVERFIT_K", "8"))
STEPS = int(os.environ.get("OVERFIT_STEPS", "800"))
LOG = int(os.environ.get("OVERFIT_LOG", "150"))
WIN_PROJ_PX = 20.0          # 5x5 M-array cells x 4 px = decodable projector window
CAM_FLOOR_PX = 4.5          # rough camera resolving floor for a unique window


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_batch(name: str, dev: str, seed: int):
    rng = np.random.default_rng(seed)
    d = DATA / name
    img = io.load_image(str(d / "marray" / "cap_pat_00.png"), gray=True).astype(np.float32)
    gt = np.load(d / "gt_proj.npy").astype(np.float32)
    rig = json.loads((d / "rig.json").read_text())
    pw = (rig["projector"]["width"], rig["projector"]["height"])
    H, W = img.shape
    ics, tcs, vs = [], [], []
    for _ in range(K):
        for _ in range(120):
            y, x = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
            g = gt[y:y + CROP, x:x + CROP]
            v = np.isfinite(g[..., 0])
            if v.mean() > 0.15:
                break
        ics.append(img[y:y + CROP, x:x + CROP])
        tcs.append(np.nan_to_num(g / np.asarray(pw, np.float32)))
        vs.append(v)
    ic = torch.from_numpy(np.stack(ics)[:, None]).to(dev)
    tc = torch.from_numpy(np.stack(tcs).transpose(0, 3, 1, 2).copy()).to(dev)
    vv = torch.from_numpy(np.stack(vs)[:, None].astype(np.float32)).to(dev)
    return ic, tc, vv, pw


def overfit(name, init, lr, seed, dev):
    ic, tc, vv, pw = load_batch(name, dev, seed)
    model = load_checkpoint(CONV_CKPT, device=dev)[0] if init == "conv" else ProjUNet(mid="conv").to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve = []
    for step in range(STEPS + 1):
        loss, du, dv, _, ub, vb = proj_loss(model(ic), tc, vv, pw)
        if step % LOG == 0:
            curve.append((step, ub.item() * 100, vb.item() * 100))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    umax = max(u for _, u, _ in curve)
    vmax = max(v for _, _, v in curve)
    plateaued = curve[-1][1] - curve[-2][1] < 1.5     # u-bin gain over last LOG window < 1.5pt
    return curve, umax, vmax, plateaued


def main():
    dev = device()
    rows = list(csv.DictReader(open(CSV)))
    ob = lambda r: max(float(r["theta_cam_deg"]), float(r["theta_proj_deg"]))  # noqa: E731
    rows.sort(key=ob)

    def pick(lo, hi):
        c = [r for r in rows if lo <= ob(r) < hi]
        c.sort(key=lambda r: float(r["bin_acc"]))     # clearest cliff = lowest tiled bin-acc
        return c[len(c) // 2] if c else None

    bands = {"frontal ~7°": (0, 12), "oblique ~60°": (58, 63), "oblique ~74°": (72, 76)}
    samp = {k: pick(*v) for k, v in bands.items()}
    # (band, init, lr, seed). conv-init is primary: capacity is obviously present (a
    # trained model), it converges in ~hundreds of steps, and a low plateau ACROSS the
    # LR sweep then isolates the information limit. fresh-init at 74° is the
    # no-pretrained-bias cross-check (may need >STEPS to converge — read its plateau flag).
    runs = [("frontal ~7°", "conv", 1e-3, 0),       # control: must hit ~100%
            ("oblique ~60°", "conv", 1e-3, 0),
            ("oblique ~60°", "conv", 3e-3, 0),
            ("oblique ~74°", "conv", 5e-4, 0),
            ("oblique ~74°", "conv", 1e-3, 0),
            ("oblique ~74°", "conv", 3e-3, 0),
            ("oblique ~74°", "conv", 1e-3, 1),      # 2nd batch (rule out one-batch fluke)
            ("oblique ~74°", "fresh", 1e-3, 0)]     # no-pretrained-bias cross-check

    print(f"device {dev}; {K} fixed 256px crops/batch, no aug, {STEPS} steps, fp32", flush=True)
    print("INFO-LIMIT test: sweep LR + >=2 batches, ceiling = best achievable (rules out optimisation)\n", flush=True)
    best = {}   # band -> best (u,v) ceiling across its runs
    for band, init, lr, seed in runs:
        r = samp[band]
        if r is None:
            print(f"(no sample for {band})", flush=True)
            continue
        curve, umax, vmax, plat = overfit(r["name"], init, lr, seed, dev)
        tail = " ".join(f"{u:.0f}/{v:.0f}" for _, u, v in curve[-3:])
        print(f"=== {band} [{r['name']}, tilt {ob(r):.0f}°, conv-tiled {float(r['bin_acc'])*100:.0f}%]  "
              f"init={init} lr={lr:g} seed={seed} ===", flush=True)
        print(f"    u/v ceiling {umax:.0f}/{vmax:.0f}%   last3(u/v) {tail}   plateaued={plat}", flush=True)
        bu, bv = best.get(band, (0, 0))
        best[band] = (max(bu, umax), max(bv, vmax))

    print("\n=== SUMMARY: best achievable overfit ceiling per obliquity ===", flush=True)
    print(f"{'obliquity':14s} {'u-ceil':>7s} {'v-ceil':>7s} {'window@cam':>11s}  verdict", flush=True)
    for band, r in samp.items():
        if r is None or band not in best:
            continue
        u, v = best[band]
        tilt = ob(r)
        wcam = WIN_PROJ_PX * math.cos(math.radians(tilt))
        m = min(u, v)
        verdict = ("INFO-SUFFICIENT (cliff = training/generalisation, fixable with oblique data)"
                   if m > 90 else
                   f"PARTIAL (~{m:.0f}% of pixels decodable; rest ambiguous)" if m > 60 else
                   "INFO-LIMIT (window aliases; needs multi-scale pattern, not more data)")
        print(f"{band:14s} {u:6.0f}% {v:6.0f}%  {wcam:6.1f}px/{CAM_FLOOR_PX:.0f}floor  {verdict}", flush=True)
    print(f"\n(window@cam = {WIN_PROJ_PX:.0f}px projector M-array window x cos(tilt); "
          f"below ~{CAM_FLOOR_PX:.0f}px camera floor => aliasing.)", flush=True)


if __name__ == "__main__":
    main()
