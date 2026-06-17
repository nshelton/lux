#!/usr/bin/env python3
"""Overfit-one-batch information-limit test for the obliquity cliff.

Run at the TRAINING crop scale (256 px) so the result is not contaminated by the
full-frame inference artifact (see docs/tiling_brief.md). Take a *fixed* batch of
256-px crops (no augmentation) from a scene at a given obliquity and train the model
to memorise just those crops. The best achievable TRAIN bin-accuracy on a fixed batch
is a direct measure of input distinguishability:

  - reaches ~100%  -> the 20px M-array window still uniquely determines position at
                      this obliquity -> the cliff is a training/generalisation problem
                      (fixable with oblique data), NOT an information limit.
  - plateaus low   -> different projector positions produce indistinguishable
                      (anamorphically compressed) windows -> genuine information limit;
                      no amount of training data fixes it, needs a different pattern.

Frontal control should overfit to ~100% and validates the setup. GPU/MPS, ~minutes.

    python scripts/overfit_oblique.py
"""
from __future__ import annotations
import csv
import json
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
CROP, K, STEPS, LOG = 256, 16, 2000, 200


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_batch(name: str, dev: str, seed: int = 0):
    """K fixed 256-px crops (no aug) with >20% valid GT, as MPS tensors."""
    rng = np.random.default_rng(seed)
    d = DATA / name
    img = io.load_image(str(d / "marray" / "cap_pat_00.png"), gray=True).astype(np.float32)
    gt = np.load(d / "gt_proj.npy").astype(np.float32)
    rig = json.loads((d / "rig.json").read_text())
    pw = (rig["projector"]["width"], rig["projector"]["height"])
    H, W = img.shape
    ics, tcs, vs = [], [], []
    for _ in range(K):
        for _ in range(80):
            y, x = int(rng.integers(0, H - CROP + 1)), int(rng.integers(0, W - CROP + 1))
            g = gt[y:y + CROP, x:x + CROP]
            v = np.isfinite(g[..., 0])
            if v.mean() > 0.2:
                break
        ics.append(img[y:y + CROP, x:x + CROP])
        tcs.append(np.nan_to_num(g / np.asarray(pw, np.float32)))
        vs.append(v)
    ic = torch.from_numpy(np.stack(ics)[:, None]).to(dev)
    tc = torch.from_numpy(np.stack(tcs).transpose(0, 3, 1, 2).copy()).to(dev)
    vv = torch.from_numpy(np.stack(vs)[:, None].astype(np.float32)).to(dev)
    return ic, tc, vv, pw


def overfit(name: str, init: str, dev: str):
    ic, tc, vv, pw = load_batch(name, dev)
    if init == "conv":
        model, _ = load_checkpoint(CONV_CKPT, device=dev)
    else:
        model = ProjUNet(mid="conv").to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curve = []
    for step in range(STEPS + 1):
        loss, du, dv, _, ub, vb = proj_loss(model(ic), tc, vv, pw)
        if step % LOG == 0:
            curve.append((step, ub.item() * 100, vb.item() * 100, du.item(), dv.item()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return curve


def main():
    dev = device()
    rows = list(csv.DictReader(open(CSV)))
    ob = lambda r: max(float(r["theta_cam_deg"]), float(r["theta_proj_deg"]))  # noqa: E731
    rows.sort(key=ob)

    def pick(lo, hi):
        c = [r for r in rows if lo <= ob(r) < hi]
        c.sort(key=lambda r: float(r["bin_acc"]))   # clearest cliff = lowest tiled bin-acc
        return c[len(c) // 2] if c else None

    conds = [("frontal  ~7deg", rows[0], "fresh"),
             ("oblique ~60deg", pick(58, 63), "fresh"),
             ("oblique ~74deg", pick(72, 76), "fresh"),
             ("oblique ~74deg", pick(72, 76), "conv")]
    print(f"device {dev}; {K} fixed 256px crops, no aug, {STEPS} steps, lr 1e-3\n")
    summary = []
    for label, r, init in conds:
        if r is None:
            print(f"(no sample for {label})")
            continue
        name = r["name"]
        curve = overfit(name, init, dev)
        print(f"=== {label}  [{name}, max-tilt {ob(r):.1f}°, conv-tiled bin {float(r['bin_acc'])*100:.1f}%]  init={init} ===")
        print("   step  u-bin  v-bin   |du|px  |dv|px")
        for st, ub, vb, du, dv in curve:
            print(f"  {st:5d}  {ub:5.1f} {vb:5.1f}  {du:7.1f} {dv:7.1f}")
        fu, fv = curve[-1][1], curve[-1][2]
        summary.append((label, init, fu, fv))
        print()
    print("=== overfit ceilings (final-step train bin-acc on the fixed batch) ===")
    for label, init, fu, fv in summary:
        verdict = "INFO-SUFFICIENT (fixable)" if min(fu, fv) > 90 else \
                  "PARTIAL" if min(fu, fv) > 60 else "INFO-LIMIT (ambiguous)"
        print(f"  {label:16s} init={init:5s}  u {fu:5.1f}%  v {fv:5.1f}%  -> {verdict}")


if __name__ == "__main__":
    main()
