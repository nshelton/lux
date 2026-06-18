#!/usr/bin/env python3
"""§9.5 matched-down baseline: train the M-array bin decoder on the IDENTICAL 400-plane
render-train regime the co-designed quad used (``train_quad_rendered.py``), so Table 1
becomes pattern+decoder vs pattern+decoder with everything else held fixed.

This is a near-verbatim clone of ``train_quad_rendered.py``: SAME ProjSamples pipeline
(crop/crops-per-sample/jitter aug), SAME optimizer (AdamW, lr, wd), SAME cosine schedule,
epochs, batch, workers. The ONLY changes are what's intrinsic to the M-array system:
  - pattern set ``marray`` instead of ``codesign_learned`` (the projected PNG),
  - the model's native bin-classification+offset head (``head='cls'``, conv backbone),
  - ``proj_loss`` instead of ``quad_loss``,
  - a bin-checkpoint (``save_checkpoint``) so ``eval_hemisphere.py`` reads it.
No carrier ramp / coord-L1 (quad-only); offset_weight=2 matches the production M-array run.

    python scripts/train_marray_rendered.py --data evals/hemisphere/data_learned_train \
        --pattern-set marray --epochs 30 --out checkpoints/marray_rendered400.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lux.proj_net import ProjUNet, ProjSamples, proj_loss, save_checkpoint  # noqa: E402
from scripts.train_proj_net import auto_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evals/hemisphere/data_learned_train")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-sample", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--offset-weight", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default=auto_device())
    ap.add_argument("--out", default="checkpoints/marray_rendered400.pt")
    ap.add_argument("--snapshots", action="store_true")
    args = ap.parse_args()
    dev = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    ds = ProjSamples(args.data, pattern_set=args.pattern_set, frame=args.frame,
                     crop=args.crop, crops_per_sample=args.crops_per_sample, jitter=True)
    proj_wh = ds.proj_wh
    print(f"rendered marray-train: {len(ds)} samples x {args.crops_per_sample} crops | "
          f"pattern={args.pattern_set} | proj_wh={proj_wh} | device {dev}", flush=True)
    model = ProjUNet(base=32, mid="conv", head="cls").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(ds, batch_size=max(1, args.batch // args.crops_per_sample),
                        shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        w_loss = w_du = w_dv = w_ub = w_vb = 0.0
        n = 0
        for img, target, valid in loader:
            img, target, valid = (t.flatten(0, 1).to(dev) for t in (img, target, valid))
            loss, du, dv, _, ub, vb = proj_loss(model(img), target, valid, proj_wh,
                                                offset_weight=args.offset_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            w_loss += float(loss.detach()); w_du += float(du); w_dv += float(dv)
            w_ub += float(ub); w_vb += float(vb); n += 1
        sched.step()
        meta = {"epoch": ep, "pattern_set": args.pattern_set, "source": "rendered",
                "regime": "matched-down-400", "base": 32}
        save_checkpoint(args.out, model, list(proj_wh), meta)
        if args.snapshots:
            save_checkpoint(str(Path(args.out).with_name(f"{Path(args.out).stem}_ep{ep:02d}.pt")),
                            model, list(proj_wh), meta)
        print(f"epoch {ep:3d}  loss {w_loss/max(n,1):.3f}  |du| {w_du/max(n,1):.2f}px "
              f"|dv| {w_dv/max(n,1):.2f}px  bin u {w_ub/max(n,1):.3f} v {w_vb/max(n,1):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone -> {args.out}  (eval: scripts/eval_hemisphere.py --ckpt {args.out} "
          f"--data evals/hemisphere/data --pattern-set marray)", flush=True)


if __name__ == "__main__":
    main()
