#!/usr/bin/env python3
"""§9.5: train the PRODUCTION continuous-phase decoder on RENDERED captures of the (frozen) learned
pattern. The co-design chose the pattern on the fast analytic proxy; the zero-shot transfer to
rendered captures failed on the unwrap (decoder sim2real gap), so the production decoder is trained
on the real render distribution -- the standard design-on-proxy / train-on-render split.

Pattern is FIXED (baked into the rendered PNGs), so the generator is loaded only for its carrier
periods (to turn gt_proj into per-carrier target phases for quad_loss); nothing in it trains.
coord-L1 dropped (the ablation showed it bought no du). Coarse-first carrier weighting kept.

    python scripts/train_quad_rendered.py --data evals/hemisphere/data_learned_train \
        --pattern-set codesign_learned --epochs 30 --out checkpoints/codesign_quad_rendered.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lux import codesign as cd  # noqa: E402
from lux.proj_net import ProjUNet, ProjSamples  # noqa: E402
from scripts.train_codesign_quad import U_PERIODS, V_PERIODS, carrier_weights  # noqa: E402
from scripts.train_proj_net import auto_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evals/hemisphere/data_learned_train")
    ap.add_argument("--pattern-set", default="codesign_learned")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-sample", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ramp-steps", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default=auto_device())
    ap.add_argument("--out", default="checkpoints/codesign_quad_rendered.pt")
    ap.add_argument("--snapshots", action="store_true")
    args = ap.parse_args()
    dev = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    gen = cd.PatternGenerator.ladder((1920, 1080), U_PERIODS, V_PERIODS)   # periods only; not trained
    proj_wh = gen.proj_wh
    ds = ProjSamples(args.data, pattern_set=args.pattern_set, frame=args.frame,
                     crop=args.crop, crops_per_sample=args.crops_per_sample, jitter=True)
    print(f"rendered quad-train: {len(ds)} samples x {args.crops_per_sample} crops | "
          f"u={U_PERIODS} v={V_PERIODS} | device {dev}", flush=True)
    model = ProjUNet(base=32, head="quad", n_cu=len(U_PERIODS), n_cv=len(V_PERIODS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(ds, batch_size=max(1, args.batch // args.crops_per_sample),
                        shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0)

    gstep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        w_loss = w_cu = w_cv = 0.0
        n = 0
        for img, target, valid in loader:
            img, target, valid = (t.flatten(0, 1).to(dev) for t in (img, target, valid))
            loss, ua, va, bce = cd.quad_loss(model(img), target, valid, gen, proj_wh,
                                             carrier_weights=carrier_weights(gstep, args.ramp_steps))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            w_loss += float(loss.detach()); w_cu += float(ua[-1]); w_cv += float(va[-1]); n += 1
            gstep += 1
        sched.step()
        meta = {"epoch": ep, "u_periods": U_PERIODS, "v_periods": V_PERIODS, "source": "rendered"}
        ck = {"state": model.state_dict(), "gen_state": gen.state_dict(), "proj_wh": list(proj_wh),
              "head": "quad", "n_cu": len(U_PERIODS), "n_cv": len(V_PERIODS), "meta": meta}
        torch.save(ck, args.out)
        if args.snapshots:
            torch.save(ck, str(Path(args.out).with_name(f"{Path(args.out).stem}_ep{ep:02d}.pt")))
        print(f"epoch {ep:3d}  loss {w_loss/max(n,1):.3f}  coarse-align u {w_cu/max(n,1):.3f} "
              f"v {w_cv/max(n,1):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone -> {args.out}  (eval: scripts/eval_hemisphere_quad.py --ckpt {args.out} "
          f"--data evals/hemisphere/data_learned --pattern-set codesign_learned)", flush=True)


if __name__ == "__main__":
    main()
