#!/usr/bin/env python3
"""Joint pattern + decoder co-design on the differentiable homography-warp proxy.

Trains the one-shot correspondence U-Net (:class:`lux.proj_net.ProjUNet`) AND a learnable
projected pattern (:class:`lux.codesign.PatternGenerator`) end-to-end against synthetic
oblique-plane captures (:class:`lux.codesign.ProxyBatcher`) -- no Mitsuba in the loop. The
target is the 60-75 deg grazing floor that is a hard information limit of the fixed M-array
(``docs/cliff_plan.md`` step 7); the pattern's carrier params get gradient so it can learn a
multi-scale code that survives anamorphic compression.

Structure mirrors ``scripts/train_proj_net.py`` (same optimizer / cosine / TB+CSV logging /
AMP+NaN tripwire / snapshots); the differences are the synthetic data source, a second
optimizer param group for the generator (with a decoder warm-up), and pattern checkpointing.

    # Milestone 1 -- proxy fidelity: freeze the real M-array, confirm the proxy reproduces
    # the documented ~8% bin-acc floor at 60-75 deg (else the proxy is too clean).
    python scripts/train_codesign.py --pattern-init raster_marray --epochs 12 \
        --out checkpoints/codesign_marray.pt

    # Milestone 2 -- learn the pattern (decoder warms up 4 epochs, then joint).
    python scripts/train_codesign.py --pattern-init carrier_bank --n-carriers 12 \
        --gen-freeze-epochs 4 --epochs 30 --snapshots --out checkpoints/codesign.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lux import codesign as cd  # noqa: E402
from lux.proj_net import (  # noqa: E402
    ProjUNet, proj_loss, predict_tiled, save_checkpoint, N_BINS_U, N_BINS_V,
)
from scripts.train_proj_net import auto_device, _launch_tensorboard  # noqa: E402


@torch.no_grad()
def banded_eval(model, bank: cd.EvalBank, device: str, overlap: int = 64
                ) -> dict[str, tuple[float, float, float]]:
    """Per-obliquity-band (median |du| px, u-bin acc, v-bin acc) on the eval bank, using the
    same tiled center-crop inference as the real hemisphere bench. Returns {band_label: (...)}."""
    model.eval()
    binw_u, binw_v = bank.proj_wh[0] / N_BINS_U, bank.proj_wh[1] / N_BINS_V
    rows: dict[str, list] = {}
    for i in range(len(bank)):
        img, gt = bank.full(i)
        pred = predict_tiled(model, img, bank.proj_wh, device=device,
                             overlap=overlap, select="center")
        both = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
        if not both.any():
            continue
        du = np.abs((pred[..., 0] - gt[..., 0])[both])
        ub = (np.clip(pred[..., 0][both] // binw_u, 0, N_BINS_U - 1)
              == np.clip(gt[..., 0][both] // binw_u, 0, N_BINS_U - 1))
        vb = (np.clip(pred[..., 1][both] // binw_v, 0, N_BINS_V - 1)
              == np.clip(gt[..., 1][both] // binw_v, 0, N_BINS_V - 1))
        lab = bank.band_label(bank.obliq[i])
        rows.setdefault(lab, []).append((du, ub, vb))
    out = {}
    for lab, parts in sorted(rows.items()):
        du = np.concatenate([p[0] for p in parts])
        ub = np.concatenate([p[1] for p in parts])
        vb = np.concatenate([p[2] for p in parts])
        out[lab] = (float(np.median(du)), float(ub.mean()), float(vb.mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proj-w", type=int, default=1920)
    ap.add_argument("--proj-h", type=int, default=1080)
    ap.add_argument("--pattern-init", default="carrier_bank",
                    choices=["carrier_bank", "marray_fit", "raster_marray"],
                    help="carrier_bank/marray_fit: learnable PatternGenerator init; "
                         "raster_marray: fixed real M-array (proxy fidelity check, Milestone 1)")
    ap.add_argument("--n-carriers", type=int, default=12)
    ap.add_argument("--obliq-min", type=float, default=45.0)
    ap.add_argument("--obliq-max", type=float, default=75.0)
    ap.add_argument("--supersample", type=int, default=4,
                    help="area-integration supersample factor (anti-aliasing that scales with "
                         "grazing compression -- the M1 grazing-fidelity fix)")
    ap.add_argument("--base-psf", type=float, default=1.0,
                    help="always-on intrinsic camera PSF sigma (px), anchored to the rig's camera "
                         "MTF + projector defocus_px: erodes grazing-compressed cells, spares frontal")
    ap.add_argument("--grazing-floor", type=float, default=0.12,
                    help="grazing irradiance falloff: signal ~ max(cos(theta), floor); the dimmer "
                         "grazing signal makes shot noise bite (physical, compression-coupled hardener)")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable the differentiable photometric stack (debug only -- the "
                         "pattern would then optimize for a cleaner-than-real world)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4, help="decoder LR")
    ap.add_argument("--gen-lr", type=float, default=2e-3, help="pattern-generator LR (separate)")
    ap.add_argument("--gen-freeze-epochs", type=int, default=4,
                    help="train the decoder only for the first N epochs (pattern fixed at "
                         "init), then unfreeze for joint co-design -- avoids a random decoder "
                         "sending garbage gradients that collapse the carrier bank")
    ap.add_argument("--joint-from-start", action="store_true",
                    help="A/B: optimize pattern + decoder jointly from epoch 1 (no warm-up)")
    ap.add_argument("--gen-reg", type=float, default=0.02,
                    help="weight on the generator anti-collapse regularizer (carrier "
                         "repulsion + contrast floor)")
    ap.add_argument("--offset-weight", type=float, default=2.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--v-weight", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--mid", choices=["conv", "attn"], default="conv")
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--eval-per-band", type=int, default=8, help="eval-bank samples per band")
    ap.add_argument("--eval-overlap", type=int, default=64)
    ap.add_argument("--device", default=auto_device())
    ap.add_argument("--out", default="checkpoints/codesign.pt")
    ap.add_argument("--resume", default=None, help="warm-start decoder + generator from a ckpt")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--logdir", default="runs/codesign")
    ap.add_argument("--no-tensorboard", dest="launch_tb", action="store_false")
    ap.add_argument("--tb-port", type=int, default=6006)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    proj_wh = (args.proj_w, args.proj_h)
    dev = args.device

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(args.logdir)
    except ImportError:
        tb = None
    if args.launch_tb:
        _launch_tensorboard(args.logdir, args.tb_port)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.out).with_name("metrics.csv")
    if not csv_path.exists():
        csv_path.write_text("wall_time,epoch,step,tag,value\n")

    def log_scalar(tag: str, value: float, step: int, epoch: int):
        if tb is not None:
            tb.add_scalar(tag, value, step)
        with open(csv_path, "a") as f:
            f.write(f"{time.time():.0f},{epoch},{step},{tag},{value:.4f}\n")

    # -- pattern source ------------------------------------------------------
    learnable = args.pattern_init != "raster_marray"
    if learnable:
        gen = cd.PatternGenerator(proj_wh, n_carriers=args.n_carriers,
                                  init=args.pattern_init, n_bins=(N_BINS_U, N_BINS_V),
                                  seed=args.seed).to(dev)
    else:
        gen = cd.RasterPattern.from_marray(proj_wh).to(dev)
    batcher = cd.ProxyBatcher(proj_wh, crop=args.crop, device=dev,
                              obliq_deg=(args.obliq_min, args.obliq_max),
                              augment=args.augment, ss=args.supersample,
                              base_psf=args.base_psf, grazing_floor=args.grazing_floor,
                              seed=args.seed)
    bank = cd.EvalBank(gen.sample_at, proj_wh, per_band=args.eval_per_band,
                       device=dev, augment=args.augment, ss=args.supersample,
                       base_psf=args.base_psf, grazing_floor=args.grazing_floor)

    model = ProjUNet(base=args.base, mid=args.mid).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    gpar = sum(p.numel() for p in gen.parameters())
    print(f"ProjUNet base={args.base} mid={args.mid}: {n_par / 1e6:.2f}M params | "
          f"pattern {'learnable ' + args.pattern_init if learnable else 'FIXED raster_marray'} "
          f"({gpar} params) | obliq {args.obliq_min:.0f}-{args.obliq_max:.0f} deg | device {dev}",
          flush=True)

    if args.resume:
        from lux.proj_net import load_weights_compatible
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        load_weights_compatible(model, ck["state"])
        if learnable and ck.get("gen_state"):
            gen.load_state_dict(ck["gen_state"])
        print(f"resumed from {args.resume}", flush=True)

    groups = [{"params": model.parameters(), "lr": args.lr, "weight_decay": 1e-4}]
    if learnable:
        groups.append({"params": gen.parameters(), "lr": args.gen_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(dev, enabled=args.amp)

    def materialize_pattern(path_pt: str):
        """Dump the current pattern as a PNG set feedable to gen_rasterizer_dataset.py."""
        from lux import io
        out_dir = Path(path_pt).with_suffix("")
        io.save_image_stack(str(out_dir.parent / f"{out_dir.name}_pattern"),
                            gen.materialize(proj_wh), prefix="pat")

    best, gstep = np.inf, 0
    bad_steps = 0
    for ep in range(1, args.epochs + 1):
        unfrozen = learnable and (args.joint_from_start or ep > args.gen_freeze_epochs)
        if learnable:
            gen.requires_grad_(unfrozen)
        model.train()
        t0 = time.time()
        w_loss = w_du = w_ub = w_vb = w_reg = 0.0
        tw = time.time()
        for k in range(1, args.steps_per_epoch + 1):
            cap, target, valid = batcher.sample(args.batch, gen.sample_at)
            with torch.autocast(dev, dtype=torch.float16, enabled=args.amp):
                loss, du_px, dv_px, _, ubacc, vbacc = proj_loss(
                    model(cap), target, valid, proj_wh, offset_weight=args.offset_weight,
                    focal_gamma=args.focal_gamma, v_weight=args.v_weight)
                reg = gen.regularizer() if unfrozen else cap.new_zeros(())
                total = loss + args.gen_reg * reg
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], args.grad_clip)
            scaler.step(opt)
            scaler.update()
            lval = total.item()
            if not np.isfinite(lval):
                bad_steps += 1
                if bad_steps >= 30:
                    raise SystemExit(f"ABORT: {bad_steps} non-finite losses (ep {ep} step {k})")
                continue
            bad_steps = 0
            w_loss += lval; w_du += du_px.item(); w_ub += ubacc.item()
            w_vb += vbacc.item(); w_reg += float(reg.detach())
            gstep += 1
            if k % args.log_every == 0:
                e = args.log_every
                print(f"  ep {ep:3d} step {k:4d}/{args.steps_per_epoch}  loss {w_loss / e:8.4f}  "
                      f"|du| {w_du / e:6.2f}px  bin u {w_ub / e * 100:4.1f}% v {w_vb / e * 100:4.1f}%  "
                      f"reg {w_reg / e:.3f}  {e * args.batch / (time.time() - tw):5.1f} img/s",
                      flush=True)
                log_scalar("train/loss", w_loss / e, gstep, ep)
                log_scalar("train/du_px", w_du / e, gstep, ep)
                log_scalar("train/bin_acc", w_ub / e, gstep, ep)
                log_scalar("train/vbin_acc", w_vb / e, gstep, ep)
                log_scalar("train/gen_reg", w_reg / e, gstep, ep)
                w_loss = w_du = w_ub = w_vb = w_reg = 0.0
                tw = time.time()
        sched.step()

        bands = banded_eval(model, bank, dev, overlap=args.eval_overlap)
        graze = bands.get("60-75", (np.inf, 0.0, 0.0))
        for lab, (du, ub, vb) in bands.items():
            log_scalar(f"val/du_px[{lab}]", du, gstep, ep)
            log_scalar(f"val/ubin[{lab}]", ub, gstep, ep)
            log_scalar(f"val/vbin[{lab}]", vb, gstep, ep)
        log_scalar("train/lr", sched.get_last_lr()[0], gstep, ep)
        band_str = "  ".join(f"{lab}:{ub * 100:.0f}/{vb * 100:.0f}%"
                             for lab, (_, ub, vb) in bands.items())
        # selection metric: grazing-band u-bin accuracy (the leg's whole point)
        score = -graze[1]
        meta = {"epoch": ep, "bands": {k: list(v) for k, v in bands.items()},
                "pattern_init": args.pattern_init, "unfrozen": unfrozen}
        flag = " warmup" if (learnable and not unfrozen) else ""
        if score < best:
            best = score
            save_checkpoint(args.out, model, proj_wh, meta=meta)
            if learnable:
                ck = torch.load(args.out, weights_only=False)
                ck["gen_state"] = gen.state_dict()
                ck["gen_cfg"] = {"n_carriers": args.n_carriers, "proj_wh": proj_wh}
                torch.save(ck, args.out)
            materialize_pattern(args.out)
            flag += " *best*"
        if args.snapshots:
            stem = Path(args.out)
            snap = str(stem.with_name(f"{stem.stem}_ep{ep:02d}.pt"))
            save_checkpoint(snap, model, proj_wh, meta=meta)
            save_checkpoint(str(stem.with_name(f"{stem.stem}_last.pt")), model, proj_wh, meta=meta)
            flag += " +snap"
        print(f"epoch {ep:3d}  bin(u/v) {band_str}  graze60-75 u {graze[1] * 100:.1f}% "
              f"med|du| {graze[0]:.1f}px  ({time.time() - t0:.1f}s){flag}", flush=True)

    print(f"\nbest grazing-band u-bin acc {-best * 100:.1f}% -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
