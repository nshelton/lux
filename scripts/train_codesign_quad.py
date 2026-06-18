#!/usr/bin/env python3
"""Build 4: end-to-end co-design of the hierarchical coprime PATTERN + continuous-phase DECODER.

Frozen-coprime-period ladder generator (learns amp/phase/bias) + quadrature (cos,sin) decoder head,
trained on the faithful homography proxy (analytic anisotropic-Gaussian blur + grazing falloff +
noise). Decoded by the CRT consensus vote (lux.codesign_vote). Schedule (reviewer-locked):
  - coarse-first carrier weighting (coarsest=1, finer ramp 0.3->1 over --ramp steps);
  - phase-only until the coord-L1 GATE fires: min(coarse u-align, coarse v-align) >= --coord-gate
    (slower axis), then ramp coord-L1 weight 0->--coord-weight over --coord-ramp steps;
  - coord-L1 = |soft-vote coord - gt|, MASKED to pixels where the soft-vote period agrees with the
    coarse-carrier direct estimate (wrong-unwrap pixels don't contribute a misleading gradient).

Per-band eval decodes via the fast CRT vote and reports bin-acc + median|du| vs obliquity (A/B vs
the M-array+hard-bin baseline -- report as "the co-designed PAIR lifts grazing").

    python scripts/train_codesign_quad.py --epochs 24 --steps-per-epoch 250 --snapshots
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
from lux.codesign_vote import vote_fast, consensus_vote  # noqa: E402
from lux.proj_net import ProjUNet, N_BINS_U, N_BINS_V  # noqa: E402
from scripts.train_proj_net import auto_device, _launch_tensorboard  # noqa: E402

U_PERIODS = [13, 19, 33, 139]
V_PERIODS = [11, 17, 29, 113]


def carrier_weights(step, ramp):
    fg = min(step / max(ramp, 1), 1.0)
    w = []
    for blk in (U_PERIODS, V_PERIODS):
        for i in range(len(blk)):
            w.append(1.0 if i == len(blk) - 1 else 0.3 + 0.7 * fg)
    return w


def decode(pred, gen, proj_wh, device):
    """quad head -> per-carrier (psi,mag) -> fast CRT vote -> (uv (H,W,2), margin)."""
    nu, nv = gen.n_u, gen.n_v
    cu, su = pred[:2 * nu:2], pred[1:2 * nu:2]
    cv, sv = pred[2 * nu:2 * nu + 2 * nv:2], pred[2 * nu + 1:2 * nu + 2 * nv:2]
    psi_u = torch.atan2(su, cu).permute(1, 2, 0)              # (H,W,nu)
    mag_u = torch.sqrt(cu ** 2 + su ** 2).permute(1, 2, 0)
    psi_v = torch.atan2(sv, cv).permute(1, 2, 0)
    mag_v = torch.sqrt(cv ** 2 + sv ** 2).permute(1, 2, 0)
    u, mu = vote_fast(psi_u, mag_u, U_PERIODS, proj_wh[0])
    v, mv = vote_fast(psi_v, mag_v, V_PERIODS, proj_wh[1])
    valid = pred[-1] > 0.0
    uv = torch.stack([u, v], -1)
    uv = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan")))
    return uv.cpu().numpy(), torch.minimum(mu, mv).cpu().numpy()


@torch.no_grad()
def banded_eval(model, gen, bank, device):
    model.eval()
    binu, binv = bank.proj_wh[0] / N_BINS_U, bank.proj_wh[1] / N_BINS_V
    rows = {}
    for i in range(len(bank)):
        img, gt = bank.full(i)
        H, W = img.shape
        ph, pw = (-H) % 16, (-W) % 16
        x = torch.from_numpy(img.astype(np.float32))[None, None]
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect").to(device)
        pred = model(x)[0, :, :H, :W]
        uv, _ = decode(pred, gen, bank.proj_wh, device)
        both = np.isfinite(gt[..., 0]) & np.isfinite(uv[..., 0])
        if both.sum() < 1000:
            continue
        du = np.abs((uv[..., 0] - gt[..., 0])[both])
        ub = (np.clip(uv[..., 0][both] // binu, 0, N_BINS_U - 1) == np.clip(gt[..., 0][both] // binu, 0, N_BINS_U - 1))
        vb = (np.clip(uv[..., 1][both] // binv, 0, N_BINS_V - 1) == np.clip(gt[..., 1][both] // binv, 0, N_BINS_V - 1))
        rows.setdefault(bank.band_label(bank.obliq[i]), []).append((du, ub, vb))
    out = {}
    for lab, parts in sorted(rows.items()):
        du = np.concatenate([p[0] for p in parts])
        out[lab] = (float(np.median(du)), float(np.concatenate([p[1] for p in parts]).mean()),
                    float(np.concatenate([p[2] for p in parts]).mean()))
    return out


def coord_l1(pred, target, valid, gen, proj_wh, n_sample=1024):
    """Masked soft-vote coordinate L1 on a random subset of valid pixels (the vote is expensive;
    this is a low-weight refiner). Mask = soft-vote period agrees with the coarse-carrier direct
    estimate, so wrong-unwrap pixels contribute no gradient."""
    nu, nv = gen.n_u, gen.n_v
    B = pred.shape[0]
    m = valid[:, 0] > 0.5
    idx = m.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return pred.new_zeros(())
    sel = idx[torch.randint(0, idx.shape[0], (min(n_sample, idx.shape[0]),), device=pred.device)]
    bb, yy, xx = sel[:, 0], sel[:, 1], sel[:, 2]
    px = pred[bb, :, yy, xx]                                  # (n, C)
    W, H = proj_wh
    loss = pred.new_zeros(())
    for axis, (periods, no, off, ext) in enumerate(
            [(U_PERIODS, nu, 0, W), (V_PERIODS, nv, 2 * nu, H)]):
        c = px[:, off:off + 2 * no:2]
        s = px[:, off + 1:off + 2 * no:2]
        psi = torch.atan2(s, c)
        mag = torch.sqrt(c ** 2 + s ** 2)
        u_soft, _ = consensus_vote(psi, mag, periods, ext, step=0.5, soft=True, temp=0.1)
        gt = target[bb, axis, yy, xx] * ext
        # mask to pixels where the soft-vote selected the RIGHT period (within +/- P0/2 of GT) so
        # wrong-unwrap pixels (which the coord-L1 would push the wrong way) contribute no gradient.
        P0 = max(periods)
        agree = (torch.abs(u_soft - gt) < P0 / 2).float()
        d = torch.abs(u_soft - gt)
        loss = loss + (d * agree).sum() / agree.sum().clamp(min=1.0)
    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--steps-per-epoch", type=int, default=250)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gen-lr", type=float, default=2e-3)
    ap.add_argument("--ramp", type=int, default=600, help="coarse->fine carrier-weight ramp (steps)")
    ap.add_argument("--coord-gate", type=float, default=0.97, help="min coarse align to start coord-L1")
    ap.add_argument("--coord-weight", type=float, default=0.05, help="target coord-L1 weight (low)")
    ap.add_argument("--coord-ramp", type=int, default=400, help="coord-L1 warmup steps after gate")
    ap.add_argument("--sigma-mtf", type=float, default=0.7)
    ap.add_argument("--sigma-def", type=float, default=1.0)
    ap.add_argument("--appearance", choices=["raster", "analytic"], default="raster",
                    help="raster: appearance-fixed proxy (materialize -> 8-bit quantize -> "
                         "grid_sample = the renderer's quantized/resampled appearance, optics as "
                         "proj-defocus blur + camera-MTF base_psf + footprint-ss anisotropy); "
                         "analytic: pre-fix continuous-carrier eval with per-carrier MTF (old proxy)")
    ap.add_argument("--eval-per-band", type=int, default=6)
    ap.add_argument("--device", default=auto_device())
    ap.add_argument("--out", default="checkpoints/codesign_quad.pt")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--logdir", default="runs/codesign_quad")
    ap.add_argument("--no-tensorboard", dest="launch_tb", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    proj_wh = (1920, 1080)
    mtf = (args.sigma_mtf, args.sigma_def)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(args.logdir)
    except ImportError:
        tb = None
    if args.launch_tb:
        _launch_tensorboard(args.logdir, 6006)
    csv = Path(args.out).with_name("metrics_quad.csv")
    if not csv.exists():
        csv.write_text("wall,epoch,step,tag,value\n")

    def log(tag, val, step, ep):
        if tb:
            tb.add_scalar(tag, val, step)
        with open(csv, "a") as f:
            f.write(f"{time.time():.0f},{ep},{step},{tag},{val:.4f}\n")

    gen = cd.PatternGenerator.ladder(proj_wh, U_PERIODS, V_PERIODS, seed=args.seed).to(dev)
    model = ProjUNet(base=32, head="quad", n_cu=len(U_PERIODS), n_cv=len(V_PERIODS)).to(dev)
    if args.appearance == "raster":
        # appearance-fixed: sample the quantized pattern raster via grid_sample; optics split into
        # projector-defocus blur (inside pat_at), camera-MTF base_psf, and footprint-ss anisotropy.
        pat_at = cd.raster_appearance(gen, sigma_def=args.sigma_def)
        batcher = cd.ProxyBatcher(proj_wh, crop=args.crop, device=dev, obliq_deg=(0, 78),
                                  seed=args.seed, ss=4, base_psf=args.sigma_mtf, grazing_floor=0.12)
        bank = cd.EvalBank(pat_at, proj_wh, per_band=args.eval_per_band, device=dev,
                           ss=4, base_psf=args.sigma_mtf)
        sample_kw = {}
    else:
        pat_at = gen.sample_at
        batcher = cd.ProxyBatcher(proj_wh, crop=args.crop, device=dev, obliq_deg=(0, 78), seed=args.seed)
        bank = cd.EvalBank(lambda c: gen.sample_at(c, mtf=mtf), proj_wh, per_band=args.eval_per_band,
                           device=dev)
        sample_kw = {"mtf": mtf}
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": args.lr, "weight_decay": 1e-4},
                             {"params": [p for p in gen.parameters() if p.requires_grad],
                              "lr": args.gen_lr, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"co-design[{args.appearance}]: u={U_PERIODS} v={V_PERIODS} | quad head "
          f"{2*(len(U_PERIODS)+len(V_PERIODS))+1}ch | device {dev}", flush=True)

    gstep = 0
    gate_step = None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        w_loss = w_cl = 0.0
        last_cu = last_cv = 0.0
        for k in range(1, args.steps_per_epoch + 1):
            cap, target, valid = batcher.sample(args.batch, pat_at, **sample_kw)
            pred = model(cap)
            loss, ua, va, bce = cd.quad_loss(pred, target, valid, gen, proj_wh,
                                             carrier_weights=carrier_weights(gstep, args.ramp))
            last_cu, last_cv = ua[-1].item(), va[-1].item()
            if gate_step is None and min(last_cu, last_cv) >= args.coord_gate:
                gate_step = gstep
                print(f"  coord-L1 GATE fired at gstep {gstep} (coarse u {last_cu:.3f} v {last_cv:.3f})", flush=True)
            cw = 0.0
            if gate_step is not None:
                cw = args.coord_weight * min((gstep - gate_step) / max(args.coord_ramp, 1), 1.0)
            if cw > 0:
                cl = coord_l1(pred, target, valid, gen, proj_wh)
                loss = loss + cw * cl
                w_cl += float(cl.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            w_loss += float(loss.detach())
            gstep += 1
            if k % 50 == 0:
                us = " ".join(f"{x:.2f}" for x in ua.tolist())
                vs = " ".join(f"{x:.2f}" for x in va.tolist())
                print(f"  ep {ep} {k}/{args.steps_per_epoch} loss {w_loss/50:.3f} coordL1 {w_cl/50:.3f} "
                      f"cw {cw:.3f} | u[{us}] v[{vs}]", flush=True)
                log("train/loss", w_loss / 50, gstep, ep)
                log("train/coord_l1", w_cl / 50, gstep, ep)
                for j, p in enumerate(U_PERIODS):
                    log(f"align/u{p}", ua[j].item(), gstep, ep)
                for j, p in enumerate(V_PERIODS):
                    log(f"align/v{p}", va[j].item(), gstep, ep)
                w_loss = w_cl = 0.0
        sched.step()
        bands = banded_eval(model, gen, bank, dev)
        bstr = "  ".join(f"{lab}:{ub*100:.0f}/{vb*100:.0f}%(du{du:.1f})" for lab, (du, ub, vb) in bands.items())
        for lab, (du, ub, vb) in bands.items():
            log(f"val/ubin[{lab}]", ub, gstep, ep); log(f"val/vbin[{lab}]", vb, gstep, ep)
            log(f"val/du[{lab}]", du, gstep, ep)
        graze = bands.get("60-75", (9, 0, 0))
        meta = {"epoch": ep, "bands": {k: list(v) for k, v in bands.items()},
                "u_periods": U_PERIODS, "v_periods": V_PERIODS, "gate_step": gate_step}
        ck = {"state": model.state_dict(), "gen_state": gen.state_dict(), "proj_wh": proj_wh,
              "head": "quad", "n_cu": len(U_PERIODS), "n_cv": len(V_PERIODS), "meta": meta}
        torch.save(ck, args.out)
        if args.snapshots:
            torch.save(ck, str(Path(args.out).with_name(f"{Path(args.out).stem}_ep{ep:02d}.pt")))
        from lux import io
        io.save_image_stack(str(Path(args.out).with_name(f"{Path(args.out).stem}_pattern")),
                            gen.materialize(proj_wh), prefix="pat")
        print(f"epoch {ep:3d}  {bstr}  graze {graze[1]*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
