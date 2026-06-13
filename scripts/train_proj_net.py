#!/usr/bin/env python3
"""Train the one-shot M-array correspondence U-Net (lux/proj_net.py).

Maps a single capture (``<sample>/marray/cap_pat_00.png``) to the dense
projector correspondence ``gt_proj.npy`` + validity. Trains on random crops
from ``renders/train``-style folders; the first ``--val`` samples are held out
and reported as median |du| in projector px (the column error that drives
triangulation accuracy).

    python scripts/train_proj_net.py --data renders/train --epochs 30
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

from lux.proj_net import (  # noqa: E402
    ProjUNet, ProjSamples, LoafSamples, ConcatLoaf, proj_loss, predict_full,
    save_checkpoint, load_weights_compatible,
)


def auto_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, ds, val_idx, device) -> tuple[float, float, float, float]:
    """Median |du|, |dv| (px), validity IoU and u-bin accuracy, full-frame."""
    from lux.proj_net import N_BINS_U
    du, dv, ious, bacc = [], [], [], []
    for i in val_idx:
        img, gt = ds.full(i)
        pred = predict_full(model, img, ds.proj_wh, device=device)
        both = np.isfinite(gt[..., 0]) & np.isfinite(pred[..., 0])
        if both.any():
            du.append(np.abs((pred[..., 0] - gt[..., 0])[both]))
            dv.append(np.abs((pred[..., 1] - gt[..., 1])[both]))
            binw = ds.proj_wh[0] / N_BINS_U
            pb = np.clip(pred[..., 0][both] // binw, 0, N_BINS_U - 1)
            gb = np.clip(gt[..., 0][both] // binw, 0, N_BINS_U - 1)
            bacc.append(pb == gb)
        union = np.isfinite(gt[..., 0]) | np.isfinite(pred[..., 0])
        ious.append(both.sum() / max(union.sum(), 1))
    du = np.concatenate(du) if du else np.array([np.inf])
    dv = np.concatenate(dv) if dv else np.array([np.inf])
    bacc = np.concatenate(bacc) if bacc else np.array([0.0])
    return (float(np.median(du)), float(np.median(dv)),
            float(np.mean(ious)), float(bacc.mean()))


def _launch_tensorboard(logdir: str, port: int) -> None:
    """Spawn a detached TensorBoard on the parent of logdir (so sibling runs
    show up too), unless something is already serving on the port."""
    import socket
    import subprocess

    root = str(Path(logdir).parent or ".")
    with socket.socket() as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            print(f"TensorBoard already at http://localhost:{port} (logdir {root})", flush=True)
            return
    tb_bin = Path(sys.executable).with_name("tensorboard")
    cmd = [str(tb_bin)] if tb_bin.exists() else [sys.executable, "-m", "tensorboard.main"]
    cmd += ["--logdir", root, "--port", str(port)]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"TensorBoard launched at http://localhost:{port} (logdir {root})", flush=True)
    except Exception as e:  # noqa: BLE001 - launch is best-effort
        print(f"TensorBoard launch failed ({e}); start it manually if you want curves", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="renders/train")
    ap.add_argument("--loaf", default=None, nargs="+",
                    help="one or more memmapped loaf dirs (scripts/build_loaf.py); "
                         "overrides --data and makes the data path ~free. Multiple "
                         "loaves train as one (e.g. the original set + the planar set); "
                         "they mix in proportion to their sizes and the first --val of "
                         "each is held out for eval. All must share projector dims.")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast (MPS/CUDA) - ~1.5x faster forward/backward")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops-per-sample", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="cosine anneal floor; >0 keeps learning alive at end of "
                         "schedule instead of decaying to zero")
    ap.add_argument("--offset-weight", type=float, default=2.0,
                    help="weight of the within-bin offset L1 term (raise once "
                         "bin accuracy saturates to push subpixel learning)")
    ap.add_argument("--gate-offset", type=float, default=0.0,
                    help="self-paced curriculum: add up to this much offset weight "
                         "as train bin accuracy rises from 70%% to 95%% — the micro "
                         "task (offset) earns gradient share as its prerequisite "
                         "(bin classification) is met; 0 disables")
    ap.add_argument("--base", type=int, default=32, help="U-Net width multiplier")
    ap.add_argument("--mid", choices=["conv", "attn"], default="conv",
                    help="bottleneck: conv block or transformer (global attention at 1/16)")
    ap.add_argument("--val", type=int, default=1, help="samples held out for eval")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of training samples (quick runs)")
    ap.add_argument("--device", default=auto_device())
    ap.add_argument("--out", default="checkpoints/proj_net.pt")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to warm-start the weights from")
    ap.add_argument("--log-every", type=int, default=20,
                    help="steps between intra-epoch log lines / TB points")
    ap.add_argument("--logdir", default="runs/proj_net",
                    help="TensorBoard run dir (view: tensorboard --logdir runs)")
    ap.add_argument("--no-tensorboard", dest="launch_tb", action="store_false",
                    help="don't auto-launch a TensorBoard server")
    ap.add_argument("--tb-port", type=int, default=6006,
                    help="port for the auto-launched TensorBoard server")
    args = ap.parse_args()

    # Three monitoring layers: flushed stdout (tail -f the log), TensorBoard
    # scalars (live curves), and a plain CSV (checkpoints/metrics.csv).
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(args.logdir)
    except ImportError:
        tb = None
    if args.launch_tb:
        _launch_tensorboard(args.logdir, args.tb_port)
    csv_path = Path(args.out).with_name("metrics.csv")
    if not csv_path.exists():
        csv_path.write_text("wall_time,epoch,step,tag,value\n")

    def log_scalar(tag: str, value: float, step: int, epoch: int):
        if tb is not None:
            tb.add_scalar(tag, value, step)
        with open(csv_path, "a") as f:
            f.write(f"{time.time():.0f},{epoch},{step},{tag},{value:.4f}\n")

    if args.loaf:
        loaves = [LoafSamples(p, crop=args.crop, crops_per_sample=args.crops_per_sample)
                  for p in args.loaf]
        ds = loaves[0] if len(loaves) == 1 else ConcatLoaf(loaves)
        names = ds.names
        # Hold out the first --val of *each* loaf so eval spans every domain.
        starts = ds.part_starts() if isinstance(ds, ConcatLoaf) else [0]
    else:
        ds = ProjSamples(args.data, args.pattern_set, args.frame,
                         crop=args.crop, crops_per_sample=args.crops_per_sample)
        names = [d.name for d in ds.dirs]
        starts = [0]
    val_idx = [s + j for s in starts for j in range(args.val)]
    val_set = set(val_idx)
    train_idx = [i for i in range(len(ds)) if i not in val_set]
    if args.limit:
        train_idx = train_idx[:args.limit]
    train_set = torch.utils.data.Subset(ds, train_idx)
    if len(train_set) == 0:
        raise SystemExit("need more samples than --val")
    src = f"{len(args.loaf)} loaves" if args.loaf and len(args.loaf) > 1 else "1 source"
    print(f"train {len(train_set)} samples x {args.crops_per_sample} crops ({src}), "
          f"val {[names[i] for i in val_idx]}, device {args.device}, amp {args.amp}")

    model = ProjUNet(base=args.base, mid=args.mid).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"ProjUNet base={args.base} mid={args.mid}: {n_par / 1e6:.2f}M params")
    resume_best = np.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        n_loaded = load_weights_compatible(model, ck["state"])
        n_total = len(model.state_dict())
        # Only inherit the best-val bar on a *full* resume; after a head swap the
        # old val number is meaningless (and would block early checkpoints).
        if args.resume == args.out and n_loaded == n_total:
            resume_best = ck["meta"].get("val_median_du_px", np.inf)
        print(f"resumed {n_loaded}/{n_total} tensors from {args.resume} "
              f"(epoch {ck['meta'].get('epoch')}, "
              f"val {ck['meta'].get('val_median_du_px', float('nan')):.1f}px)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr_min)
    # One dataset item = one sample folder carrying `crops_per_sample` crops, so
    # the DataLoader batch dim counts samples; flatten (B, P, ...) -> (B*P, ...).
    loader = DataLoader(train_set, batch_size=max(1, args.batch // args.crops_per_sample),
                        shuffle=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    scaler = torch.amp.GradScaler(args.device, enabled=args.amp)

    best, gstep = resume_best, 0
    off_w = args.offset_weight            # gated upward as bin accuracy is earned
    ep_bin = 0.0
    for ep in range(1, args.epochs + 1):
        if args.gate_offset > 0:
            gate = min(max((ep_bin - 0.70) / 0.25, 0.0), 1.0)
            off_w = args.offset_weight + args.gate_offset * gate
        model.train()
        t0, tot, l1s, bins = time.time(), 0.0, 0.0, 0.0
        w_loss, w_l1, w_bin, tw = 0.0, 0.0, 0.0, time.time()   # window since last log line
        for k, (img, target, valid) in enumerate(loader, 1):
            img, target, valid = (t.flatten(0, 1).to(args.device)
                                  for t in (img, target, valid))
            with torch.autocast(args.device, dtype=torch.float16, enabled=args.amp):
                loss, l1_px, _, bin_acc = proj_loss(model(img), target, valid, ds.proj_wh,
                                                    offset_weight=off_w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            l1s += l1_px.item()
            bins += bin_acc.item()
            w_loss += loss.item()
            w_l1 += l1_px.item()
            w_bin += bin_acc.item()
            gstep += 1
            if k % args.log_every == 0:
                n_img = args.log_every * img.shape[0]
                print(f"  ep {ep:3d} step {k:4d}/{len(loader)}  "
                      f"loss {w_loss / args.log_every:8.4f}  "
                      f"|du| {w_l1 / args.log_every:7.2f}px  "
                      f"bin {w_bin / args.log_every * 100:5.1f}%  "
                      f"{n_img / (time.time() - tw):5.1f} img/s", flush=True)
                log_scalar("train/loss", w_loss / args.log_every, gstep, ep)
                log_scalar("train/du_px", w_l1 / args.log_every, gstep, ep)
                log_scalar("train/bin_acc", w_bin / args.log_every, gstep, ep)
                w_loss, w_l1, w_bin, tw = 0.0, 0.0, 0.0, time.time()
        sched.step()
        ep_bin = bins / len(loader)              # feeds next epoch's offset gate
        log_scalar("train/offset_weight", off_w, gstep, ep)
        med, medv, iou, vbin = evaluate(model, ds, val_idx, args.device)
        log_scalar("val/median_du_px", med, gstep, ep)
        log_scalar("val/median_dv_px", medv, gstep, ep)
        log_scalar("val/valid_iou", iou, gstep, ep)
        log_scalar("val/bin_acc", vbin, gstep, ep)
        log_scalar("train/lr", sched.get_last_lr()[0], gstep, ep)
        n = len(loader)
        flag = ""
        if med < best:
            best = med
            save_checkpoint(args.out, model, ds.proj_wh,
                            meta={"epoch": ep, "val_median_du_px": med,
                                  "pattern_set": args.pattern_set, "base": args.base})
            flag = "  *saved*"
        print(f"epoch {ep:3d}  loss {tot / n:7.4f}  train|du| {l1s / n:7.2f}px  "
              f"val median|du| {med:7.2f}px |dv| {medv:7.2f}px  bin {vbin * 100:.1f}%  "
              f"valid-IoU {iou:.3f}  ({time.time() - t0:.1f}s){flag}", flush=True)

    print(f"\nbest val median|du| {best:.2f}px -> {args.out}")


if __name__ == "__main__":
    main()
