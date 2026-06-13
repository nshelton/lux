#!/usr/bin/env python3
"""Train the per-pixel neural decoder on synthetic captures.

Generates (image-stack -> GT projector-column) supervision from the simulator
across all scenes, then fits the small PixelMLP. Saves a checkpoint that
``NeuralMethod`` loads automatically.

    python scripts/train_neural.py --epochs 30 --out checkpoints/neural.pt
    python scripts/run_benchmark.py --methods neural   # uses the checkpoint via env or edit factory

Note: the neural method falls back to phase-shift when no checkpoint is loaded,
so the benchmark always runs even before you train.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux.geometry import Intrinsics, Rig  # noqa: E402
from lux.methods.neural import make_patterns  # noqa: E402
from lux.render import RenderConfig, render  # noqa: E402
from lux.scene import SCENES, build_scene  # noqa: E402


def build_dataset(width: int, height: int, baseline: float, noise: float):
    cam = Intrinsics.from_fov(width, height, 50.0)
    proj = Intrinsics.from_fov(width, height, 45.0)
    rig = Rig.make(cam, proj, baseline=baseline)
    patterns = make_patterns(width, height)
    cfg = RenderConfig(read_noise=noise, shot_noise=0.01)

    X, Y = [], []
    for name in SCENES:
        scene = build_scene(name, cam)
        cap = render(scene, patterns, rig, cfg)
        lit = cap.lit_mask
        feats = cap.images.transpose(1, 2, 0)[lit]          # (M, C)
        target = (cap.gt_proj_col[lit] / (width - 1)).astype(np.float32)  # (M,) in [0,1]
        X.append(feats.astype(np.float32))
        Y.append(target)
    return np.concatenate(X), np.concatenate(Y), patterns.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--baseline", type=float, default=0.12)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/neural.pt")
    args = ap.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        print("torch is not installed; `pip install torch` to train the neural method.")
        sys.exit(1)

    from lux.methods.neural import PixelMLP

    X, Y, n_ch = build_dataset(args.width, args.height, args.baseline, args.noise)
    print(f"dataset: {len(X)} pixels, {n_ch} channels")

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y[:, None]))
    dl = DataLoader(ds, batch_size=8192, shuffle=True)
    model = PixelMLP(n_ch, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss()

    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {total / len(ds):.6f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_channels": n_ch, "hidden": args.hidden}, out)
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
