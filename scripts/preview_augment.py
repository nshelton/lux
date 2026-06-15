#!/usr/bin/env python3
"""Visual sanity check for the training-crop augmentation.

Pulls random crops from a loaf (default the planar set — the oblique cells are
the hardest case for blur), applies the *exact* train-time augmentation
(`lux.proj_net._augment_crop`), and writes two montages:

  aug_grid.png   N augmented crops tiled (overall feel of the distribution)
  aug_pairs.png  clean (top) vs augmented (bottom) for a few crops, full-res,
                 so you can see whether the 4 px M-array cells survive

    python scripts/preview_augment.py --loaf renders/planar_loaf --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux import io  # noqa: E402
from lux.proj_net import _augment_crop, _GT_SENTINEL  # noqa: E402


def _grid(tiles: list[np.ndarray], cols: int, pad: int = 2) -> np.ndarray:
    """Tile equal-size 2D crops into a grid with a light border between them."""
    h, w = tiles[0].shape
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (h + pad) - pad, cols * (w + pad) - pad), 0.15)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = t
    return np.clip(out, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loaf", default="renders/planar_loaf")
    ap.add_argument("--n", type=int, default=100, help="crops in the grid")
    ap.add_argument("--pairs", type=int, default=6, help="clean/aug comparison pairs")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp")
    args = ap.parse_args()

    meta = json.loads((Path(args.loaf) / "meta.json").read_text())
    caps = np.load(Path(args.loaf) / "caps.npy", mmap_mode="r")
    gts = np.load(Path(args.loaf) / "gt.npy", mmap_mode="r")
    N, H, W = caps.shape
    S = args.crop
    rng = np.random.default_rng(args.seed)

    def random_valid_crop():
        for _ in range(20):
            i = int(rng.integers(0, N))
            y, x = int(rng.integers(0, H - S + 1)), int(rng.integers(0, W - S + 1))
            v = gts[i, y:y + S, x:x + S, 0] != _GT_SENTINEL
            if v.mean() > 0.3:                       # a crop that mostly sees the surface
                return caps[i, y:y + S, x:x + S].astype(np.float32) / 255.0
        return caps[i, y:y + S, x:x + S].astype(np.float32) / 255.0

    aug = [_augment_crop(random_valid_crop(), rng) for _ in range(args.n)]
    cols = int(np.ceil(np.sqrt(args.n)))
    io.save_image(str(Path(args.out) / "aug_grid.png"), _grid(aug, cols))
    print(f"grid: {args.n} augmented crops -> {args.out}/aug_grid.png")

    # clean (top row) vs augmented (bottom row), full-res, same crop.
    cleans, augs = [], []
    for _ in range(args.pairs):
        c = random_valid_crop()
        cleans.append(c)
        augs.append(_augment_crop(c.copy(), rng))
    top = _grid(cleans, args.pairs)
    bot = _grid(augs, args.pairs)
    gap = np.full((6, top.shape[1]), 0.4)
    io.save_image(str(Path(args.out) / "aug_pairs.png"), np.vstack([top, gap, bot]))
    print(f"pairs: clean (top) vs augmented (bottom) -> {args.out}/aug_pairs.png")


if __name__ == "__main__":
    main()
