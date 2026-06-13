#!/usr/bin/env python3
"""Pack a rendered training set into a memory-mapped "loaf" for fast training.

Reads every ``sample_*`` folder under --data and writes three files to --out:
``caps.npy`` (N, H, W) uint8 captures, ``gt.npy`` (N, H, W, 2) uint16
fixed-point projector coords (0xFFFF = invalid, resolution ~0.03 px), and
``meta.json``. Train with ``train_proj_net.py --loaf <out>`` — random crops
then read ~130 KB of mmap pages instead of decoding a 25 MB sample.

    python scripts/build_loaf.py --data renders/val --out renders/val_loaf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lux.proj_net import build_loaf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="renders/train", help="sample folders root")
    ap.add_argument("--out", default=None, help="loaf dir (default: <data>_loaf)")
    ap.add_argument("--pattern-set", default="marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    args = ap.parse_args()
    out = args.out or f"{args.data.rstrip('/')}_loaf"
    build_loaf(args.data, out, args.pattern_set, args.frame)
    print(f"loaf -> {out}/ (caps.npy, gt.npy, meta.json)")


if __name__ == "__main__":
    main()
