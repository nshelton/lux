#!/usr/bin/env python3
"""Add an M-array capture beside the co-designed pattern's capture in an existing
hemisphere sample set, on **bit-identical geometry**.

The §9.5 matched-down baseline needs the M-array pattern rendered through the exact
same scenes/poses the co-designed quad trained on. Rather than re-sampling geometry
(which would rely on seed+param determinism), this re-renders each sample straight
from its saved ``scene.json`` / ``rig.json`` — so the ONLY difference from the quad's
training data is the projected pattern PNG. ``render_sample`` re-writes the
geometry-derived GT (gt_depth/gt_proj/white — identical, deterministic from geometry)
and adds a ``marray/`` capture folder beside the existing pattern-set folder.

Idempotent: skips samples that already have the M-array capture.

    python scripts/add_marray_captures.py --data evals/hemisphere/data_learned_train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_training_data import render_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="evals/hemisphere/data_learned_train")
    ap.add_argument("--patterns", default="patterns/marray")
    ap.add_argument("--frame", default="cap_pat_00.png")
    args = ap.parse_args()

    pset = Path(args.patterns).name
    dirs = sorted(d for d in Path(args.data).glob("sample_*")
                  if (d / "scene.json").exists() and (d / "rig.json").exists())
    if not dirs:
        raise SystemExit(f"no samples with scene.json/rig.json under {args.data!r}")

    todo = [d for d in dirs if not (d / pset / args.frame).exists()]
    print(f"{len(dirs)} samples, {len(todo)} need a {pset} capture", flush=True)
    for i, d in enumerate(todo):
        render_sample(d / "scene.json", d / "rig.json", [args.patterns], str(d), lean=True)
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)}  ({d.name})", flush=True)
    print(f"done -> {args.data}/*/{pset}/{args.frame}")


if __name__ == "__main__":
    main()
