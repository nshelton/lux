#!/usr/bin/env python3
"""Run the structured-light benchmark over scenes x methods and print a table.

Examples
--------
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --methods graycode phaseshift --scenes spheres_on_plane
    python scripts/run_benchmark.py --noise 0.03 --shot 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lux.harness import HarnessConfig, format_table, run  # noqa: E402
from lux.methods import REGISTRY  # noqa: E402
from lux.render import RenderConfig  # noqa: E402
from lux.scene import SCENES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    ap.add_argument("--methods", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--baseline", type=float, default=0.12)
    ap.add_argument("--noise", type=float, default=0.01, help="read-noise sigma")
    ap.add_argument("--shot", type=float, default=0.0, help="shot-noise scale")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="skip saving captured/pattern PNGs")
    args = ap.parse_args()

    cfg = HarnessConfig(
        width=args.width,
        height=args.height,
        baseline=args.baseline,
        out_dir=args.out,
        save_artifacts=not args.no_artifacts,
        save_frames=not args.no_frames,
        render=RenderConfig(read_noise=args.noise, shot_noise=args.shot, cast_shadows=not args.no_shadows),
    )

    records = run(args.scenes, args.methods, cfg)
    print(format_table(records))
    if not args.no_artifacts:
        print(f"\nartifacts + scores.json written to ./{args.out}/")


if __name__ == "__main__":
    main()
