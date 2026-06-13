#!/usr/bin/env python3
"""Compare the ray-caster backend against Mitsuba on the same scenes.

Both backends derive the ground truth from the same geometry + calibration, so
their ``gt_depth`` and ``gt_proj`` should agree to sub-pixel / sub-mm on smooth
interiors; disagreement is expected only at silhouettes (1 sample/pixel ray-caster
vs Mitsuba's anti-aliasing) and wavy folds. Captures differ radiometrically
(analytic Lambertian vs path-traced BSDF) so we compare the GT, not pixel values.

    python scripts/compare_backends.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux.geometry import Intrinsics, Rig  # noqa: E402
from lux.datasets.correspondence import projector_subpixel  # noqa: E402
from lux.datasets import mitsuba_gen, raster_gen  # noqa: E402


def _rig(width=480, height=360, standoff=1.4):
    """Baseline rig stood back ``standoff`` m so it views a scene at the origin."""
    from lux.geometry import look_at_basis
    cam = Intrinsics.from_fov(width, height, 45.0)
    proj = Intrinsics.from_fov(512, 512, 40.0)
    R = look_at_basis([0, 0, 1], [0, -1, 0])
    return Rig.from_poses(cam, proj, R, np.array([0.0, 0.0, -standoff]),
                          R, np.array([0.18, 0.0, -standoff]))


def _stats(a, b, mask, scale, unit):
    m = mask & np.isfinite(a) & np.isfinite(b)
    e = np.abs(a[m] - b[m]) * scale
    return f"median={np.median(e):.3f}{unit} p95={np.percentile(e, 95):.3f}{unit} ({m.sum()} px)"


def main() -> None:
    rig = _rig()
    for scene in ("blocks", "wavy"):
        gm = mitsuba_gen.load_geometry(scene)
        gr = raster_gen.load_geometry(scene)
        dm, _ = mitsuba_gen.render_ground_truth(rig, geometry=gm, spp=32)
        dr, _ = raster_gen.render_ground_truth(rig, geometry=gr)
        both = np.isfinite(dm) & np.isfinite(dr)

        pm = projector_subpixel(rig, dm)[..., 0]
        pr = projector_subpixel(rig, dr)[..., 0]

        cov_m, cov_r = np.isfinite(dm).mean(), np.isfinite(dr).mean()
        print(f"\n[{scene}]  coverage mitsuba={cov_m*100:.1f}%  raster={cov_r*100:.1f}%")
        print(f"  gt_depth raster vs mitsuba:  {_stats(dr, dm, both, 1000.0, 'mm')}")
        print(f"  gt_proj  raster vs mitsuba:  {_stats(pr, pm, both & np.isfinite(pm) & np.isfinite(pr), 1.0, 'px')}")


if __name__ == "__main__":
    main()
