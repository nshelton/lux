#!/usr/bin/env python3
"""Verify gt_proj is the exact projector subpixel a decoder recovers from renders.

We render a Gray-code sequence through the projector and decode it to a projector
column per camera pixel (Gray code is robust to global illumination, unlike a raw
intensity ramp). That decoded column is compared against ``projector_subpixel``:

  * **No-distortion control** -- decoded column should match gt_proj to ~a pixel,
    confirming the projector emitter and lux's ``project()`` share a convention.
  * **Projector distortion** -- decoded column should match the *corrected*
    gt_proj (``proj_optics`` passed), not the uncorrected (ideal-pinhole) one.

Camera is kept ideal (no distortion/DoF) so captures and GT share one image space.
Works for either renderer (the raster backend is exact by construction, so the
control should be at the Gray-code quantisation floor):

    python scripts/verify_gt_proj.py --backend mitsuba
    python scripts/verify_gt_proj.py --backend raster
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from lux.geometry import Intrinsics, Rig  # noqa: E402
from lux.datasets.correspondence import projector_subpixel  # noqa: E402
from lux.datasets.optics import RigOptics, DeviceOptics  # noqa: E402
from lux.methods import build_method  # noqa: E402

_BACKENDS = {"mitsuba": "lux.datasets.mitsuba_gen", "raster": "lux.datasets.raster_gen"}


def _make_rig(width=320, height=240, proj_w=256, proj_h=256, cam_hfov=45.0,
              proj_hfov=40.0, baseline=0.18, standoff=1.4):
    """Baseline rig stood back ``standoff`` m so it views a scene at the origin."""
    from lux.geometry import look_at_basis
    cam = Intrinsics.from_fov(width, height, cam_hfov)
    proj = Intrinsics.from_fov(proj_w, proj_h, proj_hfov)
    fwd, up = [0, 0, 1], [0, -1, 0]
    Rc = Rp = look_at_basis(fwd, up)
    return Rig.from_poses(cam, proj, Rc, np.array([0.0, 0.0, -standoff]),
                          Rp, np.array([baseline, 0.0, -standoff]))


def _decode_columns(backend, rig, geo, optics):
    method = build_method("graycode")
    pats = method.patterns(rig.projector.width, rig.projector.height)
    stack = np.stack([backend.render_capture(rig, p, geometry=geo, optics=optics) for p in pats])
    return method.decode(stack, rig).proj_col


def _compare(name, decoded, gt):
    m = np.isfinite(decoded) & np.isfinite(gt)
    e = np.abs(decoded[m] - gt[m])
    med, p95 = float(np.median(e)), float(np.percentile(e, 95))
    print(f"  {name:32s} median={med:6.2f}px  p95={p95:6.2f}px  ({m.sum()} px)")
    return med, p95


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(_BACKENDS), default="mitsuba")
    args = ap.parse_args()
    backend = importlib.import_module(_BACKENDS[args.backend])
    print(f"backend: {args.backend}")

    rig = _make_rig()
    geo = backend.load_geometry("wavy")
    depth, _ = backend.render_ground_truth(rig, geometry=geo)
    pdist = DeviceOptics(dist=(-0.30, 0.10, 0.0, 0.0))

    gt_ideal = projector_subpixel(rig, depth)[..., 0]
    gt_corr = projector_subpixel(rig, depth, proj_optics=pdist)[..., 0]

    print("[control] no distortion -- decoded Gray code vs gt_proj:")
    dec0 = _decode_columns(backend, rig, geo, RigOptics())
    ctrl_med, _ = _compare("decoded vs gt_proj", dec0, gt_ideal)

    print("\n[distortion] projector distortion -- decoded Gray code vs gt_proj:")
    decd = _decode_columns(backend, rig, geo, RigOptics(camera=DeviceOptics(), projector=pdist))
    _, unc_p95 = _compare("decoded vs uncorrected gt_proj", decd, gt_ideal)
    cor_med, cor_p95 = _compare("decoded vs corrected gt_proj", decd, gt_corr)

    # Median is floored by Gray-code integer quantisation (~0.25px), so judge the
    # distortion correction by the edge tail (p95) where distortion actually bites.
    ok = ctrl_med < 1.0 and cor_med < 1.0 and unc_p95 > 3 * cor_p95
    print(f"\n{'PASS' if ok else 'FAIL'}: convention sound (control {ctrl_med:.2f}px median); correction "
          f"fixes the distorted edge tail (p95 {unc_p95:.2f}px -> {cor_p95:.2f}px).")


if __name__ == "__main__":
    main()
