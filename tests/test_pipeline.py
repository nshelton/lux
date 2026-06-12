"""Smoke + correctness tests for the lux pipeline.

Run with:  python -m pytest -q   (or just: python tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lux.geometry import Intrinsics, Rig, camera_rays, triangulate_columns
from lux.methods import build_method
from lux.render import RenderConfig, render
from lux.scene import build_scene


def _rig(w=160, h=120):
    cam = Intrinsics.from_fov(w, h, 50.0)
    proj = Intrinsics.from_fov(w, h, 45.0)
    return Rig.make(cam, proj, baseline=0.12)


def test_triangulation_roundtrip():
    """GT projector columns must triangulate back to the GT depth."""
    rig = _rig()
    cam = rig.camera
    scene = build_scene("slanted_plane", cam)
    # Compute exact GT projector columns from GT depth.
    rays = camera_rays(cam)
    pts_cam = scene.depth[..., None] * rays
    pts_proj = pts_cam @ rig.R.T + rig.t
    u_p = rig.projector.fx * pts_proj[..., 0] / pts_proj[..., 2] + rig.projector.cx
    depth = triangulate_columns(rig, u_p)
    valid = np.isfinite(depth)
    err = np.abs(depth[valid] - scene.depth[valid])
    assert np.nanmedian(err) < 1e-6, np.nanmedian(err)


def test_graycode_recovers_depth():
    rig = _rig()
    scene = build_scene("spheres_on_plane", rig.camera)
    method = build_method("graycode")
    cap = render(scene, method.patterns(rig.projector.width, rig.projector.height), rig,
                 RenderConfig(read_noise=0.005, shot_noise=0.0))
    res = method.decode(cap.images, rig)
    mask = scene.mask & cap.lit_mask & np.isfinite(res.proj_col)
    # Gray code is integer-precise: the decoded column must be exactly correct
    # (within rounding). Its depth error is then just column quantization.
    col_err = np.abs(res.proj_col[mask] - cap.gt_proj_col[mask])
    assert mask.sum() > 1000
    assert np.percentile(col_err, 99) < 0.5, np.percentile(col_err, 99)


def test_phaseshift_recovers_depth():
    rig = _rig()
    scene = build_scene("slanted_plane", rig.camera)
    method = build_method("phaseshift")
    cap = render(scene, method.patterns(rig.projector.width, rig.projector.height), rig,
                 RenderConfig(read_noise=0.005))
    res = method.decode(cap.images, rig)
    mask = scene.mask & cap.lit_mask & np.isfinite(res.depth)
    err_mm = np.abs(res.depth[mask] - scene.depth[mask]) * 1000
    assert np.median(err_mm) < 5.0, np.median(err_mm)


if __name__ == "__main__":
    test_triangulation_roundtrip()
    test_graycode_recovers_depth()
    test_phaseshift_recovers_depth()
    print("all tests passed")
