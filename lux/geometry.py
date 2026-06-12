"""Camera / projector geometry for a structured-light rig.

The convention throughout lux:

  * The **camera** sits at the world origin looking down +Z.
  * The **projector** is posed relative to the camera by (R, t): a point
    expressed in camera coordinates ``X_cam`` maps to projector coordinates
    via ``X_proj = R @ X_cam + t``.
  * Both devices are pinhole, described by a 3x3 intrinsic matrix ``K``.

A structured-light decoder recovers, per camera pixel, *which projector column
(or row) illuminated it*. Given that correspondence plus the calibrated rig,
depth is a closed-form ray/plane intersection — implemented here in
:func:`triangulate_columns` so every column-decoding method shares the same
triangulation and the comparison stays apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a camera or projector."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Build intrinsics from a horizontal field of view, centered principal point."""
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(width=width, height=height, fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Rig:
    """A calibrated camera + projector pair.

    ``R`` (3x3) and ``t`` (3,) transform camera-frame points into the
    projector frame: ``X_proj = R @ X_cam + t``.
    """

    camera: Intrinsics
    projector: Intrinsics
    R: np.ndarray
    t: np.ndarray

    @classmethod
    def make(
        cls,
        camera: Intrinsics,
        projector: Intrinsics,
        baseline: float = 0.15,
        toe_in_deg: float = 0.0,
    ) -> "Rig":
        """Construct a rig with the projector translated ``baseline`` metres along
        camera +X (a sideways stereo offset), optionally toed-in toward the scene.

        ``baseline`` is the magnitude of the projector-to-camera offset; a larger
        baseline yields more disparity and better depth precision but more shadowing.
        """
        ang = np.radians(toe_in_deg)
        # Rotate the projector about its Y axis so it points slightly inward.
        R = np.array(
            [[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]],
            dtype=np.float64,
        )
        # Camera origin sits at -baseline in the projector frame.
        t = R @ np.array([-baseline, 0.0, 0.0], dtype=np.float64)
        return cls(camera=camera, projector=projector, R=R, t=t)


def camera_rays(intr: Intrinsics) -> np.ndarray:
    """Per-pixel viewing ray directions in the camera frame, shape (H, W, 3).

    Rays are *not* normalized; their Z component is 1, so ``depth * ray`` gives
    the 3D point and ``depth`` is the standard Z-depth used everywhere in lux.
    """
    us, vs = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    x = (us - intr.cx) / intr.fx
    y = (vs - intr.cy) / intr.fy
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def project(intr: Intrinsics, pts_cam: np.ndarray) -> np.ndarray:
    """Project camera-frame points (..., 3) to pixel coordinates (..., 2).

    Points behind the image plane (Z <= 0) yield NaNs.
    """
    z = pts_cam[..., 2]
    safe = z > 1e-9
    u = np.where(safe, intr.fx * pts_cam[..., 0] / z + intr.cx, np.nan)
    v = np.where(safe, intr.fy * pts_cam[..., 1] / z + intr.cy, np.nan)
    return np.stack([u, v], axis=-1)


def triangulate_columns(rig: Rig, proj_cols: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Recover metric depth from a decoded projector-column map.

    For each camera pixel we know the projector *column* ``u_p`` that lit it.
    A projector column defines a plane through the projector centre (the set of
    points whose normalized projector x equals ``(u_p - cx_p) / fx_p``). We
    intersect that plane with the camera viewing ray to get the 3D point, and
    return its Z-depth.

    Parameters
    ----------
    proj_cols : (H, W) float
        Decoded projector column index per camera pixel (sub-pixel allowed).
    valid : (H, W) bool, optional
        Pixels to triangulate; others are returned as NaN.

    Returns
    -------
    depth : (H, W) float
        Metric Z-depth in camera frame, NaN where invalid or degenerate.
    """
    cam = rig.camera
    proj = rig.projector
    rays = camera_rays(cam)  # (H, W, 3), Z == 1

    if valid is None:
        valid = np.isfinite(proj_cols)
    valid = valid & np.isfinite(proj_cols)

    # Plane for column u_p in the projector frame: points where X/Z == x_p,
    # i.e. normal n_p = (1, 0, -x_p) through the projector origin (offset 0).
    # Zero-fill invalid columns so NaNs don't propagate through the matmuls;
    # the `valid` mask removes them from the result below.
    x_p = np.where(valid, (proj_cols - proj.cx) / proj.fx, 0.0)  # (H, W)
    n_p = np.stack([np.ones_like(x_p), np.zeros_like(x_p), -x_p], axis=-1)  # (H, W, 3)

    # Express that plane in the camera frame. With X_proj = R X_cam + t,
    #   n_p . X_proj = 0  ->  (R^T n_p) . X_cam + (n_p . t) = 0.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        n_cam = n_p @ rig.R  # == (R^T n_p) per pixel, since (n_p @ R) = R^T n_p
        offset = n_p @ rig.t  # (H, W)
        # Camera ray X_cam = s * ray (through camera origin). Solve for s:
        #   n_cam . (s ray) + offset = 0  ->  s = -offset / (n_cam . ray)
        denom = np.einsum("hwc,hwc->hw", n_cam, rays)
        s = -offset / denom

    depth = s  # ray Z == 1, so the 3D point is s*ray and its depth is s
    bad = (~valid) | (np.abs(denom) < 1e-9) | ~np.isfinite(depth) | (depth <= 0)
    depth = np.where(bad, np.nan, depth)
    return depth
