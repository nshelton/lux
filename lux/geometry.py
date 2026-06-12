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


def look_at_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->device rotation for a device looking along ``forward`` in the world.

    Rows are the device axes expressed in world coordinates, in lux's image order:
    ``[+X right, +Y down, +Z forward]``, so ``X_device = R @ (X_world - centre)``.
    ``up`` is the world direction that should appear at the **top** of the image
    (image-up is ``-Y``, so the device's down axis is ``-up`` orthogonalised against
    forward). The default camera (forward ``+Z``, up ``(0,-1,0)``) gives identity.
    """
    z = np.asarray(forward, float)
    z = z / np.linalg.norm(z)
    u = np.asarray(up, float)
    y = (u @ z) * z - u                      # image-down ~ -up, orthogonal to forward
    if np.linalg.norm(y) < 1e-9:             # up parallel to forward -> pick any
        y = np.array([0.0, 1.0, 0.0]) - z * z[1]
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return np.array([x, y, z])


@dataclass(frozen=True)
class Rig:
    """A calibrated camera + projector pair.

    ``R`` (3x3) and ``t`` (3,) transform camera-frame points into the projector
    frame: ``X_proj = R @ X_cam + t`` — the relative transform every SL routine
    uses (triangulation, correspondence). The optional world poses ``(R_cam, C_cam)``
    and ``(R_proj, C_proj)`` place each device in a shared world frame (rows of
    ``R_*`` are the device axes in world; ``C_*`` is its centre); renderers use
    these to support arbitrary device placement. They default to a camera at the
    origin looking ``+Z`` (so world == camera frame), with the projector pose
    derived from ``R, t`` — making legacy ``baseline`` rigs behave exactly as before.
    """

    camera: Intrinsics
    projector: Intrinsics
    R: np.ndarray
    t: np.ndarray
    R_cam: np.ndarray = None
    C_cam: np.ndarray = None
    R_proj: np.ndarray = None
    C_proj: np.ndarray = None

    def __post_init__(self):
        if self.R_cam is None:
            object.__setattr__(self, "R_cam", np.eye(3))
        if self.C_cam is None:
            object.__setattr__(self, "C_cam", np.zeros(3))
        if self.R_proj is None:                         # projector world pose from R, t
            object.__setattr__(self, "R_proj", self.R @ self.R_cam)
        if self.C_proj is None:
            object.__setattr__(self, "C_proj",
                               self.C_cam - self.R_cam.T @ (self.R.T @ self.t))

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

    @classmethod
    def from_poses(cls, camera: Intrinsics, projector: Intrinsics,
                   R_cam: np.ndarray, C_cam: np.ndarray,
                   R_proj: np.ndarray, C_proj: np.ndarray) -> "Rig":
        """Construct a rig from explicit world poses for the camera and projector.

        Each pose is a world->device rotation (see :func:`look_at_basis`) and a
        world-space centre. The relative ``R, t`` are derived so all SL routines
        work unchanged: ``R = R_proj @ R_cam.T``, ``t = R_proj @ (C_cam - C_proj)``.
        """
        R_cam, C_cam = np.asarray(R_cam, float), np.asarray(C_cam, float)
        R_proj, C_proj = np.asarray(R_proj, float), np.asarray(C_proj, float)
        R = R_proj @ R_cam.T
        t = R_proj @ (C_cam - C_proj)
        return cls(camera=camera, projector=projector, R=R, t=t,
                   R_cam=R_cam, C_cam=C_cam, R_proj=R_proj, C_proj=C_proj)


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
