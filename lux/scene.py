"""Synthetic scenes: ground-truth depth + albedo, viewed by the rig's camera.

A :class:`Scene` is intentionally minimal — it is just what the camera sees:
a per-pixel metric depth map (the ground truth we score against) and a per-pixel
albedo (reflectance) that modulates how bright the projected pattern appears.

Scenes are defined directly in the camera image plane via :func:`camera_rays`,
so there is no separate world mesh to manage. Generators below build a few
canonical test surfaces; add your own by returning a (depth, albedo) pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Intrinsics, camera_rays


@dataclass
class Scene:
    name: str
    depth: np.ndarray   # (H, W) metric Z-depth, NaN = no surface (background)
    albedo: np.ndarray  # (H, W) in [0, 1]

    @property
    def mask(self) -> np.ndarray:
        """Pixels that actually contain surface (finite, positive depth)."""
        return np.isfinite(self.depth) & (self.depth > 0)


def _grid(cam: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    rays = camera_rays(cam)
    return rays[..., 0], rays[..., 1]  # normalized x, y (per unit depth)


def slanted_plane(cam: Intrinsics, z0: float = 1.0, slant_x: float = 0.3, slant_y: float = 0.1) -> Scene:
    """A tilted plane filling the frame — the simplest non-trivial depth field."""
    x, y = _grid(cam)
    # Plane: depth varies linearly with the normalized ray coords.
    depth = z0 / (1.0 - slant_x * x - slant_y * y)
    albedo = np.full((cam.height, cam.width), 0.8)
    return Scene("slanted_plane", depth.astype(np.float64), albedo.astype(np.float64))


def spheres_on_plane(cam: Intrinsics, z0: float = 1.4, n: int = 4, seed: int = 0) -> Scene:
    """A back plane with a few spheres — depth discontinuities + curvature."""
    x, y = _grid(cam)
    depth = np.full((cam.height, cam.width), z0, dtype=np.float64)
    albedo = np.full((cam.height, cam.width), 0.65, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for _ in range(n):
        # Sphere centre in the camera frame, in front of the plane.
        cz = rng.uniform(0.8 * z0, 0.95 * z0)
        cx = rng.uniform(-0.4, 0.4) * cz
        cy = rng.uniform(-0.3, 0.3) * cz
        r = rng.uniform(0.12, 0.22) * cz
        alb = rng.uniform(0.4, 0.95)

        # Ray-sphere intersection: point = d * (x, y, 1). Solve for nearest d.
        dx, dy = x, y
        a = dx * dx + dy * dy + 1.0
        b = -2.0 * (dx * cx + dy * cy + cz)
        c = cx * cx + cy * cy + cz * cz - r * r
        disc = b * b - 4 * a * c
        hit = disc > 0
        d = np.where(hit, (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2 * a), np.inf)
        closer = hit & (d > 0) & (d < depth)
        depth = np.where(closer, d, depth)
        albedo = np.where(closer, alb, albedo)

    return Scene("spheres_on_plane", depth, albedo)


def depth_ramp_steps(cam: Intrinsics, z_near: float = 0.8, z_far: float = 1.8, steps: int = 5) -> Scene:
    """Fronto-parallel staircase — clean depth edges to stress decoding at boundaries."""
    H, W = cam.height, cam.width
    cols = (np.arange(W) / W * steps).astype(int)
    levels = np.linspace(z_near, z_far, steps)
    depth = np.broadcast_to(levels[np.clip(cols, 0, steps - 1)], (H, W)).astype(np.float64).copy()
    # Alternating albedo bands so texture doesn't perfectly align with depth steps.
    albedo = np.broadcast_to(np.where((np.arange(W) // 17) % 2 == 0, 0.85, 0.5), (H, W)).astype(np.float64).copy()
    return Scene("depth_ramp_steps", depth, albedo)


# Registry of zero-arg-ish scene builders, parameterized only by camera intrinsics.
SCENES = {
    "slanted_plane": slanted_plane,
    "spheres_on_plane": spheres_on_plane,
    "depth_ramp_steps": depth_ramp_steps,
}


def build_scene(name: str, cam: Intrinsics) -> Scene:
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r}; available: {sorted(SCENES)}")
    return SCENES[name](cam)
