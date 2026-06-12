"""Artifact export — depth maps, error visualizations, and point clouds.

Everything the web viewer (or Rerun, or any other tool) needs is written to
disk in plain formats: ``.npy`` for arrays, ``.ply`` for coloured point clouds,
``.png`` for quick-look images, and ``.json`` for scores. No viewer-specific
coupling — the harness produces data, viewers consume it.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .geometry import Rig, camera_rays


def save_image(path: str, img: np.ndarray) -> None:
    """Write a float image in [0, 1] (grayscale or RGB) to an 8-bit PNG."""
    arr = np.clip(np.nan_to_num(img), 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim == 3:  # RGB -> BGR for OpenCV
        u8 = u8[..., ::-1]
    cv2.imwrite(path, u8)


def proj_to_rgb(gt_proj: np.ndarray, proj_width: int, proj_height: int) -> np.ndarray:
    """Encode a projector-subpixel map ``(H, W, 2)`` as a quick-look RGB image.

    R = row / projector height, G = column / projector width, B = valid (1 where the
    pixel has a correspondence, else 0). Invalid pixels are black.
    """
    valid = np.isfinite(gt_proj[..., 0]) & np.isfinite(gt_proj[..., 1])
    rgb = np.zeros(gt_proj.shape[:2] + (3,), dtype=np.float64)
    rgb[..., 0] = np.nan_to_num(gt_proj[..., 1]) / proj_height   # row  -> red
    rgb[..., 1] = np.nan_to_num(gt_proj[..., 0]) / proj_width    # col  -> green
    rgb[..., 2] = valid.astype(np.float64)                       # valid -> blue
    return rgb


def load_image(path: str, gray: bool = True) -> np.ndarray:
    """Read a PNG/JPG into a float image in [0, 1]; grayscale (H, W) by default."""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    arr = img.astype(np.float64) / 255.0
    if not gray:  # BGR -> RGB
        arr = arr[..., ::-1]
    return arr


def save_image_stack(dir_path: str, stack: np.ndarray, prefix: str = "frame") -> None:
    """Write each (H, W) frame of an (N, H, W) stack as a zero-padded PNG."""
    ensure_dir(dir_path)
    for i, frame in enumerate(stack):
        save_image(os.path.join(dir_path, f"{prefix}_{i:02d}.png"), frame)


def montage(stack: np.ndarray, cols: int = 6, pad: int = 2, bg: float = 0.15) -> np.ndarray:
    """Tile an (N, H, W) or (N, H, W, 3) stack into a single contact-sheet image."""
    n, h, w = stack.shape[:3]
    color = stack.ndim == 4
    rows = int(np.ceil(n / cols))
    shape = (rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad)
    out = np.full(shape + ((3,) if color else ()), bg, dtype=np.float64)
    for i in range(n):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        out[y:y + h, x:x + w] = np.clip(np.nan_to_num(stack[i]), 0, 1)
    return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_npy(path: str, arr: np.ndarray) -> None:
    np.save(path, arr)


def depth_to_points(depth: np.ndarray, rig: Rig, color: np.ndarray | None = None):
    """Back-project a depth map to an (M, 3) point cloud + (M, 3) uint8 colours."""
    rays = camera_rays(rig.camera)
    valid = np.isfinite(depth) & (depth > 0)
    pts = (depth[..., None] * rays)[valid]
    if color is None:
        c = np.full((valid.sum(), 3), 200, dtype=np.uint8)
    else:
        col = color
        if col.ndim == 2:
            col = np.repeat(col[..., None], 3, axis=-1)
        c = np.clip(col[valid] * 255.0, 0, 255).astype(np.uint8)
    return pts.astype(np.float32), c


def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a binary-free ASCII PLY point cloud (small clouds; easy to load in JS)."""
    n = len(points)
    has_color = colors is not None
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    lines = ["\n".join(header)]
    if has_color:
        for (x, y, z), (r, g, b) in zip(points, colors):
            lines.append(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}")
    else:
        for x, y, z in points:
            lines.append(f"{x:.5f} {y:.5f} {z:.5f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def error_colormap(pred: np.ndarray, gt: np.ndarray, max_mm: float = 5.0) -> np.ndarray:
    """Map signed depth error to an RGB image (blue=near, red=far, grey=invalid)."""
    err = (pred - gt) * 1000.0
    out = np.full(pred.shape + (3,), 0.4, dtype=np.float64)
    valid = np.isfinite(err)
    t = np.clip((err[valid] / max_mm + 1) / 2, 0, 1)  # 0..1
    rgb = np.stack([t, 0.2 + 0.0 * t, 1 - t], axis=-1)
    out[valid] = rgb
    return out


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
