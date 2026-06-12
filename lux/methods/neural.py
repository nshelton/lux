"""Learned structured-light decoding (scaffold).

This is the extension point for the neural-net approach. It uses a *small*
projected pattern stack (a handful of frequencies, like the classical methods)
and learns to regress a projector-column map per pixel from the captured
images. The triangulation step is shared with the classical methods, so the
network only has to solve the (hard) correspondence problem, not the geometry.

The model is deliberately tiny — a per-pixel MLP over the stack of intensities
plus a couple of fixed sinusoidal pattern channels — enough to train end to end
on synthetic data and prove the loop. Swap in a CNN/U-Net when you want spatial
context.

Without a trained checkpoint (or without torch installed) :meth:`decode` falls
back to a phase-shift decode so the harness always produces *a* depth map and
the comparison stays meaningful. Train with ``scripts/train_neural.py``.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Rig
from .base import DepthResult, Method
from .phaseshift import PhaseShiftMethod, _shifted_sines

try:  # torch is optional — the fallback path does not need it.
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# Fixed pattern design shared by training and inference.
_SHIFTS = 4
_PERIODS = 16


def make_patterns(width: int, height: int) -> np.ndarray:
    stack = []
    stack += _shifted_sines(width, height, _PERIODS, _SHIFTS)
    stack += _shifted_sines(width, height, 1, _SHIFTS)
    return np.stack(stack, axis=0)


if _HAS_TORCH:

    class PixelMLP(nn.Module):
        """Per-pixel MLP: maps the per-pixel intensity vector to a column fraction."""

        def __init__(self, n_channels: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):  # x: (..., n_channels) -> (..., 1) in [0, 1]
            return torch.sigmoid(self.net(x))


class NeuralMethod(Method):
    name = "neural"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        if _HAS_TORCH and checkpoint:
            self._load(checkpoint)

    def _load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        model = PixelMLP(ckpt["n_channels"], ckpt.get("hidden", 64)).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model

    def patterns(self, width: int, height: int) -> np.ndarray:
        return make_patterns(width, height)

    def decode(self, images: np.ndarray, rig: Rig) -> DepthResult:
        width = rig.projector.width
        if not _HAS_TORCH or self._model is None:
            # Graceful fallback: behave like phase-shift so the harness runs.
            res = PhaseShiftMethod(shifts=_SHIFTS, high_periods=_PERIODS).decode(images, rig)
            return DepthResult(depth=res.depth, proj_col=res.proj_col, confidence=res.confidence)

        x = torch.from_numpy(images.astype(np.float32)).permute(1, 2, 0).to(self.device)  # (H,W,C)
        with torch.no_grad():
            frac = self._model(x).squeeze(-1).cpu().numpy()  # (H, W) in [0, 1]
        proj_col = frac * (width - 1)

        # Use projector modulation to gate invalid pixels.
        amp = images.max(axis=0) - images.min(axis=0)
        lit = amp > 0.05
        proj_col = np.where(lit, proj_col, np.nan)
        depth = self.triangulate(rig, proj_col, valid=lit)
        return DepthResult(depth=depth, proj_col=proj_col, confidence=np.clip(amp, 0, 1))
