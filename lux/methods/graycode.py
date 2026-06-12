"""Classic binary Gray-code structured light.

Projects a stack of vertical stripe patterns whose widths halve each frame,
encoded in reflected-binary (Gray) code so adjacent columns differ by one bit —
this bounds decoding error at bit boundaries. Each pattern is shown alongside
its inverse so a per-pixel threshold (rather than a global one) decides each
bit, which is robust to albedo variation.

Decoding yields an **integer** projector column per pixel; depth comes from the
shared ray/plane triangulation.

Known, correct behavior — depth terracing ("sawtooth")
------------------------------------------------------
Because the decoded column is an integer, every camera pixel that maps to the
same projector column triangulates to the *same* depth plane. When the camera
over-samples the projector (a single projector column covers N camera pixels —
e.g. a high-res camera against a lower-res projector), a block of N camera
pixels shares one depth, while the true surface ramps smoothly across it. The
result is a depth error that ramps then resets with **period N pixels** (≈ N/2
over-estimating, ≈ N/2 under-estimating) — i.e. a sawtooth. The period equals
camera-pixels-per-projector-column.

This is intrinsic to *binary* Gray code, not a decoding error: feeding the
continuous (sub-pixel) ground-truth column into the same triangulation gives
~0 error. It is exactly why phase-shift (sub-pixel by construction) beats Gray
code on smooth surfaces in the benchmark. For sub-pixel Gray code, layer a
phase-shift sequence on top (Gray-coded phase shifting); the Gray bits then
only resolve the integer fringe index. Kept here as the honest integer baseline.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Rig
from .base import DepthResult, Method


def _num_bits(width: int) -> int:
    return int(np.ceil(np.log2(width)))


def _gray_to_binary(g: np.ndarray) -> np.ndarray:
    """Vectorized reflected-binary-Gray to natural-binary over the bit axis (last)."""
    b = g.copy()
    for i in range(1, b.shape[-1]):
        b[..., i] = np.bitwise_xor(b[..., i], b[..., i - 1])
    return b


class GrayCodeMethod(Method):
    name = "graycode"

    def __init__(self, invert_pairs: bool = True):
        self.invert_pairs = invert_pairs  # project each pattern + its inverse

    def patterns(self, width: int, height: int) -> np.ndarray:
        nbits = _num_bits(width)
        cols = np.arange(width)
        # Natural binary -> Gray code.
        gray = cols ^ (cols >> 1)
        stack = []
        for bit in range(nbits - 1, -1, -1):  # MSB first
            row = ((gray >> bit) & 1).astype(np.float64)
            pat = np.broadcast_to(row, (height, width)).astype(np.float64)
            stack.append(pat)
            if self.invert_pairs:
                stack.append(1.0 - pat)
        # Plus all-white / all-black for shadow + albedo normalization.
        stack.append(np.ones((height, width)))
        stack.append(np.zeros((height, width)))
        return np.stack(stack, axis=0)

    def decode(self, images: np.ndarray, rig: Rig) -> DepthResult:
        width = rig.projector.width
        nbits = _num_bits(width)
        H, W = images.shape[1:]

        white = images[-2]
        black = images[-1]
        amplitude = white - black
        # Pixels with too little projector modulation are unreliable.
        lit = amplitude > 0.08

        gray_bits = np.zeros((H, W, nbits), dtype=np.int64)
        confidence = np.ones((H, W), dtype=np.float64)
        idx = 0
        for bit in range(nbits):
            if self.invert_pairs:
                pos = images[idx]
                neg = images[idx + 1]
                idx += 2
                gray_bits[..., bit] = (pos > neg).astype(np.int64)
                # Confidence = how cleanly the bit separates, normalized by amplitude.
                sep = np.abs(pos - neg) / np.maximum(amplitude, 1e-6)
                confidence = np.minimum(confidence, np.clip(sep, 0, 1))
            else:
                pat = images[idx]
                idx += 1
                mid = black + 0.5 * amplitude
                gray_bits[..., bit] = (pat > mid).astype(np.int64)

        binary = _gray_to_binary(gray_bits)  # MSB at index 0
        weights = (1 << np.arange(nbits - 1, -1, -1)).astype(np.int64)
        proj_col = (binary * weights).sum(axis=-1).astype(np.float64)
        proj_col = np.where(lit & (proj_col < width), proj_col, np.nan)

        depth = self.triangulate(rig, proj_col, valid=np.isfinite(proj_col))
        return DepthResult(depth=depth, proj_col=proj_col, confidence=np.where(lit, confidence, 0.0))
