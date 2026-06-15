"""Phase-shifting profilometry (sinusoidal fringe patterns).

Projects ``K`` sinusoidal fringe patterns, each phase-shifted by ``2*pi*k/K``.
The per-pixel phase recovered from these is sub-pixel accurate but *wrapped*
to ``[0, 2*pi)`` — it only locates a column within one fringe period. To
unwrap, we add a second, lower frequency (a temporal/multi-frequency scheme):
the unit-frequency phase gives an absolute, unambiguous estimate that selects
the correct period of the high-frequency phase.

Result: a continuous, sub-pixel projector-column map — typically the most
accurate of the classical methods on smooth surfaces, at the cost of struggling
across depth discontinuities where phase unwrapping breaks.
"""

from __future__ import annotations

import numpy as np

from ..geometry import Rig
from .base import DepthResult, Method


def _shifted_sines(width: int, height: int, periods: float, shifts: int) -> list[np.ndarray]:
    x = np.arange(width)
    out = []
    for k in range(shifts):
        phase = 2 * np.pi * periods * x / width + 2 * np.pi * k / shifts
        row = 0.5 + 0.5 * np.cos(phase)
        out.append(np.broadcast_to(row, (height, width)).astype(np.float64))
    return out


def _demodulate(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N-step phase: returns wrapped phase in [0, 2pi) and modulation amplitude."""
    K = images.shape[0]
    k = np.arange(K)
    sin_w = np.sin(2 * np.pi * k / K)[:, None, None]
    cos_w = np.cos(2 * np.pi * k / K)[:, None, None]
    num = -(images * sin_w).sum(axis=0)
    den = (images * cos_w).sum(axis=0)
    phase = np.mod(np.arctan2(num, den), 2 * np.pi)
    amplitude = (2.0 / K) * np.sqrt(num**2 + den**2)
    return phase, amplitude


class PhaseShiftMethod(Method):
    name = "phaseshift"

    def __init__(self, shifts: int = 4, high_periods: int = 16):
        self.shifts = shifts
        self.high_periods = high_periods

    def patterns(self, width: int, height: int) -> np.ndarray:
        stack = []
        # High-frequency set (precision) + unit-frequency set (unwrapping).
        stack += _shifted_sines(width, height, self.high_periods, self.shifts)
        stack += _shifted_sines(width, height, 1, self.shifts)
        return np.stack(stack, axis=0)

    def fringe_phase(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """High-frequency wrapped phase [0, 2pi) + modulation amplitude — the
        sub-pixel within-fringe position. Pair with an external integer fringe
        index (e.g. a Gray-code column) for robust *Gray-coded* phase shifting,
        which avoids this method's own fragile unit-frequency unwrap."""
        return _demodulate(images[: self.shifts])

    def decode_columns(self, images: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        """Sub-pixel projector column per pixel via dual-frequency phase shift,
        *without* a rig — returns (proj_col, amplitude), NaN where unlit/out of
        range. ``decode`` builds on this."""
        hi = images[: self.shifts]
        lo = images[self.shifts : 2 * self.shifts]

        phase_hi, amp_hi = _demodulate(hi)
        phase_lo, amp_lo = _demodulate(lo)

        # Absolute phase from the unit-frequency set, in [0, 2pi) over the width.
        abs_phase = phase_lo
        # Which high-frequency period does each pixel belong to?
        period = np.round((self.high_periods * abs_phase - phase_hi) / (2 * np.pi))
        unwrapped = phase_hi + 2 * np.pi * period  # in [0, 2pi*high_periods)

        proj_col = unwrapped / (2 * np.pi * self.high_periods) * width

        amplitude = np.minimum(amp_hi, amp_lo)
        lit = (amplitude > 0.05) & np.isfinite(proj_col) & (proj_col >= 0) & (proj_col < width)
        return np.where(lit, proj_col, np.nan), np.clip(amplitude, 0, 1)

    def decode(self, images: np.ndarray, rig: Rig) -> DepthResult:
        proj_col, amplitude = self.decode_columns(images, rig.projector.width)
        depth = self.triangulate(rig, proj_col, valid=np.isfinite(proj_col))
        return DepthResult(depth=depth, proj_col=proj_col, confidence=amplitude)
