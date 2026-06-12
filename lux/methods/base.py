"""The pluggable-method interface — the core extension point of the testbed.

A *method* is one structured-light algorithm. It declares the projector
patterns it needs and knows how to turn the resulting camera images back into a
metric depth map. The harness renders the requested patterns through the
simulator and hands the images back for decoding, so a method fully owns the
"images in -> depth out" contract.

Two styles are supported:

  * **Column decoders** (Gray code, phase shift) recover a projector-column map
    and call :func:`lux.geometry.triangulate_columns` for depth. The base class
    offers :meth:`triangulate` as a convenience.
  * **Direct regressors** (a neural net) may skip correspondence and return
    depth straight away.

Either way, :meth:`decode` returns a :class:`DepthResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Rig, triangulate_columns


@dataclass
class DepthResult:
    depth: np.ndarray              # (H, W) metric depth, NaN where the method abstains
    proj_col: np.ndarray | None = None   # (H, W) decoded projector column, if applicable
    confidence: np.ndarray | None = None # (H, W) in [0, 1], optional


class Method:
    """Subclass and implement :meth:`patterns` and :meth:`decode`."""

    name: str = "base"
    # If True, the method needs the GT rig at decode time to triangulate.
    # (All methods receive it; learned methods may simply ignore it.)

    def patterns(self, width: int, height: int) -> np.ndarray:
        """Return the (N, height, width) projector pattern stack to display."""
        raise NotImplementedError

    def decode(self, images: np.ndarray, rig: Rig) -> DepthResult:
        """Turn the (N, H, W) camera image stack into a :class:`DepthResult`."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def triangulate(rig: Rig, proj_col: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        return triangulate_columns(rig, proj_col, valid)
