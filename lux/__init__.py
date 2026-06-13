"""lux — a structured-light algorithm testbed.

Pipeline:  scene (GT depth + albedo)  ->  render(patterns)  ->  captured images
           ->  method.decode  ->  metric depth  ->  compare vs GT  ->  scores.

Key modules:
  * :mod:`lux.geometry`  — camera/projector rig + triangulation
  * :mod:`lux.scene`     — synthetic ground-truth scenes
  * :mod:`lux.render`    — the analytic forward simulator
  * :mod:`lux.methods`   — pluggable algorithms (graycode, phaseshift, neural)
  * :mod:`lux.metrics`   — depth comparison
  * :mod:`lux.harness`   — orchestration + reporting
"""

from .geometry import Intrinsics, Rig
from .harness import HarnessConfig, run, format_table
from .metrics import compare_depth
from .methods import REGISTRY, build_method
from .scene import SCENES, build_scene

__all__ = [
    "Intrinsics", "Rig", "HarnessConfig", "run", "format_table",
    "compare_depth", "REGISTRY", "build_method", "SCENES", "build_scene",
]
