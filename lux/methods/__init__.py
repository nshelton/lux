"""Method registry — every structured-light algorithm registers here.

Add a new algorithm by subclassing :class:`lux.methods.base.Method` and adding
a factory to ``REGISTRY``. The harness and CLI discover methods by name.
"""

from __future__ import annotations

from typing import Callable

from .base import DepthResult, Method
from .graycode import GrayCodeMethod
from .phaseshift import PhaseShiftMethod
from .neural import NeuralMethod

# name -> zero-arg factory (configure variants by adding entries)
REGISTRY: dict[str, Callable[[], Method]] = {
    "graycode": GrayCodeMethod,
    "phaseshift": PhaseShiftMethod,
    "neural": NeuralMethod,
}


def build_method(name: str) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


__all__ = ["DepthResult", "Method", "REGISTRY", "build_method"]
