"""Result objects for clean LC workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Array = Any


@dataclass
class BaseResult:
    """Common compact result interface."""

    kind: str
    metrics: dict
    samples: list[dict] = field(default_factory=list)

    def show(self) -> None:
        print(self.kind)
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.12g}")
            else:
                print(f"  {k:24s}: {v}")


@dataclass
class TDResult(BaseResult):
    """Time-dependent workflow result.

    Arrays are optional so workflows can remain lightweight by default.
    """

    theta: Array | None = None
    A_last: Array | None = None
    final_intensity: Array | None = None


@dataclass
class StaticResult(BaseResult):
    """Static/self-consistent workflow result."""

    theta: Array | None = None
    intensity: Array | None = None
    history: list[dict] = field(default_factory=list)
    converged: bool = False


__all__ = [
    "Array",
    "BaseResult",
    "TDResult",
    "StaticResult",
]
