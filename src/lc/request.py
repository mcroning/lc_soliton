"""Human-facing request and compact result objects for the clean LC package.

Request objects contain user choices only. They contain no arrays and do not
execute algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .physics.liquid_crystal import LCSpec
from .physics.beam import BeamExperiment, single_gaussian
from .numerics.backend import BackendSpec
from .numerics.grid import GridSpec, TimeSpec


TridiagMode = Literal["reference", "fast"]


@dataclass(frozen=True)
class BaseRequest:
    """Common request fields."""

    lc: LCSpec = LCSpec()
    beams: BeamExperiment = single_gaussian()
    backend: BackendSpec = BackendSpec()
    grid: GridSpec = GridSpec()
    tridiag: TridiagMode = "fast"

    def validate_common(self) -> None:
        self.lc.validate()
        self.beams.validate()
        self.backend.validate()
        self.grid.validate()
        if self.tridiag not in ("reference", "fast"):
            raise ValueError("tridiag must be 'reference' or 'fast'")


@dataclass(frozen=True)
class TDRequest(BaseRequest):
    """Request for a time-dependent LC propagation run."""

    time: TimeSpec = TimeSpec()

    def validate(self) -> None:
        self.validate_common()
        self.time.validate()


@dataclass(frozen=True)
class StaticRequest(BaseRequest):
    """Request for a static/self-consistent LC solve.

    v001 only defines the public request shape. The static workflow will be
    added next.
    """

    max_outer: int = 20
    tol_rms: float = 5e-3
    tol_max: float = 2e-2

    def validate(self) -> None:
        self.validate_common()
        if self.max_outer < 0:
            raise ValueError("max_outer must be nonnegative")
        if self.tol_rms <= 0.0:
            raise ValueError("tol_rms must be positive")
        if self.tol_max <= 0.0:
            raise ValueError("tol_max must be positive")


@dataclass(frozen=True)
class RunSummary:
    """Small result summary returned by the first run() API."""

    kind: str
    metrics: dict
    samples: list[dict]

    def show(self) -> None:
        print(self.kind)
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.12g}")
            else:
                print(f"  {k:24s}: {v}")


__all__ = [
    "TridiagMode",
    "BaseRequest",
    "TDRequest",
    "StaticRequest",
    "RunSummary",
]
