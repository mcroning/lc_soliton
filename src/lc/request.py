"""Human-facing request objects for the clean LC package."""

from __future__ import annotations

from dataclasses import dataclass

from .physics.liquid_crystal import LCSpec
from .physics.beam import BeamExperiment, single_gaussian
from .numerics.backend import BackendSpec
from .numerics.grid import GridSpec, TimeSpec


@dataclass(frozen=True)
class TDRequest:
    """Request for a time-dependent LC propagation run."""

    lc: LCSpec = LCSpec()
    beams: BeamExperiment = single_gaussian()
    backend: BackendSpec = BackendSpec()
    grid: GridSpec = GridSpec()
    time: TimeSpec = TimeSpec()
    tridiag: str = "fast"

    def validate(self) -> None:
        self.lc.validate()
        self.beams.validate()
        self.backend.validate()
        self.grid.validate()
        self.time.validate()
        if self.tridiag not in ("reference", "fast"):
            raise ValueError("tridiag must be 'reference' or 'fast'")


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


__all__ = ["TDRequest", "RunSummary"]
