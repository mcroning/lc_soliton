"""Human-facing request objects for the clean LC package."""

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
    time: TimeSpec = TimeSpec()

    def validate(self) -> None:
        self.validate_common()
        self.time.validate()


@dataclass(frozen=True)
class StaticRequest(BaseRequest):
    """Request for a static/self-consistent LC solve.

    ``max_outer`` controls coupled optical/theta outer passes.

    ``static_max_steps`` is the strict per-slice theta relaxation depth, matching
    the old ``static_max_steps`` semantics. ``static_resid_every`` controls how
    often the inner residual is checked. ``static_relax_omega`` is the inner
    under-relaxation factor.

    ``static_inner_steps`` is retained only for the earlier simple static
    workflow; strict z-stack workflows should use ``static_max_steps``.
    """

    max_outer: int = 20
    static_inner_steps: int = 1
    static_max_steps: int = 200
    static_resid_every: int = 10
    static_relax_omega: float = 0.3

    tol_rms: float = 5e-3
    tol_max: float = 2e-2
    tol_residual_rms: float = 5e-3
    tol_residual_max: float = 2e-1

    def validate(self) -> None:
        self.validate_common()
        if self.max_outer < 0:
            raise ValueError("max_outer must be nonnegative")
        if self.static_inner_steps < 1:
            raise ValueError("static_inner_steps must be >= 1")
        if self.static_max_steps < 1:
            raise ValueError("static_max_steps must be >= 1")
        if self.static_resid_every < 1:
            raise ValueError("static_resid_every must be >= 1")
        if not (0.0 < self.static_relax_omega <= 1.0):
            raise ValueError("static_relax_omega must be in (0, 1]")
        if self.tol_rms <= 0.0:
            raise ValueError("tol_rms must be positive")
        if self.tol_max <= 0.0:
            raise ValueError("tol_max must be positive")
        if self.tol_residual_rms <= 0.0:
            raise ValueError("tol_residual_rms must be positive")
        if self.tol_residual_max <= 0.0:
            raise ValueError("tol_residual_max must be positive")


__all__ = ["TridiagMode", "BaseRequest", "TDRequest", "StaticRequest"]
