"""Existence-curve workflow using true continuation for soliton solver.

Place as: src/lc/workflows/existence_curve.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..request import StaticRequest
from ..result import BaseResult
from ..physics.beam import BeamChannel, BeamExperiment
from ..numerics.backend import BackendSpec
from .soliton import SolitonRequest, SolitonResult, run_soliton
from .static_zstack import run_static_zstack


ExistenceSolver = Literal["soliton", "static_zstack"]


@dataclass(frozen=True)
class ExistenceCurveRequest:
    base: StaticRequest
    powers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    continuation: bool = True
    solver: ExistenceSolver = "soliton"

    soliton_max_outer: int = 80
    theta_steps_per_outer: int = 50
    field_mix: float = 0.25
    theta_mix: float = 1.0
    tol_field: float = 1e-4
    tol_theta: float = 1e-5
    tol_residual_rms: float = 1e-3
    tol_residual_max: float = 1e-2

    def validate(self) -> None:
        self.base.validate()
        if self.solver not in ("soliton", "static_zstack"):
            raise ValueError("solver must be 'soliton' or 'static_zstack'")
        if len(self.powers) == 0:
            raise ValueError("powers must be nonempty")
        for p in self.powers:
            if p < 0.0:
                raise ValueError("powers must be nonnegative")
        if self.soliton_max_outer < 1:
            raise ValueError("soliton_max_outer must be >= 1")
        if self.theta_steps_per_outer < 1:
            raise ValueError("theta_steps_per_outer must be >= 1")
        if not (0.0 < self.field_mix <= 1.0):
            raise ValueError("field_mix must be in (0, 1]")
        if not (0.0 < self.theta_mix <= 1.0):
            raise ValueError("theta_mix must be in (0, 1]")


@dataclass
class ExistenceCurveResult(BaseResult):
    results: list[BaseResult] | None = None


def _with_total_power(beams: BeamExperiment, total_power: float) -> BeamExperiment:
    beams.validate()
    old_total = sum(float(ch.power) for ch in beams.channels)
    if old_total <= 0.0:
        scale_powers = [float(total_power) / len(beams.channels)] * len(beams.channels)
    else:
        scale = float(total_power) / old_total
        scale_powers = [float(ch.power) * scale for ch in beams.channels]
    new_channels: list[BeamChannel] = [
        replace(ch, power=float(p))
        for ch, p in zip(beams.channels, scale_powers)
    ]
    return BeamExperiment(channels=tuple(new_channels), coherence=beams.coherence)


def _with_backend_verbosity(req: StaticRequest, *, verbose: bool) -> StaticRequest:
    b = req.backend
    return replace(
        req,
        backend=BackendSpec(backend=b.backend, precision=b.precision, verbose=bool(verbose)),
    )


def make_static_request_for_power(base: StaticRequest, power: float, *, verbose: bool | None = None) -> StaticRequest:
    req = replace(base, beams=_with_total_power(base.beams, float(power)))
    if verbose is not None:
        req = _with_backend_verbosity(req, verbose=verbose)
    return req


def _run_one_soliton(
    request: ExistenceCurveRequest,
    static_req: StaticRequest,
    *,
    initial_A=None,
    initial_theta=None,
) -> SolitonResult:
    return run_soliton(SolitonRequest(
        base=static_req,
        max_outer=int(request.soliton_max_outer),
        theta_steps_per_outer=int(request.theta_steps_per_outer),
        field_mix=float(request.field_mix),
        theta_mix=float(request.theta_mix),
        tol_field=float(request.tol_field),
        tol_theta=float(request.tol_theta),
        tol_residual_rms=float(request.tol_residual_rms),
        tol_residual_max=float(request.tol_residual_max),
        initial_A=initial_A,
        initial_theta=initial_theta,
    ))


def run_existence_curve(request: ExistenceCurveRequest) -> ExistenceCurveResult:
    request.validate()

    rows: list[dict] = []
    results: list[BaseResult] = []
    previous_A = None
    previous_theta = None

    for i, requested_power in enumerate(request.powers):
        verbose = bool(request.base.backend.verbose and i == 0)
        static_req = make_static_request_for_power(request.base, float(requested_power), verbose=verbose)
        use_continuation = bool(request.continuation and previous_theta is not None)

        if request.solver == "soliton":
            res = _run_one_soliton(
                request,
                static_req,
                initial_A=previous_A if use_continuation else None,
                initial_theta=previous_theta if use_continuation else None,
            )
            previous_A = res.A
            previous_theta = res.theta
            continuation_used = use_continuation
        elif request.solver == "static_zstack":
            res = run_static_zstack(
                static_req,
                initial_theta=previous_theta if use_continuation else None,
            )
            previous_A = None
            previous_theta = res.theta
            continuation_used = use_continuation
        else:
            raise AssertionError("unreachable")

        results.append(res)

        row = {
            "i": int(i),
            "solver": request.solver,
            "requested_power": float(requested_power),
            "seed_power": None if i == 0 else float(request.powers[i - 1]),

            "converged": bool(res.converged),

            # continuation bookkeeping
            "continuation_used": bool(continuation_used),
            "continuation_direction": (
                "up"
                if i == 0 or requested_power >= request.powers[i - 1]
                else "down"
            ),
            "used_initial_A": bool(res.metrics.get("used_initial_A", False)),
            "used_initial_theta": bool(res.metrics.get("used_initial_theta", False)),
        }

        row.update(res.metrics)
        row["output_power"] = float(res.metrics.get("power", 0.0))
        rows.append(row)

    metrics: dict = {
        "num_points": len(rows),
        "power_min": float(min(request.powers)),
        "power_max": float(max(request.powers)),
        "converged_count": int(sum(1 for r in results if r.converged)),
        "continuation": bool(request.continuation),
        "solver": request.solver,
    }
    if rows:
        metrics["last_requested_power"] = float(rows[-1]["requested_power"])
        metrics["last_output_power"] = float(rows[-1]["output_power"])
        metrics["last_theta_max"] = float(rows[-1].get("theta_max", 0.0))
        metrics["last_Imax"] = float(rows[-1].get("Imax", 0.0))
        metrics["last_beta"] = float(rows[-1].get("beta", float("nan")))
        metrics["last_residual_rms"] = float(rows[-1].get("residual_rms", 0.0))
        metrics["last_residual_max"] = float(rows[-1].get("residual_max", 0.0))

    return ExistenceCurveResult(
        kind="ExistenceCurveResult",
        metrics=metrics,
        samples=rows,
        results=results,
    )


__all__ = [
    "ExistenceSolver",
    "ExistenceCurveRequest",
    "ExistenceCurveResult",
    "make_static_request_for_power",
    "run_existence_curve",
]
