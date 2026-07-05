"""Existence-curve workflow with theta continuation.

This workflow orchestrates a sequence of static solves while sweeping optical
power. It contains no new numerical algorithms.

v003 adds continuation: each power can start from the previous power's solved
theta field. This is the normal existence-curve behavior and is essential for
hard high-power points.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..request import StaticRequest
from ..result import BaseResult, StaticResult
from ..physics.beam import BeamChannel, BeamExperiment
from ..numerics.backend import BackendSpec
from .static_zstack import run_static_zstack


@dataclass(frozen=True)
class ExistenceCurveRequest:
    """Request for a static power sweep."""

    base: StaticRequest
    powers: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    continuation: bool = True

    def validate(self) -> None:
        self.base.validate()
        if len(self.powers) == 0:
            raise ValueError("powers must be nonempty")
        for p in self.powers:
            if p < 0.0:
                raise ValueError("powers must be nonnegative")


@dataclass
class ExistenceCurveResult(BaseResult):
    """Result for an existence-curve power sweep."""

    results: list[StaticResult] | None = None


def _with_total_power(beams: BeamExperiment, total_power: float) -> BeamExperiment:
    """Return BeamExperiment with channel powers scaled to total_power."""

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
    """Return request copy with backend verbosity changed."""
    b = req.backend
    return replace(
        req,
        backend=BackendSpec(
            backend=b.backend,
            precision=b.precision,
            verbose=bool(verbose),
        ),
    )


def make_static_request_for_power(
    base: StaticRequest,
    power: float,
    *,
    verbose: bool | None = None,
) -> StaticRequest:
    """Return a copy of ``base`` with beam total power changed."""
    req = replace(base, beams=_with_total_power(base.beams, float(power)))
    if verbose is not None:
        req = _with_backend_verbosity(req, verbose=verbose)
    return req


def run_existence_curve(request: ExistenceCurveRequest) -> ExistenceCurveResult:
    """Run a sequence of static solves over requested optical power."""

    request.validate()

    rows: list[dict] = []
    results: list[StaticResult] = []
    previous_theta = None

    for i, requested_power in enumerate(request.powers):
        verbose = bool(request.base.backend.verbose and i == 0)
        static_req = make_static_request_for_power(
            request.base,
            float(requested_power),
            verbose=verbose,
        )

        use_continuation = bool(request.continuation and previous_theta is not None)
        res = run_static_zstack(
            static_req,
            initial_theta=previous_theta if use_continuation else None,
        )
        results.append(res)
        previous_theta = res.theta

        row = {
            "i": int(i),
            "requested_power": float(requested_power),
            "converged": bool(res.converged),
            "continuation_used": bool(use_continuation),
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
    }

    if rows:
        metrics["last_requested_power"] = float(rows[-1]["requested_power"])
        metrics["last_output_power"] = float(rows[-1]["output_power"])
        metrics["last_theta_max"] = float(rows[-1].get("theta_max", 0.0))
        metrics["last_Imax"] = float(rows[-1].get("Imax", 0.0))
        metrics["last_residual_rms"] = float(rows[-1].get("residual_rms", 0.0))
        metrics["last_residual_max"] = float(rows[-1].get("residual_max", 0.0))

    return ExistenceCurveResult(
        kind="ExistenceCurveResult",
        metrics=metrics,
        samples=rows,
        results=results,
    )


__all__ = [
    "ExistenceCurveRequest",
    "ExistenceCurveResult",
    "make_static_request_for_power",
    "run_existence_curve",
]
