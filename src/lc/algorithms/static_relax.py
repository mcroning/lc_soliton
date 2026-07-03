"""Black-box static relaxation algorithm.

This module is intentionally experiment-free. It knows only arrays, step
functions, loop controls, optional convergence, and an optional observer.

It does not know about LC materials, beam geometry, launch construction, files,
GUI state, eigensolitons, continuation, or diagnostics products.

Algorithm pattern
-----------------
Given an initial theta field, an optical channel stack, and an initial
plain intensity:

    for outer in self-consistency iterations:
        theta = theta_relax(theta, intensity, outer)

        if optics_update is provided:
            A, intensity = optics_update(A, theta, outer)

        check convergence / observe

The caller supplies the theta and optics black boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

Array = Any

ThetaRelax = Callable[[Array, Array, int], Array]
OpticsUpdate = Callable[[Array, Array, int], tuple[Array, Array]]
Convergence = Callable[[Array, Array, dict[str, Any]], bool]
Observer = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class StaticRelaxControls:
    """Loop controls for ``run_static_relax``."""

    max_outer: int = 20
    observer_stride: int = 1


@dataclass
class StaticRelaxResult:
    """Result from ``run_static_relax``."""

    theta: Array
    A: Array
    intensity: Array
    outer_steps: int
    converged: bool
    history: list[dict[str, Any]] = field(default_factory=list)


def _validate_inputs(theta0: Array, A0: Array, intensity0: Array, controls: StaticRelaxControls) -> None:
    if theta0.ndim != 2:
        raise ValueError("theta0 must have shape (Nx, Ny)")
    if A0.ndim != 3:
        raise ValueError("A0 must have shape (Nch, Nx, Ny)")
    if intensity0.ndim != 2:
        raise ValueError("intensity0 must have shape (Nx, Ny)")
    if theta0.shape != intensity0.shape:
        raise ValueError("theta0 and intensity0 shapes must match")
    if A0.shape[1:] != theta0.shape:
        raise ValueError("A0 spatial shape must match theta0")
    if int(controls.max_outer) < 0:
        raise ValueError("max_outer must be nonnegative")
    if int(controls.observer_stride) < 1:
        raise ValueError("observer_stride must be >= 1")


def default_never_converged(theta: Array, theta_prev: Array, info: dict[str, Any]) -> bool:
    """Default convergence rule: never stop early."""
    return False


def run_static_relax(
    theta0: Array,
    A0: Array,
    intensity0: Array,
    *,
    theta_relax: ThetaRelax,
    optics_update: Optional[OpticsUpdate] = None,
    controls: StaticRelaxControls = StaticRelaxControls(),
    convergence: Convergence | None = None,
    observer: Observer | None = None,
) -> StaticRelaxResult:
    """Run a static self-consistency loop.

    Parameters
    ----------
    theta0
        Initial theta field, shape ``(Nx, Ny)``.
    A0
        Initial optical channel stack, shape ``(Nch, Nx, Ny)``.
    intensity0
        Initial plain intensity supplied to the theta relaxer, shape
        ``(Nx, Ny)``.
    theta_relax
        Callable ``theta_new = theta_relax(theta, intensity, outer)``.
    optics_update
        Optional callable ``A_new, intensity_new = optics_update(A, theta, outer)``.
        If omitted, optics/intensity are frozen and only theta is relaxed.
    controls
        Loop controls.
    convergence
        Optional stopping rule ``convergence(theta, theta_prev, info)``.
    observer
        Optional callable receiving dictionaries with ``outer``, ``theta``,
        ``A``, ``intensity``, and ``info``.
    """

    _validate_inputs(theta0, A0, intensity0, controls)

    conv = default_never_converged if convergence is None else convergence

    theta = theta0.copy()
    A = A0.copy()
    intensity = intensity0.copy()
    history: list[dict[str, Any]] = []

    for outer in range(1, int(controls.max_outer) + 1):
        theta_prev = theta.copy()

        theta = theta_relax(theta, intensity, outer)

        if optics_update is not None:
            A, intensity = optics_update(A, theta, outer)

        info = {"outer": outer}
        converged = bool(conv(theta, theta_prev, info))
        info["converged"] = converged
        history.append(info)

        if observer is not None and outer % int(controls.observer_stride) == 0:
            observer(
                {
                    "outer": outer,
                    "theta": theta,
                    "theta_prev": theta_prev,
                    "A": A,
                    "intensity": intensity,
                    "info": info,
                }
            )

        if converged:
            return StaticRelaxResult(
                theta=theta,
                A=A,
                intensity=intensity,
                outer_steps=outer,
                converged=True,
                history=history,
            )

    return StaticRelaxResult(
        theta=theta,
        A=A,
        intensity=intensity,
        outer_steps=int(controls.max_outer),
        converged=False,
        history=history,
    )


__all__ = [
    "ThetaRelax",
    "OpticsUpdate",
    "Convergence",
    "Observer",
    "StaticRelaxControls",
    "StaticRelaxResult",
    "default_never_converged",
    "run_static_relax",
]
