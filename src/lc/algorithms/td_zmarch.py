"""Black-box time-dependent z-march algorithm.

This module is intentionally experiment-free. It knows only arrays, step
functions, loop controls, and an optional observer.

It does not know about liquid-crystal materials, beam geometry, launch
construction, files, GUI state, eigensolitons, or diagnostics products.

Algorithm pattern
-----------------
For each physical time step:

    theta_ref = theta.copy()
    A = A0.copy()

    optional optics half-step

    for k in z:
        A, I_mid = optics_step(A, theta_ref[k], k)
        theta[k] = theta_step(theta_ref[k], I_mid, theta_ref[k-1], theta_ref[k+1], k)

    optional optics half-step

The caller supplies the optics and theta slice black boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

Array = Any

OpticsStep = Callable[[Array, Array, int], tuple[Array, Array]]
ThetaStep = Callable[[Array, Array, Array, Array, int], Array]
HalfStep = Callable[[Array], Array]
Observer = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class TDZMarchControls:
    """Loop controls for ``run_td_zmarch``."""

    Nt: int
    observer_stride_z: int = 1
    observer_stride_t: int = 1


@dataclass
class TDZMarchResult:
    """Result from ``run_td_zmarch``."""

    theta: Array
    A_last: Array
    steps: int


def _validate_inputs(theta0: Array, A0: Array, controls: TDZMarchControls) -> None:
    if theta0.ndim != 3:
        raise ValueError("theta0 must have shape (Nz, Nx, Ny)")
    if A0.ndim != 3:
        raise ValueError("A0 must have shape (Nch, Nx, Ny)")
    if theta0.shape[1:] != A0.shape[1:]:
        raise ValueError("theta0 spatial shape must match A0 spatial shape")
    if int(controls.Nt) < 0:
        raise ValueError("Nt must be nonnegative")
    if int(controls.observer_stride_z) < 1:
        raise ValueError("observer_stride_z must be >= 1")
    if int(controls.observer_stride_t) < 1:
        raise ValueError("observer_stride_t must be >= 1")


def run_td_zmarch(
    theta0: Array,
    A0: Array,
    *,
    optics_step: OpticsStep,
    theta_step: ThetaStep,
    controls: TDZMarchControls,
    optics_half_step: Optional[HalfStep] = None,
    observer: Observer | None = None,
) -> TDZMarchResult:
    """Run repeated z-passes with frozen-time theta references.

    Parameters
    ----------
    theta0
        Initial theta stack, shape ``(Nz, Nx, Ny)``.
    A0
        Entrance launch channel stack, shape ``(Nch, Nx, Ny)``.
    optics_step
        Callable ``A, I_mid = optics_step(A, theta_ref_k, k)``. It advances
        the optical channel stack through slice ``k`` and returns the midpoint
        plain intensity seen by the theta solver.
    theta_step
        Callable ``theta_new = theta_step(theta_k, I_mid, theta_prev, theta_next, k)``.
        The z-neighbor slices are from the frozen time reference stack.
    controls
        Time-loop controls.
    optics_half_step
        Optional callable applied to ``A`` at the beginning and end of each
        z-pass. This preserves the trusted runner's half-step convention while
        keeping the algorithm unaware of Fourier kernels.
    observer
        Optional callable receiving dictionaries with ``jt``, ``k``, ``A``,
        ``I_mid``, and ``theta_k``.
    """

    _validate_inputs(theta0, A0, controls)

    theta = theta0.copy()
    A_last = A0.copy()
    Nz = int(theta.shape[0])

    for jt in range(1, int(controls.Nt) + 1):
        theta_ref = theta.copy()
        A = A0.copy()

        if optics_half_step is not None:
            A = optics_half_step(A)

        for k in range(Nz):
            theta_k = theta_ref[k]
            theta_prev = theta_ref[k - 1] if k > 0 else theta_ref[k]
            theta_next = theta_ref[k + 1] if (k + 1) < Nz else theta_ref[k]

            A, I_mid = optics_step(A, theta_k, k)
            theta[k] = theta_step(theta_k, I_mid, theta_prev, theta_next, k)

            if (
                observer is not None
                and jt % int(controls.observer_stride_t) == 0
                and k % int(controls.observer_stride_z) == 0
            ):
                observer(
                    {
                        "jt": jt,
                        "k": k,
                        "A": A,
                        "I_mid": I_mid,
                        "theta_k": theta[k],
                    }
                )

        if optics_half_step is not None:
            A = optics_half_step(A)

        A_last = A

    return TDZMarchResult(theta=theta, A_last=A_last, steps=int(controls.Nt))


__all__ = [
    "TDZMarchControls",
    "TDZMarchResult",
    "run_td_zmarch",
]
