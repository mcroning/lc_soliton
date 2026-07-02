"""Black-box z-stack time-dependent LC/optics march.

Version: td_zstack_v001

This module intentionally knows nothing about the experiment that produced the
arrays. It does not know about voltages, beam waists, wavelengths, materials,
GUI settings, file paths, movies, or eigensolitons.

It consumes only:
    * theta stack arrays
    * multichannel optical launch stack
    * numerical step functions
    * loop controls
    * optional observers/diagnostics

The physical/numerical support layer is responsible for cooking its dinner:
building grids, launch fields, kernels, coefficients, buffers, and products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

Array = Any


class OpticsSliceStep(Protocol):
    """Advance the multichannel optical field through one z slice.

    Parameters
    ----------
    A:
        Current channel stack, shape ``(Nch, Nx, Ny)``.
    theta_k:
        Frozen theta slice, shape ``(Nx, Ny)``.
    k:
        z-slice index.
    jt:
        time-step index.

    Returns
    -------
    A_after, I_mid
        Updated optical channel stack and midpoint intensity used by theta.
        ``I_mid`` is plain intensity, not pre-multiplied by ``bi``.
    """

    def __call__(self, A: Array, theta_k: Array, *, k: int, jt: int) -> tuple[Array, Array]: ...


class ThetaSliceStep(Protocol):
    """Advance one theta slice using midpoint intensity and frozen z neighbors."""

    def __call__(
        self,
        theta_k: Array,
        I_mid: Array,
        theta_prev_z: Array,
        theta_next_z: Array,
        *,
        k: int,
        jt: int,
    ) -> Array: ...


Observer = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class TDZStackControls:
    """Loop controls for the black-box TD z-stack march."""

    Nt: int
    save_every_z: int = 0
    save_every_t: int = 1
    copy_theta_ref_each_t: bool = True
    reset_optics_each_t: bool = True


@dataclass
class TDZStackResult:
    """Minimal return object from the black-box march."""

    theta: Array
    A: Array
    steps: int
    history: list[dict[str, Any]] = field(default_factory=list)


def _shape3(name: str, a: Array) -> tuple[int, int, int]:
    if not hasattr(a, "shape") or len(a.shape) != 3:
        raise ValueError(f"{name} must have shape (N, Nx, Ny); got {getattr(a, 'shape', None)}")
    return int(a.shape[0]), int(a.shape[1]), int(a.shape[2])


def _z_neighbor(theta_ref: Array, k: int, offset: int) -> Array:
    Nz = int(theta_ref.shape[0])
    kk = k + offset
    if kk < 0:
        kk = 0
    elif kk >= Nz:
        kk = Nz - 1
    return theta_ref[kk]


def run_td_zstack_blackbox(
    *,
    theta0: Array,
    A0: Array,
    optics_step: OpticsSliceStep,
    theta_step: ThetaSliceStep,
    controls: TDZStackControls,
    copy_fn: Optional[Callable[[Array], Array]] = None,
    observer: Optional[Observer] = None,
) -> TDZStackResult:
    """Run the core TD z-stack algorithm.

    This is the small trusted algorithmic black box. All physics-specific and
    implementation-specific details are supplied through ``optics_step`` and
    ``theta_step``.

    Parameters
    ----------
    theta0:
        Initial theta stack, shape ``(Nz, Nx, Ny)``.
    A0:
        Initial multichannel optical launch, shape ``(Nch, Nx, Ny)``.
    optics_step:
        Numerical optics slice kernel. Returns ``(A_after, I_mid)``.
    theta_step:
        Numerical theta slice kernel. Consumes plain ``I_mid``.
    controls:
        Loop controls.
    copy_fn:
        Optional backend-specific copy function. Defaults to ``a.copy()``.
    observer:
        Optional callback receiving lightweight event dictionaries. It may
        record diagnostics/products but must not modify arrays.
    """

    Nz, Nx, Ny = _shape3("theta0", theta0)
    Nch, Ax, Ay = _shape3("A0", A0)
    if (Ax, Ay) != (Nx, Ny):
        raise ValueError(f"A0 transverse shape {(Ax, Ay)} does not match theta0 {(Nx, Ny)}")
    if int(controls.Nt) < 1:
        raise ValueError("controls.Nt must be >= 1")

    copy = copy_fn if copy_fn is not None else lambda a: a.copy()

    theta = copy(theta0)
    A_final = copy(A0)
    history: list[dict[str, Any]] = []

    for jt in range(1, int(controls.Nt) + 1):
        theta_ref = copy(theta) if controls.copy_theta_ref_each_t else theta
        A = copy(A0) if controls.reset_optics_each_t else copy(A_final)

        if observer is not None:
            observer({"event": "t_start", "jt": jt, "theta": theta, "A": A})

        for k in range(Nz):
            A, I_mid = optics_step(A, theta_ref[k], k=k, jt=jt)

            theta[k] = theta_step(
                theta_ref[k],
                I_mid,
                _z_neighbor(theta_ref, k, -1),
                _z_neighbor(theta_ref, k, +1),
                k=k,
                jt=jt,
            )

            do_save_t = (int(controls.save_every_t) > 0 and jt % int(controls.save_every_t) == 0)
            do_save_z = (int(controls.save_every_z) > 0 and k % int(controls.save_every_z) == 0)
            if observer is not None and do_save_t and do_save_z:
                observer({
                    "event": "z_sample",
                    "jt": jt,
                    "k": k,
                    "theta_k": theta[k],
                    "A": A,
                    "I_mid": I_mid,
                })

        A_final = A

        row = {"jt": jt, "Nz": Nz, "Nch": Nch}
        history.append(row)
        if observer is not None:
            observer({"event": "t_end", "jt": jt, "theta": theta, "A": A_final})

    return TDZStackResult(theta=theta, A=A_final, steps=int(controls.Nt), history=history)


__all__ = [
    "OpticsSliceStep",
    "ThetaSliceStep",
    "TDZStackControls",
    "TDZStackResult",
    "run_td_zstack_blackbox",
]
