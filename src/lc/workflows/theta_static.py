"""Steady-state LC director workflow for a prescribed intensity.

This module solves the time-independent director equation

    lap(theta) + (b + bi I) sin(2 theta) = 0

for a fixed transverse intensity ``I(x,y)``. It does not propagate light and
does not solve the nonlinear optical eigenmode problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time as _time

from ..request import StaticRequest
from ..result import StaticResult
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.bias import build_bias
from ..physics.launch import build_launch
from ..physics.liquid_crystal import resolved_b
from ..algorithms.static_relax import StaticRelaxControls, run_static_relax
from ..physics.coupling import resolved_bi
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.theta_picard import cn_trapezoid_picard_step
from ..algorithms.thomas import solve_const_offdiag_batched
from ..algorithms.splitstep import total_intensity
from ..products.diagnostics import intensity_metrics, residual_theta_static


@dataclass(frozen=True)
class ThetaStaticControls:
    """Controls for the fixed-intensity steady theta solve."""

    theta_steps_per_outer: int = 25
    theta_picard_iters: int = 4
    theta_picard_tol: float = 1e-6
    theta_dt: float = 7.5e-4
    observer_stride: int = 1


def _select_tridiag_solver(request: StaticRequest):
    if request.tridiag == "fast":
        from ..algorithms.thomas_fast import solve_const_offdiag_batched_fast
        return solve_const_offdiag_batched_fast
    return solve_const_offdiag_batched


def _prepare_initial_theta(theta0: Any, bias, *, xp):
    if theta0 is None:
        return bias.theta_2d.copy()
    theta = xp.asarray(theta0, dtype=bias.theta_2d.dtype)
    if theta.ndim == 3:
        theta = theta[int(theta.shape[0]) // 2]
    if theta.shape != bias.theta_2d.shape:
        raise ValueError(f"theta0 shape {theta.shape} does not match {bias.theta_2d.shape}")
    return theta.copy()


def _prepare_intensity(intensity: Any, launch, grid, *, xp):
    if intensity is None:
        return total_intensity(launch.A0, coherent=(launch.coherence == "coherent"), xp=xp)
    I = xp.asarray(intensity, dtype=grid.real_dtype)
    if I.shape != (grid.Nx, grid.Ny):
        raise ValueError(f"intensity shape {I.shape} does not match {(grid.Nx, grid.Ny)}")
    return I.copy()


def run_theta_static(
    request: StaticRequest,
    *,
    intensity: Any | None = None,
    theta0: Any | None = None,
    A0: Any | None = None,
    controls: ThetaStaticControls | None = None,
) -> StaticResult:
    """Solve the fixed-intensity steady director PDE."""

    request.validate()
    controls = ThetaStaticControls() if controls is None else controls

    backend = get_backend(request.backend)
    xp = backend.xp
    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    tridiag_solver = _select_tridiag_solver(request)

    b = resolved_b(request.lc)
    bi = resolved_bi(request.lc, request.beams)

    theta_init = _prepare_initial_theta(theta0, bias, xp=xp)
    I_fixed = _prepare_intensity(intensity, launch, grid, xp=xp)

    if A0 is None:
        A_init = launch.A0.copy()
    else:
        A_init = xp.asarray(A0, dtype=launch.A0.dtype).copy()
        if A_init.shape != launch.A0.shape:
            raise ValueError(f"A0 shape {A_init.shape} does not match {launch.A0.shape}")

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=float(controls.theta_dt),
        mobility=request.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    def theta_relax(theta, intensity_local, outer):
        out = theta
        for _ in range(int(controls.theta_steps_per_outer)):
            out = cn_trapezoid_picard_step(
                out,
                intensity_local,
                intensity_local,
                b=b,
                bi=bi,
                dt=float(controls.theta_dt),
                mobility=request.lc.mobility,
                dx=grid.du,
                dy=grid.dv,
                s=s_cn,
                off=off_cn,
                diag=diag_cn,
                max_iter=int(controls.theta_picard_iters),
                tol_update=float(controls.theta_picard_tol),
                clamp=bias.theta_clamp,
                tridiag_solver=tridiag_solver,
                xp=xp,
            )
            out[0, :] = request.lc.cell.theta_bc
            out[-1, :] = request.lc.cell.theta_bc
        return out

    def convergence(theta, theta_prev, info):
        d = theta - theta_prev
        dtheta_rms = float(asnumpy(xp.sqrt(xp.mean(d * d))))
        dtheta_max = float(asnumpy(xp.max(xp.abs(d))))

        resid = residual_theta_static(
            theta,
            I_fixed,
            b=b,
            bi=bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=request.lc.cell.theta_bc,
            xp=xp,
        )

        info.update({
            "dtheta_rms": dtheta_rms,
            "dtheta_max": dtheta_max,
            **resid,
        })

        return bool(
            dtheta_rms < float(request.tol_rms)
            and dtheta_max < float(request.tol_max)
            and resid["residual_rms"] < float(request.tol_residual_rms)
            and resid["residual_max"] < float(request.tol_residual_max)
        )

    history: list[dict] = []

    def observer(payload):
        info = dict(payload["info"])
        info.update(intensity_metrics(I_fixed, grid))
        info["theta_max"] = float(asnumpy(xp.max(payload["theta"])))
        history.append(info)

    synchronize(xp)
    t0 = _time.perf_counter()

    rr = run_static_relax(
        theta_init,
        A_init,
        I_fixed,
        theta_relax=theta_relax,
        optics_update=None,
        controls=StaticRelaxControls(
            max_outer=int(request.max_outer),
            observer_stride=int(controls.observer_stride),
        ),
        convergence=convergence,
        observer=observer,
    )

    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    metrics = intensity_metrics(rr.intensity, grid)
    metrics.update(residual_theta_static(
        rr.theta,
        rr.intensity,
        b=b,
        bi=bi,
        dx=grid.du,
        dy=grid.dv,
        theta_bc=request.lc.cell.theta_bc,
        xp=xp,
    ))

    if rr.history:
        last = rr.history[-1]
        for key in ("dtheta_rms", "dtheta_max"):
            if key in last:
                metrics[key] = last[key]

    metrics.update({
        "backend": backend.name,
        "precision": request.backend.precision,
        "tridiag": request.tridiag,
        "Nx": int(grid.Nx),
        "Ny": int(grid.Ny),
        "Nz": int(grid.Nz),
        "outer_steps": int(rr.outer_steps),
        "theta_steps_per_outer": int(controls.theta_steps_per_outer),
        "theta_picard_iters": int(controls.theta_picard_iters),
        "theta_dt": float(controls.theta_dt),
        "converged": bool(rr.converged),
        "b": float(b),
        "bi": float(bi),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(rr.theta))),
    })

    return StaticResult(
        kind="ThetaStaticResult",
        metrics=metrics,
        samples=history,
        theta=rr.theta,
        intensity=rr.intensity,
        history=rr.history,
        converged=bool(rr.converged),
    )


__all__ = [
    "ThetaStaticControls",
    "run_theta_static",
]
