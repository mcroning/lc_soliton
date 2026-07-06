"""Static optical propagation workflow.

"Static" here means optical propagation through an LC director field that is
assumed to relax instantaneously to the steady time-independent PDE.

For each outer self-consistency pass:

    launch A
    for each z slice:
        compute midpoint intensity
        solve steady theta PDE for that slice
        propagate A through that theta slice

This is the workflow for fixed/instantaneous LC response beam propagation:
collisions, angled beams, spiraling beams, and other prescribed launches.

For the lower-level fixed-intensity director solve, use
``workflows.theta_static.run_theta_static``.
For stationary nonlinear eigenmodes, use ``workflows.soliton.run_soliton``.
For finite-time LC dynamics, use ``workflows.timedependent``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import time as _time

from ..request import StaticRequest
from ..result import StaticResult
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.bias import build_bias, stack_theta
from ..physics.launch import build_launch
from ..physics.liquid_crystal import resolved_b, neff_from_theta
from ..physics.coupling import resolved_bi
from ..algorithms.splitstep import (
    advance_slice_with_midintensity,
    linear_kernel,
    total_intensity,
)
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.theta_picard import cn_trapezoid_picard_step
from ..algorithms.thomas import solve_const_offdiag_batched
from ..products.diagnostics import intensity_metrics, residual_theta_static


@dataclass(frozen=True)
class StaticPropagationControls:
    """Controls for instantaneous-equilibrium static propagation."""

    theta_steps_per_slice: int = 25
    theta_picard_iters: int = 4
    theta_picard_tol: float = 1e-6
    theta_dt: float = 7.5e-4

    observer_stride: int = 1

    dz_opt_max_phi: float = 0.30
    dn_max_est: float = 0.02
    max_substeps: int = 16


def _select_tridiag_solver(request: StaticRequest):
    if request.tridiag == "fast":
        from ..algorithms.thomas_fast import solve_const_offdiag_batched_fast
        return solve_const_offdiag_batched_fast
    return solve_const_offdiag_batched


def _prepare_theta_stack(initial_theta: Any, bias, grid, theta_bc: float):
    xp = grid.xp
    if initial_theta is None:
        theta_stack = bias.theta_stack.copy()
    else:
        theta = xp.asarray(initial_theta, dtype=grid.real_dtype)
        if theta.shape == bias.theta_2d.shape:
            theta_stack = stack_theta(theta, grid)
        elif theta.shape == bias.theta_stack.shape:
            theta_stack = theta.copy()
        else:
            raise ValueError(
                f"initial_theta shape {theta.shape} must be "
                f"{bias.theta_2d.shape} or {bias.theta_stack.shape}"
            )
    theta_stack[:, 0, :] = theta_bc
    theta_stack[:, -1, :] = theta_bc
    return theta_stack


def _stack_update_metrics(theta_stack, theta_prev, xp) -> dict[str, float]:
    d = theta_stack - theta_prev
    return {
        "dtheta_rms": float(asnumpy(xp.sqrt(xp.mean(d * d)))),
        "dtheta_max": float(asnumpy(xp.max(xp.abs(d)))),
    }


def _stack_residual_metrics(theta_stack, intensity_stack, *, b, bi, grid, theta_bc) -> dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    count = 0

    for k in range(int(grid.Nz)):
        r = residual_theta_static(
            theta_stack[k],
            intensity_stack[k],
            b=b,
            bi=bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=theta_bc,
            xp=grid.xp,
        )
        n = (int(grid.Nx) - 2) * int(grid.Ny)
        total_sq += float(r["residual_rms"]) ** 2 * n
        max_abs = max(max_abs, float(r["residual_max"]))
        count += n

    return {
        "residual_rms": math.sqrt(total_sq / max(1, count)),
        "residual_max": max_abs,
    }


def run_static(
    request: StaticRequest,
    *,
    initial_theta: Any | None = None,
    controls: StaticPropagationControls | None = None,
) -> StaticResult:
    """Run static instantaneous-equilibrium optical propagation."""

    request.validate()
    controls = StaticPropagationControls() if controls is None else controls

    backend = get_backend(request.backend)
    xp = backend.xp

    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    tridiag_solver = _select_tridiag_solver(request)

    b = resolved_b(request.lc)
    bi = resolved_bi(request.lc, request.beams)
    theta_bc = float(request.lc.cell.theta_bc)

    theta_stack = _prepare_theta_stack(initial_theta, bias, grid, theta_bc)

    n_ref = float(asnumpy(neff_from_theta(
        bias.theta_2d,
        ne=request.lc.material.ne,
        no=request.lc.material.no,
        xp=xp,
    )).mean())
    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))

    phi_est = (2.0 * math.pi / wavelength_um) * float(grid.dz_um) * float(controls.dn_max_est)
    Nsub = max(
        1,
        min(
            int(controls.max_substeps),
            int(math.ceil(phi_est / float(controls.dz_opt_max_phi))),
        ),
    )
    dz_sub = float(grid.dz_um) / int(Nsub)
    h_sub = linear_kernel(
        grid.fxy2_um,
        dz=dz_sub,
        wavelength=wavelength_um,
        n_ref=n_ref,
        xp=xp,
    )

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=float(controls.theta_dt),
        mobility=request.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    def solve_theta_slice(theta_seed, intensity):
        out = theta_seed
        for _ in range(int(controls.theta_steps_per_slice)):
            out = cn_trapezoid_picard_step(
                out,
                intensity,
                intensity,
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
            out[0, :] = theta_bc
            out[-1, :] = theta_bc
        return out

    I_stack = xp.zeros_like(theta_stack)
    history: list[dict] = []
    samples: list[dict] = []
    converged = False

    A_last = launch.A0.copy()
    final_I = total_intensity(A_last, coherent=(launch.coherence == "coherent"), xp=xp)

    synchronize(xp)
    t0 = _time.perf_counter()

    for outer in range(int(request.max_outer)):
        theta_old = theta_stack.copy()
        theta_new = theta_stack.copy()

        A = launch.A0.copy()

        for k in range(int(grid.Nz)):
            # First propagate a candidate slice with the previous theta to get
            # a midpoint intensity. Then solve the instantaneous steady theta
            # for that midpoint intensity and re-propagate the same incoming
            # field through the updated theta.
            A_in = A.copy()

            A_tmp, _I0, _I1, I_mid = advance_slice_with_midintensity(
                A_in.copy(),
                theta_old[k],
                kernel=h_sub,
                dz=float(grid.dz_um),
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.lc.material.ne,
                no=request.lc.material.no,
                Nsub=int(Nsub),
                coherent=(launch.coherence == "coherent"),
                theta_weights=launch.theta_weights,
                xp=xp,
            )

            th = solve_theta_slice(theta_old[k], I_mid)
            theta_new[k] = th
            I_stack[k] = I_mid.astype(grid.real_dtype, copy=False)

            A, _I0b, _I1b, I_mid_b = advance_slice_with_midintensity(
                A_in,
                th,
                kernel=h_sub,
                dz=float(grid.dz_um),
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.lc.material.ne,
                no=request.lc.material.no,
                Nsub=int(Nsub),
                coherent=(launch.coherence == "coherent"),
                theta_weights=launch.theta_weights,
                xp=xp,
            )
            I_stack[k] = I_mid_b.astype(grid.real_dtype, copy=False)

        theta_stack = theta_new
        theta_stack[:, 0, :] = theta_bc
        theta_stack[:, -1, :] = theta_bc

        A_last = A
        final_I = total_intensity(A_last, coherent=(launch.coherence == "coherent"), xp=xp)

        update = _stack_update_metrics(theta_stack, theta_old, xp)
        residual = _stack_residual_metrics(
            theta_stack,
            I_stack,
            b=b,
            bi=bi,
            grid=grid,
            theta_bc=theta_bc,
        )

        info = {
            "outer": int(outer + 1),
            **update,
            **residual,
            "update_converged": bool(
                update["dtheta_rms"] < request.tol_rms
                and update["dtheta_max"] < request.tol_max
            ),
            "residual_converged": bool(
                residual["residual_rms"] < request.tol_residual_rms
                and residual["residual_max"] < request.tol_residual_max
            ),
            "theta_max": float(asnumpy(xp.max(theta_stack))),
        }
        info.update(intensity_metrics(final_I, grid))
        history.append(info)

        if outer % max(1, int(controls.observer_stride)) == 0:
            samples.append(dict(info))

        if info["update_converged"] and info["residual_converged"]:
            converged = True
            break

    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    metrics = intensity_metrics(final_I, grid)
    metrics.update(_stack_residual_metrics(
        theta_stack,
        I_stack,
        b=b,
        bi=bi,
        grid=grid,
        theta_bc=theta_bc,
    ))

    if history:
        for key in (
            "dtheta_rms",
            "dtheta_max",
            "update_converged",
            "residual_converged",
        ):
            metrics[key] = history[-1][key]

    metrics.update({
        "backend": backend.name,
        "precision": request.backend.precision,
        "tridiag": request.tridiag,
        "Nx": int(grid.Nx),
        "Ny": int(grid.Ny),
        "Nz": int(grid.Nz),
        "outer_steps": len(history),
        "theta_steps_per_slice": int(controls.theta_steps_per_slice),
        "theta_picard_iters": int(controls.theta_picard_iters),
        "theta_dt": float(controls.theta_dt),
        "converged": bool(converged),
        "used_initial_theta": bool(initial_theta is not None),
        "b": float(b),
        "bi": float(bi),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(theta_stack))),
        "theta_shape": tuple(int(x) for x in theta_stack.shape),
    })

    return StaticResult(
        kind="StaticPropagationResult",
        metrics=metrics,
        samples=samples,
        theta=theta_stack,
        intensity=I_stack,
        history=history,
        converged=bool(converged),
    )


__all__ = [
    "StaticPropagationControls",
    "run_static",
]
