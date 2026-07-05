"""Static/self-consistent LC workflow.

v006 adds ``static_inner_steps``: multiple theta pseudo-time CN/Picard updates
are performed for each optical outer step. This makes the static workflow a
more faithful static relaxer instead of a single-step fixed-point iteration.
"""

from __future__ import annotations

import math
import time as _time
from typing import Any

from ..request import StaticRequest
from ..result import StaticResult
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.liquid_crystal import resolved_b, neff_from_theta
from ..physics.bias import build_bias
from ..physics.launch import build_launch
from ..algorithms.splitstep import advance_slice, linear_kernel, total_intensity
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.theta_picard import cn_trapezoid_picard_step
from ..algorithms.thomas import solve_const_offdiag_batched
from ..algorithms.static_relax import StaticRelaxControls, run_static_relax
from ..products.diagnostics import intensity_metrics, theta_update_metrics, residual_theta_static


def _select_tridiag_solver(request: StaticRequest):
    if request.tridiag == "fast":
        from ..algorithms.thomas_fast import solve_const_offdiag_batched_fast
        return solve_const_offdiag_batched_fast
    return solve_const_offdiag_batched


def _prepare_initial_theta(initial_theta: Any, bias_theta, grid, theta_bc: float):
    if initial_theta is None:
        return bias_theta
    xp = grid.xp
    theta = xp.asarray(initial_theta, dtype=grid.real_dtype).copy()
    if theta.shape != bias_theta.shape:
        raise ValueError(f"initial_theta shape {theta.shape} does not match {bias_theta.shape}")
    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc
    return theta


def run_static(request: StaticRequest, *, initial_theta: Any | None = None) -> StaticResult:
    """Run a static/self-consistent LC request."""

    request.validate()

    backend = get_backend(request.backend)
    xp = backend.xp

    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    tridiag_solver = _select_tridiag_solver(request)

    b = resolved_b(request.lc)
    bi = 214.28571428571428

    theta0 = _prepare_initial_theta(initial_theta, bias.theta_2d, grid, request.lc.cell.theta_bc)

    n_ref = float(asnumpy(neff_from_theta(
        theta0,
        ne=request.lc.material.ne,
        no=request.lc.material.no,
        xp=xp,
    )).mean())

    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))

    dn_max_est = 0.02
    dz_opt_max_phi = 0.30
    max_substeps = 16
    phi_est = (2.0 * math.pi / wavelength_um) * float(grid.dz_um) * dn_max_est
    Nsub = max(1, min(max_substeps, int(math.ceil(phi_est / dz_opt_max_phi))))
    dz_sub = float(grid.dz_um) / Nsub

    h_sub = linear_kernel(
        grid.fxy2_um,
        dz=dz_sub,
        wavelength=wavelength_um,
        n_ref=n_ref,
        xp=xp,
    )

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=7.5e-4,
        mobility=request.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    A0 = launch.A0
    intensity0 = total_intensity(A0, coherent=(launch.coherence == "coherent"), xp=xp)
    latest_intensity = {"value": intensity0}

    samples: list[dict] = []

    def one_theta_step(theta, intensity):
        out = cn_trapezoid_picard_step(
            theta,
            intensity,
            intensity,
            b=b,
            bi=bi,
            dt=7.5e-4,
            mobility=request.lc.mobility,
            dx=grid.du,
            dy=grid.dv,
            s=s_cn,
            off=off_cn,
            diag=diag_cn,
            max_iter=4,
            tol_update=1e-6,
            clamp=bias.theta_clamp,
            tridiag_solver=tridiag_solver,
            xp=xp,
        )
        out[0, :] = request.lc.cell.theta_bc
        out[-1, :] = request.lc.cell.theta_bc
        return out

    def theta_relax(theta, intensity, outer):
        out = theta
        for _ in range(int(request.static_inner_steps)):
            out = one_theta_step(out, intensity)
        return out

    def optics_update(A, theta, outer):
        Awork = A0.copy()
        for _k in range(grid.Nz):
            advance_slice(
                Awork,
                theta,
                kernel=h_sub,
                dz=float(grid.dz_um),
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.lc.material.ne,
                no=request.lc.material.no,
                Nsub=Nsub,
                xp=xp,
            )
        I = total_intensity(Awork, coherent=(launch.coherence == "coherent"), xp=xp)
        latest_intensity["value"] = I
        return Awork, I

    def convergence(theta, theta_prev, info):
        intensity = latest_intensity["value"]
        update = theta_update_metrics(theta, theta_prev)
        residual = residual_theta_static(
            theta,
            intensity,
            b=b,
            bi=bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=request.lc.cell.theta_bc,
            xp=xp,
        )
        info.update(update)
        info.update(residual)
        info["update_converged"] = bool(
            update["dtheta_rms"] < request.tol_rms
            and update["dtheta_max"] < request.tol_max
        )
        info["residual_converged"] = bool(
            residual["residual_rms"] < request.tol_residual_rms
            and residual["residual_max"] < request.tol_residual_max
        )
        return bool(info["update_converged"] and info["residual_converged"])

    def observer(payload):
        info = dict(payload["info"])
        theta = payload["theta"]
        intensity = payload["intensity"]
        mm = intensity_metrics(intensity, grid)
        mm.update(info)
        mm["theta_max"] = float(asnumpy(xp.max(theta)))
        samples.append(mm)

    synchronize(xp)
    t0 = _time.perf_counter()
    result = run_static_relax(
        theta0,
        A0,
        intensity0,
        theta_relax=theta_relax,
        optics_update=optics_update,
        controls=StaticRelaxControls(max_outer=request.max_outer, observer_stride=1),
        convergence=convergence,
        observer=observer,
    )
    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    metrics = intensity_metrics(result.intensity, grid)
    metrics.update(residual_theta_static(
        result.theta,
        result.intensity,
        b=b,
        bi=bi,
        dx=grid.du,
        dy=grid.dv,
        theta_bc=request.lc.cell.theta_bc,
        xp=xp,
    ))
    metrics.update({
        "backend": backend.name,
        "precision": request.backend.precision,
        "tridiag": request.tridiag,
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Nz": grid.Nz,
        "outer_steps": int(result.outer_steps),
        "static_inner_steps": int(request.static_inner_steps),
        "converged": bool(result.converged),
        "used_initial_theta": bool(initial_theta is not None),
        "b": float(b),
        "bi": float(bi),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(result.theta))),
    })

    if result.history:
        last = result.history[-1]
        for key in (
            "dtheta_rms",
            "dtheta_max",
            "update_converged",
            "residual_converged",
        ):
            if key in last:
                metrics[key] = last[key]

    return StaticResult(
        kind="StaticResult",
        metrics=metrics,
        samples=samples,
        theta=result.theta,
        intensity=result.intensity,
        history=result.history,
        converged=result.converged,
    )


__all__ = ["run_static"]
