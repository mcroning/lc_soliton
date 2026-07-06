"""Static z-stack LC workflow using strict self-consistent slice solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import time as _time

from ..request import StaticRequest
from ..result import StaticResult
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.liquid_crystal import resolved_b, neff_from_theta
from ..physics.coupling import resolved_bi
from ..physics.bias import build_bias, stack_theta
from ..physics.launch import build_launch
from ..algorithms.splitstep import linear_kernel, total_intensity
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.thomas import solve_const_offdiag_batched
from ..products.diagnostics import intensity_metrics
from .static_strict import strict_selfconsistent_slice, residual_stats_zcoupled_slice


@dataclass
class StaticZStackControls:
    observer_stride: int = 1
    theta_picard_iters: int = 4
    theta_picard_tol: float = 1e-6
    dz_opt_max_phi: float = 0.30
    dn_max_est: float = 0.02
    max_substeps: int = 16
    static_dt: float = 7.5e-4
    max_strict_outer_passes: int = 8
    max_selfcons_passes: int = 3
    selfcons_tol_theta: float = 1e-4
    selfcons_tol_I: float = 1e-4


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
            raise ValueError(f"initial_theta shape {theta.shape} must be {bias.theta_2d.shape} or {bias.theta_stack.shape}")
    theta_stack[:, 0, :] = theta_bc
    theta_stack[:, -1, :] = theta_bc
    return theta_stack


def _neighbors(theta_old, k: int):
    Nz = int(theta_old.shape[0])
    return theta_old[max(0, k - 1)], theta_old[min(Nz - 1, k + 1)]


def _stack_update_metrics(theta_stack, theta_prev, xp) -> dict[str, float]:
    d = theta_stack - theta_prev
    return {
        "dtheta_rms": float(asnumpy(xp.sqrt(xp.mean(d * d)))),
        "dtheta_max": float(asnumpy(xp.max(xp.abs(d)))),
    }


def _stack_residual_metrics(theta_stack, intensity_stack, *, b, bi, gamma_z, inv_dz2, grid) -> dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    count = 0
    for k in range(int(grid.Nz)):
        tp, tn = _neighbors(theta_stack, k)
        r = residual_stats_zcoupled_slice(
            theta_stack[k], intensity_stack[k], theta_prev=tp, theta_next=tn,
            b=b, bi=bi, gamma_z=gamma_z, inv_dz2=inv_dz2,
            dx=grid.du, dy=grid.dv, xp=grid.xp,
        )
        n = (int(grid.Nx) - 2) * int(grid.Ny)
        total_sq += float(r["rms_interior"]) ** 2 * n
        max_abs = max(max_abs, float(r["max_interior"]))
        count += n
    return {"residual_rms": math.sqrt(total_sq / max(1, count)), "residual_max": max_abs}

def _beta_from_overlap(A0, A1, *, grid, wavelength_um: float, n_ref: float, d_um: float):
    """Estimate dimensionless beta from end-to-end complex field overlap.

    Uses dimensionless propagation length

        zeta = 2 z / (k_ref d^2)

    with k_ref = 2 pi n_ref / wavelength.

    This is the natural propagation β estimate for a nearly stationary mode.
    """
    xp = grid.xp
    dxdy = float(grid.dx_um) * float(grid.dy_um)

    ov = xp.sum(xp.conj(A0) * A1) * dxdy
    p0 = xp.sum(xp.abs(A0) ** 2) * dxdy
    p1 = xp.sum(xp.abs(A1) ** 2) * dxdy

    ovn = ov / xp.sqrt(p0 * p1 + 1e-300)
    phase = xp.angle(ovn)

    z_um = float(grid.Nz) * float(grid.dz_um)
    k_ref = 2.0 * math.pi * float(n_ref) / float(wavelength_um)
    zeta = 2.0 * z_um / (k_ref * float(d_um) ** 2)

    return {
        "beta": float(asnumpy(phase / zeta)),
        "beta_phase_rad": float(asnumpy(phase)),
        "beta_overlap_abs": float(asnumpy(xp.abs(ovn))),
        "zeta": float(zeta),
    }

def run_static_zstack(request: StaticRequest, *, initial_theta: Any | None = None, controls: StaticZStackControls | None = None) -> StaticResult:
    request.validate()
    controls = StaticZStackControls() if controls is None else controls

    backend = get_backend(request.backend)
    xp = backend.xp
    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)
    tridiag_solver = _select_tridiag_solver(request)

    b = resolved_b(request.lc)
    bi = resolved_bi(request.lc, request.beams)
    gamma_z = float(getattr(request.lc, "theta_z_gamma", 0.0))
    inv_dz2 = 1.0 / (float(grid.dz_um) * float(grid.dz_um))

    theta_stack = _prepare_theta_stack(initial_theta, bias, grid, request.lc.cell.theta_bc)

    n_ref = float(asnumpy(neff_from_theta(theta_stack[0], ne=request.lc.material.ne, no=request.lc.material.no, xp=xp)).mean())
    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))

    phi_est = (2.0 * math.pi / wavelength_um) * float(grid.dz_um) * float(controls.dn_max_est)
    Nsub = max(1, min(int(controls.max_substeps), int(math.ceil(phi_est / float(controls.dz_opt_max_phi)))))
    dz_sub = float(grid.dz_um) / Nsub
    h_sub = linear_kernel(grid.fxy2_um, dz=dz_sub, wavelength=wavelength_um, n_ref=n_ref, xp=xp)

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=float(controls.static_dt), mobility=request.lc.mobility,
        dx=grid.du, dy=grid.dv, Ny=grid.Ny, xp=xp, dtype=backend.real_dtype,
    )

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
        slice_rms, slice_max, slice_ok, slice_niter, slice_selfcons = [], [], [], [], []

        for k in range(int(grid.Nz)):
            tp, tn = _neighbors(theta_old, k)
            th, I_mid, A, info = strict_selfconsistent_slice(
                amp_in=A, theta_seed=theta_old[k], theta_prev=tp, theta_next=tn,
                b=b, bi=bi, gamma_z=gamma_z, inv_dz2=inv_dz2,
                mobility=request.lc.mobility, dx=grid.du, dy=grid.dv,
                dt=float(controls.static_dt), s=s_cn, off=off_cn, diag=diag_cn,
                clamp=bias.theta_clamp, theta_bc=request.lc.cell.theta_bc,
                tridiag_solver=tridiag_solver, h_sub=h_sub, dz=float(grid.dz_um),
                wavelength=wavelength_um, n_ref=n_ref, ne=request.lc.material.ne,
                no=request.lc.material.no, Nsub=Nsub,
                coherent=(launch.coherence == "coherent"), theta_weights=launch.theta_weights,
                xp=xp, max_outer_passes=int(controls.max_strict_outer_passes),
                max_selfcons_passes=int(controls.max_selfcons_passes),
                selfcons_tol_theta=float(controls.selfcons_tol_theta),
                selfcons_tol_I=float(controls.selfcons_tol_I),
                residual_tol_rms=float(request.tol_residual_rms),
                residual_tol_max=float(request.tol_residual_max),
                static_max_steps=int(request.static_max_steps),
                static_resid_every=int(request.static_resid_every),
                static_relax_omega=float(request.static_relax_omega),
                picard_iters=int(controls.theta_picard_iters),
                picard_tol=float(controls.theta_picard_tol),
            )
            theta_new[k] = th
            I_stack[k] = I_mid.astype(grid.real_dtype, copy=False)
            slice_rms.append(float(info.get("rms_interior", math.inf)))
            slice_max.append(float(info.get("max_interior", math.inf)))
            slice_ok.append(bool(info.get("converged", False)))
            slice_niter.append(int(info.get("niter", -1)))
            slice_selfcons.append(int(info.get("n_selfcons_passes", -1)))

        theta_stack = theta_new
        theta_stack[:, 0, :] = request.lc.cell.theta_bc
        theta_stack[:, -1, :] = request.lc.cell.theta_bc

        A_last = A
        final_I = total_intensity(A_last, coherent=(launch.coherence == "coherent"), xp=xp)
        update = _stack_update_metrics(theta_stack, theta_old, xp)
        residual = _stack_residual_metrics(theta_stack, I_stack, b=b, bi=bi, gamma_z=gamma_z, inv_dz2=inv_dz2, grid=grid)

        info = {
            "outer": int(outer + 1), **update, **residual,
            "slice_residual_rms_max": max(slice_rms) if slice_rms else math.inf,
            "slice_residual_max_max": max(slice_max) if slice_max else math.inf,
            "slice_converged_count": int(sum(slice_ok)),
            "slice_niter_max": max(slice_niter) if slice_niter else -1,
            "slice_selfcons_passes_max": max(slice_selfcons) if slice_selfcons else -1,
            "update_converged": bool(update["dtheta_rms"] < request.tol_rms and update["dtheta_max"] < request.tol_max),
            "residual_converged": bool(residual["residual_rms"] < request.tol_residual_rms and residual["residual_max"] < request.tol_residual_max),
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

    metrics.update(_beta_from_overlap(
        launch.A0,
        A_last,
        grid=grid,
        wavelength_um=wavelength_um,
        n_ref=n_ref,
        d_um=request.lc.cell.thickness_um,
    ))
    metrics.update(_stack_residual_metrics(theta_stack, I_stack, b=b, bi=bi, gamma_z=gamma_z, inv_dz2=inv_dz2, grid=grid))
    if history:
        for key in ("dtheta_rms", "dtheta_max", "update_converged", "residual_converged", "slice_converged_count", "slice_residual_rms_max", "slice_residual_max_max", "slice_niter_max", "slice_selfcons_passes_max"):
            metrics[key] = history[-1][key]
    metrics.update({
        "backend": backend.name, "precision": request.backend.precision, "tridiag": request.tridiag,
        "Nx": grid.Nx, "Ny": grid.Ny, "Nz": grid.Nz,
        "outer_steps": len(history), "static_inner_steps": int(request.static_inner_steps),
        "static_max_steps": int(request.static_max_steps), "static_resid_every": int(request.static_resid_every),
        "static_relax_omega": float(request.static_relax_omega), "converged": bool(converged),
        "used_initial_theta": bool(initial_theta is not None), "b": float(b), "bi": float(bi),
        "theta_z_gamma": float(gamma_z), "inv_dz2": float(inv_dz2), "n_ref": float(n_ref),
        "Nsub": int(Nsub), "elapsed_s": float(elapsed), "theta_max": float(asnumpy(xp.max(theta_stack))),
        "theta_shape": tuple(int(x) for x in theta_stack.shape),
    })
    return StaticResult(kind="StaticZStackResult", metrics=metrics, samples=samples, theta=theta_stack, intensity=I_stack, history=history, converged=converged)


__all__ = ["StaticZStackControls", "run_static_zstack"]
