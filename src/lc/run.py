"""Top-level run entry point for the clean LC package."""

from __future__ import annotations

import math
import time as _time
from typing import Any

from .request import TDRequest, RunSummary
from .numerics.backend import get_backend, asnumpy, synchronize
from .numerics.grid import make_grid
from .physics.liquid_crystal import resolved_b, neff_from_theta
from .physics.bias import build_bias
from .physics.launch import build_launch
from .algorithms.splitstep import advance_slice_with_midintensity, hop_linear_inplace, linear_kernel, total_intensity
from .algorithms.theta_cn import prepare_cn_operator
from .algorithms.theta_picard import cn_trapezoid_picard_step
from .algorithms.thomas import solve_const_offdiag_batched
from .algorithms.td_zmarch import TDZMarchControls, run_td_zmarch


def _moment_metrics(I: Any, grid: Any) -> dict[str, float]:
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    P = raw * float(grid.dx_um) * float(grid.dy_um)
    x = grid.x_um
    y = grid.y_um
    xc = xp.sum(I * x[:, None]) / raw
    yc = xp.sum(I * y[None, :]) / raw
    sx = xp.sqrt(xp.sum(I * (x[:, None] - xc) ** 2) / raw)
    sy = xp.sqrt(xp.sum(I * (y[None, :] - yc) ** 2) / raw)
    return {
        "power": float(asnumpy(P)),
        "Imax": float(asnumpy(xp.max(I))),
        "sx_um": float(asnumpy(sx)),
        "sy_um": float(asnumpy(sy)),
        "xc_um": float(asnumpy(xc)),
        "yc_um": float(asnumpy(yc)),
    }


def run(request: TDRequest) -> RunSummary:
    """Run a TDRequest and return a compact summary."""

    request.validate()

    backend = get_backend(request.backend)
    xp = backend.xp
    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)

    if request.tridiag == "fast":
        from .algorithms.thomas_fast import solve_const_offdiag_batched_fast as tridiag_solver
    else:
        tridiag_solver = solve_const_offdiag_batched

    b = resolved_b(request.lc)
    n_ref = float(asnumpy(neff_from_theta(
        bias.theta_2d,
        ne=request.lc.material.ne,
        no=request.lc.material.no,
        xp=xp,
    )).mean())

    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))
    k0 = 2.0 * math.pi / wavelength_um
    dn_max_est = 0.02
    dz_opt_max_phi = 0.30
    max_substeps = 16
    Nsub = max(1, min(max_substeps, int(math.ceil(k0 * grid.dz_um * dn_max_est / dz_opt_max_phi))))
    dz_sub = grid.dz_um / Nsub

    h_sub = linear_kernel(grid.fxy2_um, dz=dz_sub, wavelength=wavelength_um, n_ref=n_ref, xp=xp)
    h_half = linear_kernel(grid.fxy2_um, dz=0.5 * grid.dz_um, wavelength=wavelength_um, n_ref=n_ref, xp=xp)

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=request.time.dt,
        mobility=request.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    def optics_half_step(A):
        return hop_linear_inplace(A, h_half, xp=xp)

    def optics_step(A, theta_k, k):
        A, _, _, I_mid = advance_slice_with_midintensity(
            A,
            theta_k,
            kernel=h_sub,
            dz=grid.dz_um,
            wavelength=wavelength_um,
            n_ref=n_ref,
            ne=request.lc.material.ne,
            no=request.lc.material.no,
            Nsub=Nsub,
            coherent=(launch.coherence == "coherent"),
            theta_weights=launch.theta_weights,
            xp=xp,
        )
        return A, I_mid

    def theta_step(theta_k, I_mid, theta_prev, theta_next, k):
        out = cn_trapezoid_picard_step(
            theta_k,
            I_mid,
            I_mid,
            b=b,
            bi=214.28571428571428,
            dt=request.time.dt,
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

    samples: list[dict] = []
    sample_every = max(1, grid.Nz // 4)

    def observer(payload):
        if payload["jt"] == request.time.Nt and payload["k"] % sample_every == 0:
            mm = _moment_metrics(payload["I_mid"], grid)
            mm.update({"jt": int(payload["jt"]), "k": int(payload["k"]), "z_um": float((payload["k"] + 1) * grid.dz_um)})
            samples.append(mm)

    synchronize(xp)
    t0 = _time.perf_counter()
    result = run_td_zmarch(
        bias.theta_stack,
        launch.A0,
        optics_step=optics_step,
        theta_step=theta_step,
        controls=TDZMarchControls(Nt=request.time.Nt, observer_stride_z=sample_every, observer_stride_t=1),
        optics_half_step=optics_half_step,
        observer=observer,
    )
    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    final_I = total_intensity(result.A_last, coherent=(launch.coherence == "coherent"), xp=xp)
    metrics = _moment_metrics(final_I, grid)
    metrics.update({
        "backend": backend.name,
        "precision": request.backend.precision,
        "tridiag": request.tridiag,
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Nz": grid.Nz,
        "Nt": request.time.Nt,
        "b": float(b),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(result.theta))),
    })

    return RunSummary(kind="TDRunSummary", metrics=metrics, samples=samples)


__all__ = ["run"]
