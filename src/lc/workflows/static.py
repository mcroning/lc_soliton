"""Static/self-consistent LC workflow.

v001 is deliberately modest: it prepares the same human-facing LC/beam/grid
inputs as the TD workflow, then runs a static outer loop using the trusted
``algorithms.static_relax`` black box.

The static theta relaxer used here is a pseudo-time CN/Picard step with fixed
optical intensity.  The optics update is one full z propagation pass through
the current theta field, returning the final intensity.  This gives us a first
clean static workflow to test the architecture; eigensoliton continuation and
beta normalization come later.
"""

from __future__ import annotations

import math
import time as _time
from typing import Any

from ..request import StaticRequest, RunSummary
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.liquid_crystal import resolved_b, neff_from_theta
from ..physics.bias import build_bias
from ..physics.launch import build_launch
from ..algorithms.splitstep import (
    advance_slice,
    linear_kernel,
    total_intensity,
)
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.theta_picard import cn_trapezoid_picard_step
from ..algorithms.thomas import solve_const_offdiag_batched
from ..algorithms.static_relax import StaticRelaxControls, run_static_relax
from ..products.diagnostics import (
    intensity_metrics,
    theta_update_metrics,
)


def _select_tridiag_solver(request: StaticRequest):
    if request.tridiag == "fast":
        from ..algorithms.thomas_fast import solve_const_offdiag_batched_fast

        return solve_const_offdiag_batched_fast
    return solve_const_offdiag_batched



def run_static(request: StaticRequest) -> RunSummary:
    """Run a static/self-consistent LC request."""

    request.validate()

    backend = get_backend(request.backend)
    xp = backend.xp

    grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(request.lc, grid)
    launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype)

    tridiag_solver = _select_tridiag_solver(request)

    b = resolved_b(request.lc)
    n_ref = float(asnumpy(neff_from_theta(
        bias.theta_2d,
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

    # Use a pseudo-time CN/Picard step as the static theta relaxer.
    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=7.5e-4,
        mobility=request.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    theta0 = bias.theta_2d
    A0 = launch.A0
    intensity0 = total_intensity(A0, coherent=(launch.coherence == "coherent"), xp=xp)

    samples: list[dict] = []

    def theta_relax(theta, intensity, outer):
        out = cn_trapezoid_picard_step(
            theta,
            intensity,
            intensity,
            b=b,
            bi=214.28571428571428,
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
        return Awork, I

    def convergence(theta, theta_prev, info):
        m = theta_update_metrics(theta, theta_prev)
        info.update(m)
        return (m["dtheta_rms"] < request.tol_rms) and (m["dtheta_max"] < request.tol_max)

    def observer(payload):
        info = dict(payload["info"])
        mm = intensity_metrics(payload["intensity"], grid)
        mm.update(info)
        mm["theta_max"] = float(asnumpy(xp.max(payload["theta"])))
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
    metrics.update({
        "backend": backend.name,
        "precision": request.backend.precision,
        "tridiag": request.tridiag,
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Nz": grid.Nz,
        "outer_steps": int(result.outer_steps),
        "converged": bool(result.converged),
        "b": float(b),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(result.theta))),
    })
    if result.history:
        last = result.history[-1]
        if "dtheta_rms" in last:
            metrics["dtheta_rms"] = float(last["dtheta_rms"])
        if "dtheta_max" in last:
            metrics["dtheta_max"] = float(last["dtheta_max"])

    return RunSummary(kind="StaticRunSummary", metrics=metrics, samples=samples)


__all__ = ["run_static"]
