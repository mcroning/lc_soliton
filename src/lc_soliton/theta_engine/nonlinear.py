"""
theta_engine/nonlinear.py

Nonlinear Crank-Nicolson solvers for theta.

No hard-coded precision. The dtype/backend are determined by the input arrays.
"""

from __future__ import annotations

from .operators import laplacian_xy, director_drive, apply_theta_bc, clip_theta
from .residuals import cn_residual, residual_stats
from .cn import prepare_cn_operator, solve_cn_linear_system


def solve_theta_cn_picard(
    theta_old,
    intensity,
    *,
    dt,
    du,
    dv,
    b,
    bi,
    mobility=1.0,
    theta_bc=0.0,
    theta_clamp=None,
    max_iter=20,
    tol_update=1e-12,
    tol_cn=None,
    monitor_cn=False,
    save_history=False,
    xp,
):
    """
    Solve one nonlinear Crank-Nicolson theta step by Picard iteration.

    This solves:

        (theta_new - theta_old)/dt =
            0.5 * [F(theta_old) + F(theta_new)] / mobility

    where

        F(theta) = laplacian_xy(theta) + (b + bi I) sin(2 theta)

    Returns
    -------
    theta_new : array
    info : dict
        Includes Picard history and final CN residual.
    """

    dtype = theta_old.dtype
    theta_n = theta_old.copy()
    I = intensity.astype(dtype, copy=False)

    theta_bc = dtype.type(theta_bc) if hasattr(dtype, "type") else dtype(theta_bc)

    s, off, diag, _ = prepare_cn_operator(
        dt=dt,
        mobility=mobility,
        du=du,
        dv=dv,
        Ny=theta_n.shape[1],
        xp=xp,
        dtype=dtype,
    )

    lap_n = laplacian_xy(theta_n, du=du, dv=dv, xp=xp)
    N_n = director_drive(theta_n, I, b=b, bi=bi, xp=xp)

    dt_over_m = dtype.type(dt / mobility) if hasattr(dtype, "type") else dtype(dt / mobility)

    rhs_base = theta_n + s * lap_n + dtype.type(0.5) * dt_over_m * N_n
    apply_theta_bc(rhs_base, theta_bc)

    theta_g = theta_n.copy()
    history = [] if save_history else None

    converged_update = False
    converged_cn = False

    for it in range(1, int(max_iter) + 1):
        N_g = director_drive(theta_g, I, b=b, bi=bi, xp=xp)

        rhs = rhs_base + dtype.type(0.5) * dt_over_m * N_g
        apply_theta_bc(rhs, theta_bc)

        theta_new = solve_cn_linear_system(
            rhs,
            off=off,
            diag=diag,
            theta_bc=theta_bc,
            xp=xp,
        )

        clip_theta(theta_new, theta_clamp, xp=xp)
        apply_theta_bc(theta_new, theta_bc)

        dtheta = theta_new - theta_g
        di = dtheta[1:-1, :]

        update_rms = float(xp.sqrt(xp.mean(di * di)))
        update_max = float(xp.max(xp.abs(di)))

        if monitor_cn:
            Rcn = cn_residual(
                theta_new,
                theta_n,
                I,
                dt=dt,
                du=du,
                dv=dv,
                b=b,
                bi=bi,
                mobility=mobility,
                xp=xp,
            )
            st_cn = residual_stats(Rcn, xp=xp)

            cn_rms = st_cn["rms_interior"]
            cn_max = st_cn["max_interior"]
        else:
            cn_rms = None
            cn_max = None
        if save_history:
            history.append(
                {
                    "iter": it,
                    "update_rms": update_rms,
                    "update_max": update_max,
                    "cn_rms": cn_rms,
                    "cn_max": cn_max,
                }
            )

        theta_g = theta_new

        last_update_rms = update_rms
        last_update_max = update_max
        last_cn_rms = cn_rms
        last_cn_max = cn_max

        if update_rms < float(tol_update):
            converged_update = True

        if monitor_cn and tol_cn is not None:
            if cn_rms < float(tol_cn):
                converged_cn = True

        if converged_update:
            if not monitor_cn:
                break
            if tol_cn is None or converged_cn:
                break

    info = {
        "niter": it,
        "converged_update": converged_update,
        "converged_cn": converged_cn if monitor_cn else None,
        "update_rms": last_update_rms,
        "update_max": last_update_max,
        "cn_rms": last_cn_rms,
        "cn_max": last_cn_max,
        "history": history,
    }

    return theta_g, info
