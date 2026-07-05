"""Strict static slice self-consistency helpers.

Clean translation of the legacy strict static slice logic:

    optics through one dz slice
    midpoint intensity
    strict theta relaxation with optional z-neighbor coupling
    repeat slice self-consistency

These are workflow helpers.  The numerical theta step is supplied by
``lc.algorithms.theta_cn_zcoupled``.
"""

from __future__ import annotations

from typing import Any
import math

from ..numerics.backend import asnumpy
from ..algorithms.splitstep import advance_slice_with_midintensity, total_intensity
from ..algorithms.theta_cn_zcoupled import cn_trapezoid_picard_step_zcoupled


def residual_stats_zcoupled_slice(
    theta,
    intensity,
    *,
    theta_prev,
    theta_next,
    b: float,
    bi: float,
    gamma_z: float,
    inv_dz2: float,
    dx: float,
    dy: float,
    xp: Any,
) -> dict[str, float]:
    """Residual stats for one z-coupled static theta slice."""
    lap = xp.zeros_like(theta)
    yp = xp.roll(theta, -1, axis=1)
    ym = xp.roll(theta, +1, axis=1)

    lap[1:-1, :] = (
        (theta[2:, :] - 2.0 * theta[1:-1, :] + theta[:-2, :]) / (float(dx) * float(dx))
        + (yp[1:-1, :] - 2.0 * theta[1:-1, :] + ym[1:-1, :]) / (float(dy) * float(dy))
    )

    alpha = float(b) + float(bi) * intensity
    zterm = float(gamma_z) * float(inv_dz2) * (theta_prev - 2.0 * theta + theta_next)
    R = lap + zterm + alpha * xp.sin(2.0 * theta)
    Ri = R[1:-1, :]

    return {
        "rms_interior": float(asnumpy(xp.sqrt(xp.mean(Ri * Ri)))),
        "max_interior": float(asnumpy(xp.max(xp.abs(Ri)))),
    }


def strict_relax_slice_zcoupled(
    theta_seed,
    intensity,
    *,
    theta_prev,
    theta_next,
    b: float,
    bi: float,
    gamma_z: float,
    inv_dz2: float,
    mobility: float,
    dx: float,
    dy: float,
    dt: float,
    s,
    off,
    diag,
    clamp,
    theta_bc: float,
    tridiag_solver,
    xp: Any,
    max_steps: int = 200,
    resid_every: int = 10,
    residual_tol_rms: float = 1e-2,
    residual_tol_max: float = 5e-2,
    omega: float = 0.3,
    picard_iters: int = 4,
    picard_tol: float = 1e-6,
):
    """Relax one theta slice to residual tolerance."""
    th = theta_seed.copy()
    th[0, :] = theta_bc
    th[-1, :] = theta_bc
    if clamp is not None:
        th = xp.clip(th, clamp[0], clamp[1])

    last = None
    converged = False

    for it in range(int(max_steps)):
        th_new = cn_trapezoid_picard_step_zcoupled(
            th,
            intensity,
            dt=dt,
            b=b,
            bi=bi,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=mobility,
            dx=dx,
            dy=dy,
            s=s,
            off=off,
            diag=diag,
            inv_dz2=inv_dz2,
            max_iter=picard_iters,
            tol_update=picard_tol,
            clamp=clamp,
            tridiag_solver=tridiag_solver,
            xp=xp,
        )

        th = (1.0 - float(omega)) * th + float(omega) * th_new
        th[0, :] = theta_bc
        th[-1, :] = theta_bc
        if clamp is not None:
            th = xp.clip(th, clamp[0], clamp[1])

        if (it % int(resid_every)) == 0 or it == int(max_steps) - 1:
            last = residual_stats_zcoupled_slice(
                th,
                intensity,
                theta_prev=theta_prev,
                theta_next=theta_next,
                b=b,
                bi=bi,
                gamma_z=gamma_z,
                inv_dz2=inv_dz2,
                dx=dx,
                dy=dy,
                xp=xp,
            )
            if (
                last["rms_interior"] <= float(residual_tol_rms)
                and last["max_interior"] <= float(residual_tol_max)
            ):
                converged = True
                break

    if last is None:
        last = residual_stats_zcoupled_slice(
            th,
            intensity,
            theta_prev=theta_prev,
            theta_next=theta_next,
            b=b,
            bi=bi,
            gamma_z=gamma_z,
            inv_dz2=inv_dz2,
            dx=dx,
            dy=dy,
            xp=xp,
        )

    return th, {
        "relax_converged": bool(converged),
        "niter": int(it + 1),
        "max_steps": int(max_steps),
        "relax_rms": float(last["rms_interior"]),
        "rms_interior": float(last["rms_interior"]),
        "max_interior": float(last["max_interior"]),
    }


def strict_selfconsistent_slice(
    *,
    amp_in,
    theta_seed,
    theta_prev,
    theta_next,
    b: float,
    bi: float,
    gamma_z: float,
    inv_dz2: float,
    mobility: float,
    dx: float,
    dy: float,
    dt: float,
    s,
    off,
    diag,
    clamp,
    theta_bc: float,
    tridiag_solver,
    h_sub,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int,
    coherent: bool,
    theta_weights,
    xp: Any,
    max_outer_passes: int = 8,
    max_selfcons_passes: int = 3,
    selfcons_tol_theta: float = 1e-4,
    selfcons_tol_I: float = 1e-4,
    residual_tol_rms: float = 1e-2,
    residual_tol_max: float = 5e-2,
    static_max_steps: int = 200,
    static_resid_every: int = 10,
    static_relax_omega: float = 0.3,
    picard_iters: int = 4,
    picard_tol: float = 1e-6,
):
    """Self-consistent strict solve for one propagation slice."""
    theta = theta_seed.copy()
    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc
    if clamp is not None:
        theta = xp.clip(theta, clamp[0], clamp[1])

    last_I_mid = None
    last_info = None
    converged = False
    amp_work = amp_in.copy()
    I_mid = total_intensity(amp_in, coherent=coherent, xp=xp)
    dtheta = math.inf
    dI = math.inf

    for sc_pass in range(int(max_selfcons_passes)):
        amp_work = amp_in.copy()
        amp_work, _I_before, _I_after, I_mid = advance_slice_with_midintensity(
            amp_work,
            theta,
            kernel=h_sub,
            dz=dz,
            wavelength=wavelength,
            n_ref=n_ref,
            ne=ne,
            no=no,
            Nsub=Nsub,
            coherent=coherent,
            theta_weights=theta_weights,
            xp=xp,
        )

        theta_new = theta
        info = {}
        #for _outer in range(int(max_outer_passes)):
        theta_new, info = strict_relax_slice_zcoupled(
            theta_new,
            I_mid,
            theta_prev=theta_prev,
            theta_next=theta_next,
            b=b,
            bi=bi,
            gamma_z=gamma_z,
            inv_dz2=inv_dz2,
            mobility=mobility,
            dx=dx,
            dy=dy,
            dt=dt,
            s=s,
            off=off,
            diag=diag,
            clamp=clamp,
            theta_bc=theta_bc,
            tridiag_solver=tridiag_solver,
            xp=xp,
            max_steps=static_max_steps,
            resid_every=static_resid_every,
            residual_tol_rms=residual_tol_rms,
            residual_tol_max=residual_tol_max,
            omega=static_relax_omega,
            picard_iters=picard_iters,
            picard_tol=picard_tol,
        )
            #if info["rms_interior"] <= residual_tol_rms and info["max_interior"] <= residual_tol_max:
            #    break

        dtheta = float(asnumpy(xp.max(xp.abs(theta_new - theta))))
        if last_I_mid is None:
            dI = math.inf
        else:
            denom = max(float(asnumpy(xp.max(xp.abs(last_I_mid)))), 1e-12)
            dI = float(asnumpy(xp.max(xp.abs(I_mid - last_I_mid)))) / denom

        theta = theta_new
        last_I_mid = I_mid.copy()
        last_info = info

        if (
            dtheta <= float(selfcons_tol_theta)
            and dI <= float(selfcons_tol_I)
            and info["rms_interior"] <= float(residual_tol_rms)
            and info["max_interior"] <= float(residual_tol_max)
        ):
            converged = True
            break

    last_info = {} if last_info is None else dict(last_info)
    last_info.update(
        {
            "converged": bool(
                converged
                and last_info.get("rms_interior", math.inf) <= float(residual_tol_rms)
                and last_info.get("max_interior", math.inf) <= float(residual_tol_max)
            ),
            "n_selfcons_passes": int(sc_pass + 1),
            "selfcons_dtheta": float(dtheta),
            "selfcons_dI": float(dI),
        }
    )
    return theta, I_mid, amp_work, last_info


__all__ = [
    "residual_stats_zcoupled_slice",
    "strict_relax_slice_zcoupled",
    "strict_selfconsistent_slice",
]
