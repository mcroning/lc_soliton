# lc_core/residuals.py
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ResidualStats:
    max_full: float
    rms_full: float
    max_interior: float
    rms_interior: float


def _xp_of(arr):
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy as cp
        return cp
    import numpy as np
    return np


def _as_float(x):
    return float(x.get() if hasattr(x, "get") else x)


# =========================================================
# XY operators: Dirichlet in x, periodic in y
# =========================================================

def laplacian_dirichletx_periody(theta, du, dv, *, dtype="float32"):
    """
    2D Laplacian with Dirichlet x rows excluded and periodic y.

    This is the cleaned version of the legacy:
        _laplacian_dirichletx_periody(theta, du, dv)

    Boundary x rows are returned as zero.
    """
    xp = _xp_of(theta)

    if dtype == "float64":
        th = theta.astype(xp.float64, copy=False)
        du2 = xp.float64(du) * xp.float64(du)
        dv2 = xp.float64(dv) * xp.float64(dv)
    else:
        th = theta.astype(xp.float32, copy=False)
        du2 = xp.float32(du * du)
        dv2 = xp.float32(dv * dv)

    th_yplus = xp.roll(th, -1, axis=1)
    th_yminus = xp.roll(th, +1, axis=1)

    lap = xp.zeros_like(th)

    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        +
        (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    return lap


# legacy-compatible alias
_laplacian_dirichletx_periody = laplacian_dirichletx_periody


# =========================================================
# 2D residual
# =========================================================

def lc_residual64(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    """
    LC steady-state residual for one z-slice:

        R = [L_xy theta + (b + bi I) sin(2 theta)] / mobility

    x boundary rows are set to zero because they are prescribed Dirichlet rows.

    This preserves the legacy lc_residual64 formula.
    """
    xp = _xp_of(theta)

    th = theta.astype(xp.float64, copy=False)
    I64 = Ixy.astype(xp.float64, copy=False)

    lap = laplacian_dirichletx_periody(th, du, dv, dtype="float64")

    drive = (xp.float64(b) + xp.float64(bi) * I64) * xp.sin(2.0 * th)

    R = (lap + drive) / xp.float64(mobility)

    R[0, :] = 0.0
    R[-1, :] = 0.0

    return R


def lc_residual2d_dirichletx_periody(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    return lc_residual64(theta, Ixy, b=b, bi=bi, du=du, dv=dv, mobility=mobility)


def lc_residual2d_ctx(ctx, theta, Ixy):
    """
    Convenience wrapper using ctx fields.
    """
    return lc_residual64(
        theta,
        Ixy,
        b=ctx.b,
        bi=ctx.bi,
        du=ctx.du,
        dv=ctx.dv,
        mobility=getattr(ctx, "mobility", 1.0),
    )


# =========================================================
# 3D z-coupled residual
# =========================================================

def lc_residual3d64(theta_full, I_full, *, b, bi, du, dv, dz, gamma_z=0.0, mobility=1.0):
    """
    LC steady-state residual for a full z stack:

        R_k = [L_xy theta_k
               + gamma_z (theta_{k+1} - 2 theta_k + theta_{k-1}) / dz^2
               + (b + bi I_k) sin(2 theta_k)] / mobility

    x boundary rows are zeroed.

    z boundary convention:
        Neumann-like copy at the ends:
            theta_{-1} -> theta_0
            theta_{Nz} -> theta_{Nz-1}

    This is a safe global diagnostic. For local strict-march diagnostics with
    supplied upstream/downstream neighbors, use lc_residual_slice3d64().
    """
    xp = _xp_of(theta_full)

    th = theta_full.astype(xp.float64, copy=False)
    I64 = I_full.astype(xp.float64, copy=False)

    du = xp.float64(du)
    dv = xp.float64(dv)
    dz = xp.float64(dz)

    du2 = du * du
    dv2 = dv * dv
    dz2 = dz * dz

    # L_xy with Dirichlet x and periodic y, vectorized over z.
    th_yplus = xp.roll(th, -1, axis=2)
    th_yminus = xp.roll(th, +1, axis=2)

    lap_xy = xp.zeros_like(th)
    lap_xy[:, 1:-1, :] = (
        (th[:, 2:, :] - 2.0 * th[:, 1:-1, :] + th[:, :-2, :]) / du2
        +
        (th_yplus[:, 1:-1, :] - 2.0 * th[:, 1:-1, :] + th_yminus[:, 1:-1, :]) / dv2
    )

    if float(gamma_z) == 0.0:
        lap_z = xp.zeros_like(th)
    else:
        tp = xp.empty_like(th)
        tn = xp.empty_like(th)

        tp[1:, :, :] = th[:-1, :, :]
        tp[0, :, :] = th[0, :, :]

        tn[:-1, :, :] = th[1:, :, :]
        tn[-1, :, :] = th[-1, :, :]

        lap_z = xp.float64(gamma_z) * (tn - 2.0 * th + tp) / dz2
        lap_z[:, 0, :] = 0.0
        lap_z[:, -1, :] = 0.0

    drive = (xp.float64(b) + xp.float64(bi) * I64) * xp.sin(2.0 * th)

    R = (lap_xy + lap_z + drive) / xp.float64(mobility)

    R[:, 0, :] = 0.0
    R[:, -1, :] = 0.0

    return R


def lc_residual3d_ctx(ctx, theta_full, I_full):
    """
    Convenience wrapper using ctx fields.
    """
    return lc_residual3d64(
        theta_full,
        I_full,
        b=ctx.b,
        bi=ctx.bi,
        du=ctx.du,
        dv=ctx.dv,
        dz=ctx.dz,
        gamma_z=getattr(ctx, "theta_z_gamma", 0.0),
        mobility=getattr(ctx, "mobility", 1.0),
    )


def lc_residual_slice3d64(theta_k, I_k, theta_prev, theta_next, *, b, bi, du, dv, dz, gamma_z=0.0, mobility=1.0):
    """
    Local z-coupled slice residual using explicit previous/next theta slices.

    This is the cleaned value-returning version of legacy:
        _slice_residual_rms_local3d_z(...)

    Unlike lc_residual3d64, this accepts externally supplied neighbors. That is
    useful inside strict z-marching where upstream/downstream neighbors may be
    predicted or lagged.
    """
    xp = _xp_of(theta_k)

    th = theta_k.astype(xp.float64, copy=False)
    I64 = I_k.astype(xp.float64, copy=False)
    tp = theta_prev.astype(xp.float64, copy=False)
    tn = theta_next.astype(xp.float64, copy=False)

    lap_xy = laplacian_dirichletx_periody(th, du, dv, dtype="float64")

    if float(gamma_z) == 0.0:
        lap_z = xp.zeros_like(th)
    else:
        lap_z = xp.float64(gamma_z) * (tn - 2.0 * th + tp) / (xp.float64(dz) * xp.float64(dz))
        lap_z[0, :] = 0.0
        lap_z[-1, :] = 0.0

    drive = (xp.float64(b) + xp.float64(bi) * I64) * xp.sin(2.0 * th)

    R = (lap_xy + lap_z + drive) / xp.float64(mobility)

    R[0, :] = 0.0
    R[-1, :] = 0.0

    return R


def lc_residual_slice3d_ctx(ctx, theta_k, I_k, theta_prev, theta_next):
    return lc_residual_slice3d64(
        theta_k,
        I_k,
        theta_prev,
        theta_next,
        b=ctx.b,
        bi=ctx.bi,
        du=ctx.du,
        dv=ctx.dv,
        dz=ctx.dz,
        gamma_z=getattr(ctx, "theta_z_gamma", 0.0),
        mobility=getattr(ctx, "mobility", 1.0),
    )


# =========================================================
# Stats / reports
# =========================================================

def residual_stats_2d(R) -> dict:
    xp = _xp_of(R)
    R = R.astype(xp.float64, copy=False)
    Rint = R[1:-1, :]

    stats = ResidualStats(
        max_full=_as_float(xp.max(xp.abs(R))),
        rms_full=_as_float(xp.sqrt(xp.mean(R * R))),
        max_interior=_as_float(xp.max(xp.abs(Rint))),
        rms_interior=_as_float(xp.sqrt(xp.mean(Rint * Rint))),
    )
    return asdict(stats)


def residual_stats_3d(R) -> dict:
    xp = _xp_of(R)
    R = R.astype(xp.float64, copy=False)
    Rint = R[:, 1:-1, :]

    stats = ResidualStats(
        max_full=_as_float(xp.max(xp.abs(R))),
        rms_full=_as_float(xp.sqrt(xp.mean(R * R))),
        max_interior=_as_float(xp.max(xp.abs(Rint))),
        rms_interior=_as_float(xp.sqrt(xp.mean(Rint * Rint))),
    )
    return asdict(stats)


def residual_report_2d(ctx, theta, Ixy) -> dict:
    R = lc_residual2d_ctx(ctx, theta, Ixy)
    out = residual_stats_2d(R)
    out.update(
        dict(
            shape=list(theta.shape),
            b=float(ctx.b),
            bi=float(ctx.bi),
            du=float(ctx.du),
            dv=float(ctx.dv),
            mobility=float(getattr(ctx, "mobility", 1.0)),
        )
    )
    return out


def residual_report_3d(ctx, theta_full, I_full) -> dict:
    R = lc_residual3d_ctx(ctx, theta_full, I_full)
    out = residual_stats_3d(R)
    out.update(
        dict(
            shape=list(theta_full.shape),
            b=float(ctx.b),
            bi=float(ctx.bi),
            du=float(ctx.du),
            dv=float(ctx.dv),
            dz=float(ctx.dz),
            theta_z_gamma=float(getattr(ctx, "theta_z_gamma", 0.0)),
            mobility=float(getattr(ctx, "mobility", 1.0)),
        )
    )
    return out


def zero_intensity_like(theta):
    xp = _xp_of(theta)
    return xp.zeros_like(theta, dtype=xp.float32)
