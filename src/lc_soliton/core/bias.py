from __future__ import annotations

import numpy as np


def compute_neff(theta, *, ne: float, no: float, xp):
    ct = xp.cos(theta)
    st = xp.sin(theta)
    return (ne * no) / xp.sqrt((ne * ct) ** 2 + (no * st) ** 2)


def lc_b_from_voltage(V, K=7e-12, De=13.0):
    eps0 = 8.8541878128e-12
    return float(eps0 * float(De) * float(V) ** 2 / (8.0 * float(K)))


def voltage_from_lc_b(b, K=7e-12, De=13.0):
    eps0 = 8.8541878128e-12
    return float((8.0 * float(K) * float(b) / (eps0 * float(De))) ** 0.5)


def theta0_from_b_zero_bc(b):
    from scipy import special as spspec

    b = float(b)
    bc = np.pi**2 / 8.0
    if b <= bc:
        return 0.0

    lo, hi = 1e-14, 1.0 - 1e-14
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if spspec.ellipk(m) ** 2 - 2.0 * b > 0:
            hi = m
        else:
            lo = m
    return float(np.arcsin(np.sqrt(0.5 * (lo + hi))))


def b_from_theta0_zero_bc(theta0):
    from scipy import special as spspec

    th = abs(float(theta0))
    if th <= 0:
        return float(np.pi**2 / 8.0)
    return float(0.5 * spspec.ellipk(np.sin(th) ** 2) ** 2)


def theta_bias_1d_exact_zero_bc(Nx, b, xp):
    from scipy import special as spspec

    b = float(b)
    if b <= np.pi**2 / 8.0:
        return xp.zeros(int(Nx), dtype=xp.float32)

    theta0 = theta0_from_b_zero_bc(b)
    m = np.sin(theta0) ** 2
    u_cpu = np.linspace(-1.0, 1.0, int(Nx), dtype=np.float64)
    _, cn, dn, _ = spspec.ellipj(np.sqrt(2.0 * b) * u_cpu, m)
    s = np.sin(theta0) * cn / dn
    theta_cpu = np.arcsin(np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)).astype(np.float32)
    theta_cpu[0] = 0.0
    theta_cpu[-1] = 0.0
    return xp.asarray(theta_cpu, dtype=xp.float32)


def theta_bias_1d_relax_bc(Nx, b, theta_bc, xp, *, max_iter=200000, tol=1e-8):
    Nx = int(Nx)
    b = float(b)
    theta_bc = float(theta_bc)
    u = np.linspace(-1.0, 1.0, Nx, dtype=np.float64)
    du = float(u[1] - u[0])

    if abs(theta_bc) < 1e-14:
        theta = np.asarray(theta_bias_1d_exact_zero_bc(Nx, b, np), dtype=np.float64)
    else:
        theta = theta_bc + max(0.0, theta0_from_b_zero_bc(b)) * np.cos(0.5 * np.pi * u)
        theta[0] = theta_bc
        theta[-1] = theta_bc

    dt = 0.2 * du * du
    for _ in range(int(max_iter)):
        old = theta.copy()
        lap = np.zeros_like(theta)
        lap[1:-1] = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / (du * du)
        theta[1:-1] += dt * (lap[1:-1] + b * np.sin(2.0 * theta[1:-1]))
        theta[0] = theta_bc
        theta[-1] = theta_bc
        if np.sqrt(np.mean((theta - old) ** 2)) < tol:
            break

    return xp.asarray(theta.astype(np.float32), dtype=xp.float32)


def theta_bias_2d_from_params(params: LCParams, xp):
    if abs(float(params.theta_bc)) < 1e-14:
        th1 = theta_bias_1d_exact_zero_bc(params.Nx, params.b, xp)
    else:
        th1 = theta_bias_1d_relax_bc(params.Nx, params.b, params.theta_bc, xp)
    th2 = xp.tile(th1[:, None], (1, int(params.Ny))).astype(xp.float32, copy=False)
    th2[0, :] = xp.float32(params.theta_bc)
    th2[-1, :] = xp.float32(params.theta_bc)
    return th2


__all__ = [
    "compute_neff",
    "lc_b_from_voltage",
    "voltage_from_lc_b",
    "theta0_from_b_zero_bc",
    "b_from_theta0_zero_bc",
    "theta_bias_1d_exact_zero_bc",
    "theta_bias_1d_relax_bc",
    "theta_bias_2d_from_params",
]