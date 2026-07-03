#!/usr/bin/env python
"""Smoke tests for the clean src/lc/algorithms package.

This test is intentionally independent of the old lc_soliton package.  It
checks that the small numerical black boxes compile, import, and behave
sensibly on tiny NumPy problems.

It is not yet a runner_core equivalence test.  That comes next.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300)


def test_thomas():
    from lc.algorithms.thomas import solve_const_offdiag_batched

    batch = 5
    n = 8
    lower = -0.2
    upper = -0.2
    diag = np.linspace(1.5, 2.0, batch).astype(np.complex128)

    rng = np.random.default_rng(1)
    x_true = rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))

    rhs = np.empty_like(x_true)
    for i in range(batch):
        M = np.diag(np.full(n, diag[i], dtype=np.complex128))
        M += np.diag(np.full(n - 1, lower, dtype=np.complex128), -1)
        M += np.diag(np.full(n - 1, upper, dtype=np.complex128), +1)
        rhs[i] = M @ x_true[i]

    x = solve_const_offdiag_batched(lower, diag, upper, rhs)
    err = rel_l2(x, x_true)
    assert err < 1e-13, err
    return {"thomas_rel_l2": err}


def test_fft_y():
    from lc.algorithms.fft_y import fft_y, ifft_y, lam_y_periodic_second_diff

    rng = np.random.default_rng(2)
    a = rng.normal(size=(7, 9)).astype(np.float64)
    ah = fft_y(a)
    ar = ifft_y(ah).real

    err = rel_l2(ar, a)
    lam = lam_y_periodic_second_diff(9, 0.25, dtype=np.float64)

    assert err < 1e-14, err
    assert lam.shape == (9,)
    assert np.isclose(lam[0], 0.0)
    assert np.all(lam <= 1e-14)
    return {"fft_roundtrip_rel_l2": err, "lam_min": float(lam.min())}


def test_theta_cn_and_picard():
    from lc.algorithms.theta_cn import prepare_cn_operator, cn_predictor_step
    from lc.algorithms.theta_picard import cn_trapezoid_picard_step

    Nx, Ny = 32, 24
    dx = 2.0 / (Nx - 1)
    dy = dx * 1.2
    dt = 1e-4
    mobility = 1.0

    x = np.linspace(-1, 1, Nx)
    y = np.arange(Ny)
    theta = (0.15 * (1 - x[:, None] ** 2) * (1 + 0.05 * np.cos(2 * np.pi * y[None, :] / Ny))).astype(np.float64)
    theta[0, :] = 0.0
    theta[-1, :] = 0.0
    I = (0.02 * np.exp(-((x[:, None] / 0.3) ** 2))).astype(np.float64) * np.ones((1, Ny))

    s, off, diag, lam_y = prepare_cn_operator(dt=dt, mobility=mobility, dx=dx, dy=dy, Ny=Ny, dtype=np.float64)

    pred = cn_predictor_step(
        theta,
        I,
        b=1.2,
        bi=3.0,
        dt=dt,
        mobility=mobility,
        dx=dx,
        dy=dy,
        s=s,
        off=off,
        diag=diag,
        clamp=(0.0, np.pi / 2),
    )

    corr = cn_trapezoid_picard_step(
        theta,
        I,
        I,
        b=1.2,
        bi=3.0,
        dt=dt,
        mobility=mobility,
        dx=dx,
        dy=dy,
        s=s,
        off=off,
        diag=diag,
        max_iter=4,
        tol_update=1e-12,
        clamp=(0.0, np.pi / 2),
    )

    assert pred.shape == theta.shape
    assert corr.shape == theta.shape
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(corr))
    assert np.allclose(pred[0, :], 0.0)
    assert np.allclose(pred[-1, :], 0.0)
    assert np.allclose(corr[0, :], 0.0)
    assert np.allclose(corr[-1, :], 0.0)

    return {
        "theta_pred_min": float(pred.min()),
        "theta_pred_max": float(pred.max()),
        "theta_picard_min": float(corr.min()),
        "theta_picard_max": float(corr.max()),
        "theta_pred_picard_rel_l2": rel_l2(pred, corr),
    }


def test_splitstep():
    from lc.algorithms.splitstep import (
        advance_slice_with_midintensity,
        linear_kernel,
        total_intensity,
    )

    Nx, Ny = 32, 24
    dx, dy = 0.25, 0.3
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X = x[:, None]
    Y = y[None, :]

    A0 = np.empty((2, Nx, Ny), dtype=np.complex128)
    A0[0] = np.exp(-((X / 1.2) ** 2 + (Y / 1.3) ** 2))
    A0[1] = 0.5 * np.exp(-(((X - 0.7) / 1.1) ** 2 + ((Y + 0.3) / 1.0) ** 2)) * np.exp(0.3j)
    P0 = np.sum(np.abs(A0) ** 2)

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    fxy2 = fx[:, None] ** 2 + fy[None, :] ** 2
    kernel = linear_kernel(fxy2, dz=0.5, wavelength=0.633, n_ref=1.55)

    theta = 0.3 * np.exp(-((X / 3.0) ** 2 + (Y / 3.0) ** 2))
    A = A0.copy()
    A, I_b, I_a, I_mid = advance_slice_with_midintensity(
        A,
        theta,
        kernel=kernel,
        dz=0.5,
        wavelength=0.633,
        n_ref=1.55,
        ne=1.7,
        no=1.5,
        Nsub=2,
    )

    P1 = np.sum(np.abs(A) ** 2)
    assert A.shape == A0.shape
    assert I_b.shape == (Nx, Ny)
    assert I_a.shape == (Nx, Ny)
    assert I_mid.shape == (Nx, Ny)
    assert abs(P1 - P0) / P0 < 1e-12

    return {
        "splitstep_power_rel_err": float(abs(P1 - P0) / P0),
        "splitstep_I_before": float(I_b.max()),
        "splitstep_I_after": float(I_a.max()),
    }


def test_td_zmarch():
    from lc.algorithms.td_zmarch import TDZMarchControls, run_td_zmarch

    Nz, Nx, Ny = 4, 5, 6
    theta0 = np.zeros((Nz, Nx, Ny), dtype=np.float64)
    A0 = np.ones((1, Nx, Ny), dtype=np.complex128)

    def optics_step(A, theta_k, k):
        I_mid = np.abs(A[0]) ** 2 + k
        return A, I_mid

    def theta_step(theta_k, I_mid, tp, tn, k):
        return theta_k + 0.1 * I_mid + 0.01 * (tp + tn)

    res = run_td_zmarch(
        theta0,
        A0,
        optics_step=optics_step,
        theta_step=theta_step,
        controls=TDZMarchControls(Nt=3, observer_stride_z=2),
    )

    assert res.theta.shape == theta0.shape
    assert res.A_last.shape == A0.shape
    assert res.steps == 3
    assert np.all(np.isfinite(res.theta))
    return {"td_zmarch_theta_max": float(res.theta.max())}


def test_static_relax():
    from lc.algorithms.static_relax import StaticRelaxControls, run_static_relax

    Nx, Ny = 5, 6
    theta0 = np.zeros((Nx, Ny), dtype=np.float64)
    A0 = np.ones((1, Nx, Ny), dtype=np.complex128)
    I0 = np.ones((Nx, Ny), dtype=np.float64)

    def theta_relax(theta, intensity, outer):
        return theta + 0.2 * intensity

    def convergence(theta, theta_prev, info):
        return info["outer"] >= 3

    res = run_static_relax(
        theta0,
        A0,
        I0,
        theta_relax=theta_relax,
        controls=StaticRelaxControls(max_outer=10),
        convergence=convergence,
    )

    assert res.converged
    assert res.outer_steps == 3
    assert np.allclose(res.theta, 0.6)
    return {"static_outer_steps": res.outer_steps, "static_theta_max": float(res.theta.max())}


def main():
    _ensure_src_on_path()

    metrics = {}
    metrics.update(test_thomas())
    metrics.update(test_fft_y())
    metrics.update(test_theta_cn_and_picard())
    metrics.update(test_splitstep())
    metrics.update(test_td_zmarch())
    metrics.update(test_static_relax())

    print("50_lc_algorithms_smoke")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.12g}")
        else:
            print(f"  {k:28s}: {v}")
    print("50_lc_algorithms_smoke: PASS")


if __name__ == "__main__":
    main()
