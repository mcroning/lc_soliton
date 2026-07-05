#!/usr/bin/env python
"""61: smoke/equivalence checks for z-coupled CN theta kernel.

Checks:
1. gamma_z = 0 matches the ordinary 2-D CN/Picard theta step.
2. nonzero gamma_z changes the result and preserves boundary rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def rel_l2(a, b, asnumpy):
    aa = asnumpy(a)
    bb = asnumpy(b)
    return float(np.linalg.norm((aa - bb).ravel()) / (np.linalg.norm(bb.ravel()) + 1e-300))


def main() -> None:
    _ensure_src_on_path()

    from lc.numerics.backend import BackendSpec, get_backend, asnumpy
    from lc.algorithms.theta_cn import prepare_cn_operator
    from lc.algorithms.theta_picard import cn_trapezoid_picard_step
    from lc.algorithms.thomas import solve_const_offdiag_batched
    from lc.algorithms.theta_cn_zcoupled import cn_trapezoid_picard_step_zcoupled

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    args = ap.parse_args()

    backend = get_backend(BackendSpec(args.backend, args.precision, verbose=True))
    xp = backend.xp

    if args.tridiag == "fast":
        from lc.algorithms.thomas_fast import solve_const_offdiag_batched_fast as tridiag_solver
    else:
        tridiag_solver = solve_const_offdiag_batched

    Nx = args.Nx
    Ny = args.Ny
    dx = 2.0 / (Nx - 1)
    dy = dx * (100.0 / 75.0)

    x = xp.linspace(-1.0, 1.0, Nx, dtype=backend.real_dtype)
    y = xp.linspace(-1.0, 1.0, Ny, dtype=backend.real_dtype)
    X = x[:, None]
    Y = y[None, :]

    theta = (0.1 + 0.6 * xp.cos(0.5 * xp.pi * X) * xp.exp(-4.0 * Y * Y)).astype(backend.real_dtype)
    theta[0, :] = 0.0
    theta[-1, :] = 0.0

    I = xp.exp(-20.0 * (X * X + Y * Y)).astype(backend.real_dtype)

    theta_prev = (theta - 0.01 * xp.cos(2.0 * xp.pi * Y)).astype(backend.real_dtype)
    theta_next = (theta + 0.02 * xp.cos(2.0 * xp.pi * Y)).astype(backend.real_dtype)
    theta_prev[0, :] = 0.0
    theta_prev[-1, :] = 0.0
    theta_next[0, :] = 0.0
    theta_next[-1, :] = 0.0

    s, off, diag, lam = prepare_cn_operator(
        dt=7.5e-4,
        mobility=1.0,
        dx=dx,
        dy=dy,
        Ny=Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    common = dict(
        dt=7.5e-4,
        b=1.71860665806,
        bi=214.28571428571428,
        mobility=1.0,
        dx=dx,
        dy=dy,
        s=s,
        off=off,
        diag=diag,
        max_iter=4,
        tol_update=1e-7 if args.precision == "float64" else 1e-6,
        clamp=(0.0, np.pi / 2),
        tridiag_solver=tridiag_solver,
        xp=xp,
    )

    ref = cn_trapezoid_picard_step(
        theta,
        I,
        I,
        b=common["b"],
        bi=common["bi"],
        dt=common["dt"],
        mobility=common["mobility"],
        dx=common["dx"],
        dy=common["dy"],
        s=common["s"],
        off=common["off"],
        diag=common["diag"],
        max_iter=common["max_iter"],
        tol_update=common["tol_update"],
        clamp=common["clamp"],
        tridiag_solver=common["tridiag_solver"],
        xp=xp,
    )

    z0 = cn_trapezoid_picard_step_zcoupled(
        theta,
        I,
        theta_prev=theta_prev,
        theta_next=theta_next,
        gamma_z=0.0,
        inv_dz2=1.0,
        **common,
    )

    zc = cn_trapezoid_picard_step_zcoupled(
        theta,
        I,
        theta_prev=theta_prev,
        theta_next=theta_next,
        gamma_z=0.25,
        inv_dz2=0.01,
        **common,
    )

    err0 = rel_l2(z0, ref, asnumpy)
    errc = rel_l2(zc, ref, asnumpy)

    zc_np = asnumpy(zc)
    bc_max = max(float(np.max(np.abs(zc_np[0, :]))), float(np.max(np.abs(zc_np[-1, :]))))

    print("61_theta_cn_zcoupled_smoke")
    print(f"  backend                 : {backend.name}")
    print(f"  precision               : {args.precision}")
    print(f"  tridiag                 : {args.tridiag}")
    print(f"  Nx                      : {Nx}")
    print(f"  Ny                      : {Ny}")
    print(f"  gamma0_rel_l2_vs_plain  : {err0:.12e}")
    print(f"  gamma_rel_l2_vs_plain   : {errc:.12e}")
    print(f"  boundary_max_abs        : {bc_max:.12e}")
    print(f"  zcoupled_min            : {float(np.min(zc_np)):.12g}")
    print(f"  zcoupled_max            : {float(np.max(zc_np)):.12g}")

    tol0 = 5e-12 if args.precision == "float64" else 5e-6
    assert err0 < tol0
    assert errc > 1e-8
    assert bc_max < 1e-12 if args.precision == "float64" else bc_max < 1e-6
    assert np.all(np.isfinite(zc_np))

    print("61_theta_cn_zcoupled_smoke: PASS")


if __name__ == "__main__":
    main()
