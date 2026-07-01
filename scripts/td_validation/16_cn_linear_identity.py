#!/usr/bin/env python3
"""
Verify that the new theta_engine CN linear solve is identical
to the legacy validated implementation.
"""

import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.cn import (
    prepare_cn_operator,
    solve_cn_linear_system,
)

from lc_soliton.validated_core.runner_core import (
    prepare_cn_ky_operator,
    _cn_solve_dirichletx_periody,
)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta, I = load_ctx_and_profile(args.profile)

    theta = theta.astype(cp.float64)

    ##################################################################
    # identical operator construction
    ##################################################################

    s_old, off_old, diag_old, lam_old = prepare_cn_ky_operator(
        dt=args.dt,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        Ny=theta.shape[1],
    )

    s_new, off_new, diag_new, lam_new = prepare_cn_operator(
        dt=args.dt,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        Ny=theta.shape[1],
        xp=cp,
        dtype=theta.dtype,
    )

    ##################################################################
    # random RHS
    ##################################################################

    rng = cp.random.default_rng(12345)

    rhs = rng.standard_normal(theta.shape).astype(theta.dtype)

    theta_bc = cp.asarray(ctx.theta_bc, dtype=theta.dtype)

    ##################################################################
    # solve
    ##################################################################

    th_old = _cn_solve_dirichletx_periody(
        rhs,
        off=off_old,
        diag=diag_old,
        theta_bc=theta_bc,
    )

    th_new = solve_cn_linear_system(
        rhs,
        off=off_new,
        diag=diag_new,
        theta_bc=theta_bc,
        xp=cp,
    )

    ##################################################################
    # compare
    ##################################################################

    D = th_new - th_old

    rms = float(cp.sqrt(cp.mean(D * D)))
    mx = float(cp.max(cp.abs(D)))

    rel = rms / (
        float(cp.sqrt(cp.mean(th_old * th_old))) + 1e-30
    )

    print()
    print("CN linear identity")
    print("-" * 60)
    print(f"diff_rms  = {rms:.6e}")
    print(f"diff_max  = {mx:.6e}")
    print(f"rel_diff  = {rel:.6e}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "cn_linear_identity.txt", "w") as f:
        f.write(f"diff_rms {rms}\n")
        f.write(f"diff_max {mx}\n")
        f.write(f"rel_diff {rel}\n")

    print(f"\n[wrote] {outdir/'cn_linear_identity.txt'}")


if __name__ == "__main__":
    main()
