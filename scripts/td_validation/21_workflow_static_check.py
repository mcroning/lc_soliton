#!/usr/bin/env python3
"""
21_workflow_static_check.py

Validate the public static workflow.

This should produce essentially the same result as
20_theta_engine_public_api.py, but through the workflow layer.
"""

import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.workflows.static import run_static_theta
from lc_soliton.theta_engine.residuals import (
    pde_residual,
    residual_stats,
)


def pde_stats(theta, I, ctx):
    R = pde_residual(
        theta,
        I,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(getattr(ctx, "mobility", 1.0)),
        xp=cp,
    )
    return residual_stats(R, xp=cp)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    #
    # Populate the minimal context expected by the engine.
    #

    ctx.dtau_static = args.dt
    ctx.static_max_steps = args.steps

    ctx.picard_iters = 20
    ctx.picard_tol_up = 1e-14
    ctx.picard_tol_cn = 1e-10

    ctx.static_tol_rms = 5e-3
    ctx.static_tol_max = 5e-2

    ctx.theta_clamp = None

    before = pde_stats(theta0, I, ctx)

    result = run_static_theta(
        theta0,
        I,
        ctx,
        dtype=cp.float64,
        xp=cp,
    )
    theta = result.theta
    info = result.info

    after = pde_stats(theta, I, ctx)

    print()
    print("Workflow static validation")
    print("-" * 72)

    print("Initial PDE")
    print(before)

    print()

    print("Final PDE")
    print(after)

    print()

    print("Workflow info")
    print(info)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "workflow_static_check.txt", "w") as f:
        f.write("initial\n")
        f.write(str(before))
        f.write("\n\n")

        f.write("final\n")
        f.write(str(after))
        f.write("\n\n")

        f.write("info\n")
        f.write(str(info))
        f.write("\n")

    print()
    print("[wrote]", outdir / "workflow_static_check.txt")


if __name__ == "__main__":
    main()
