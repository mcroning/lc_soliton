#!/usr/bin/env python3
"""
17_theta_engine_picard_pure64.py

Validate the new theta_engine Picard solver against the
pure float64 CN residual benchmark.
"""

import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.nonlinear import solve_theta_cn_picard
from lc_soliton.theta_engine.residuals import (
    cn_residual,
    residual_stats,
)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--picard-iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    theta1, info = solve_theta_cn_picard(
        theta0,
        I,
        dt=args.dt,
        du=ctx.du,
        dv=ctx.dv,
        b=ctx.b,
        bi=ctx.bi,
        mobility=ctx.mobility,
        theta_bc=ctx.theta_bc,
        theta_clamp=None,
        max_iter=args.picard_iters,
        tol_update=1e-15,
        tol_cn=1e-12,
        xp=cp,
        monitor_cn=True,
        save_history=True,
    )

    print()
    print("Theta Engine Picard Audit")
    print("-" * 80)

    print(
        f"{'it':>3} "
        f"{'update':>12} "
        f"{'CN rms':>12} "
        f"{'CN max':>12}"
    )

    for h in info["history"]:
        print(
            f"{h['iter']:3d} "
            f"{h['update_rms']:12.6e} "
            f"{h['cn_rms']:12.6e} "
            f"{h['cn_max']:12.6e}"
        )

    # independent check of returned solution
    R = cn_residual(
        theta1,
        theta0,
        I,
        dt=args.dt,
        du=ctx.du,
        dv=ctx.dv,
        b=ctx.b,
        bi=ctx.bi,
        mobility=ctx.mobility,
        xp=cp,
    )

    stats = residual_stats(R, xp=cp)

    print()
    print("Returned solution")
    print("-----------------")
    print(stats)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "theta_engine_picard.txt", "w") as f:

        for h in info["history"]:
            f.write(
                f"{h['iter']:3d} "
                f"{h['update_rms']:.12e} "
                f"{h['cn_rms']:.12e} "
                f"{h['cn_max']:.12e}\n"
            )

        f.write("\n")
        f.write(str(stats))
        f.write("\n")

    print(f"\n[wrote] {outdir/'theta_engine_picard.txt'}")


if __name__ == "__main__":
    main()
