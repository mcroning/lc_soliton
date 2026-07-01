#!/usr/bin/env python3
import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.engine import (
    advance_theta_cn,
    solve_theta_steady,
)
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
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--steady-steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    # Provide sane public-API defaults for the lightweight ctx.
    ctx.dt = float(args.dt)
    ctx.picard_iters = 20
    ctx.picard_tol_up = 1e-14
    ctx.picard_tol_cn = 1e-10
    ctx.theta_clamp = None
    ctx.dtau_static = float(args.dt)
    ctx.static_max_steps = int(args.steady_steps)
    ctx.static_tol_rms = 5e-3
    ctx.static_tol_max = 5e-2

    st0 = pde_stats(theta0, I, ctx)

    theta1, info1 = advance_theta_cn(
        theta0,
        I,
        ctx,
        dt=float(args.dt),
        dtype=cp.float64,
        xp=cp,
    )
    st1 = pde_stats(theta1, I, ctx)

    thetaS, infoS = solve_theta_steady(
        theta0,
        I,
        ctx,
        dtau=float(args.dt),
        max_steps=int(args.steady_steps),
        residual_tol_rms=5e-3,
        residual_tol_max=5e-2,
        dtype=cp.float64,
        xp=cp,
    )
    stS = pde_stats(thetaS, I, ctx)

    print()
    print("Theta engine public API validation")
    print("-" * 72)
    print("Initial PDE:", st0)
    print("One-step PDE:", st1)
    print("One-step info:", {k: v for k, v in info1.items() if k != "history"})
    print("Steady PDE:", stS)
    print("Steady info:", {k: v for k, v in infoS.items() if k != "last_cn_info"})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "public_api_check.txt", "w") as f:
        f.write("initial\n")
        f.write(str(st0) + "\n\n")
        f.write("one_step\n")
        f.write(str(st1) + "\n")
        f.write(str({k: v for k, v in info1.items() if k != "history"}) + "\n\n")
        f.write("steady\n")
        f.write(str(stS) + "\n")
        f.write(str({k: v for k, v in infoS.items() if k != "last_cn_info"}) + "\n")

    print(f"\n[wrote] {outdir / 'public_api_check.txt'}")


if __name__ == "__main__":
    main()
