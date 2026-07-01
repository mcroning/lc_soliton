#!/usr/bin/env python3
import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.nonlinear import solve_theta_cn_picard
from lc_soliton.theta_engine.residuals import pde_residual, cn_residual, residual_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    theta1, info = solve_theta_cn_picard(
        theta0,
        I,
        dt=float(args.dt),
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(ctx.mobility),
        theta_bc=float(ctx.theta_bc),
        theta_clamp=None,
        max_iter=20,
        tol_update=1e-14,
        tol_cn=1e-10,
        monitor_cn=True,
        save_history=True,
        xp=cp,
    )

    dtheta = theta1 - theta0

    R0 = pde_residual(theta0, I, du=ctx.du, dv=ctx.dv, b=ctx.b, bi=ctx.bi, mobility=ctx.mobility, xp=cp)
    R1 = pde_residual(theta1, I, du=ctx.du, dv=ctx.dv, b=ctx.b, bi=ctx.bi, mobility=ctx.mobility, xp=cp)
    C1 = cn_residual(theta1, theta0, I, dt=args.dt, du=ctx.du, dv=ctx.dv, b=ctx.b, bi=ctx.bi, mobility=ctx.mobility, xp=cp)

    print("theta_engine TD one-step")
    print("-" * 72)
    print("dtheta_rms:", float(cp.sqrt(cp.mean(dtheta[1:-1, :] ** 2))))
    print("dtheta_max:", float(cp.max(cp.abs(dtheta[1:-1, :]))))
    print("PDE residual before:", residual_stats(R0, xp=cp))
    print("PDE residual after :", residual_stats(R1, xp=cp))
    print("CN residual after  :", residual_stats(C1, xp=cp))
    print("solver info:", {k: v for k, v in info.items() if k != "history"})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "theta_engine_td_onestep.txt", "w") as f:
        f.write(str({k: v for k, v in info.items() if k != "history"}) + "\n")

    print(f"[wrote] {outdir/'theta_engine_td_onestep.txt'}")


if __name__ == "__main__":
    main()
