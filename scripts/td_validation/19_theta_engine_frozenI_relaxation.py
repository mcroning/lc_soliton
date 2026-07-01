#!/usr/bin/env python3
"""
19_theta_engine_frozenI_relaxation.py

Repeated CN timesteps with frozen intensity.

The question:

    Does the PDE residual decrease monotonically?

This is NOT a TD optics test.
The optical field is held fixed.
"""

import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.nonlinear import solve_theta_cn_picard
from lc_soliton.theta_engine.residuals import (
    pde_residual,
    residual_stats,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    ctx, mode, theta, I = load_ctx_and_profile(args.profile)

    theta = theta.astype(cp.float64)
    I = I.astype(cp.float64)

    print()
    print("Frozen-I relaxation")
    print("-"*72)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    csv = open(outdir/"history.csv","w")
    csv.write(
        "step,"
        "PDE_Rrms,"
        "PDE_Rmax,"
        "CN_Rrms,"
        "dtheta_rms,"
        "picard_iters\n"
    )

    for step in range(args.steps+1):

        Rpde = pde_residual(
            theta,
            I,
            du=ctx.du,
            dv=ctx.dv,
            b=ctx.b,
            bi=ctx.bi,
            mobility=ctx.mobility,
            xp=cp,
        )

        spde = residual_stats(Rpde, xp=cp)

        if step == 0:

            cn_rms = 0.0
            dtheta = 0.0
            nit = 0

        else:

            theta_old = theta.copy()

            theta, info = solve_theta_cn_picard(
                theta_old,
                I,
                dt=args.dt,
                du=ctx.du,
                dv=ctx.dv,
                b=ctx.b,
                bi=ctx.bi,
                mobility=ctx.mobility,
                theta_bc=ctx.theta_bc,
                theta_clamp=None,
                max_iter=20,
                tol_update=1e-14,
                tol_cn=1e-10,
                monitor_cn=True,
                save_history=False,
                xp=cp,
            )

            cn_rms = info["cn_rms"]
            nit = info["niter"]

            D = theta-theta_old
            dtheta = float(
                cp.sqrt(
                    cp.mean(
                        D[1:-1,:]**2
                    )
                )
            )

        csv.write(
            f"{step},"
            f"{spde['rms_interior']:.12e},"
            f"{spde['max_interior']:.12e},"
            f"{cn_rms:.12e},"
            f"{dtheta:.12e},"
            f"{nit}\n"
        )

        if step in [0,1,2,5,10,20,50,100]:

            print(
                f"{step:4d}  "
                f"PDE={spde['rms_interior']:.6e}  "
                f"CN={cn_rms:.3e}  "
                f"dθ={dtheta:.3e}"
            )

    csv.close()

    print()
    print("[wrote]", outdir/"history.csv")


if __name__ == "__main__":
    main()
