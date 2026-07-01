#!/usr/bin/env python3
import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.residuals import (
    pde_residual,
    residual_stats,
)

from lc_soliton.validated_core.runner_core import lc_residual64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta, I = load_ctx_and_profile(args.profile)

    theta = theta.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    R_old = lc_residual64(
        theta,
        I,
        b=float(ctx.b),
        bi=float(ctx.bi),
        du=float(ctx.du),
        dv=float(ctx.dv),
        mobility=float(ctx.mobility),
    )

    R_new = pde_residual(
        theta,
        I,
        du=float(ctx.du),
        dv=float(ctx.dv),
        b=float(ctx.b),
        bi=float(ctx.bi),
        mobility=float(ctx.mobility),
        xp=cp,
    )

    D = R_new - R_old

    s_old = residual_stats(R_old, xp=cp)
    s_new = residual_stats(R_new, xp=cp)

    diff = {
        "diff_rms": float(cp.sqrt(cp.mean(D * D))),
        "diff_max": float(cp.max(cp.abs(D))),
        "rel_diff": float(
            cp.sqrt(cp.mean(D * D))
            / (cp.sqrt(cp.mean(R_old * R_old)) + 1e-30)
        ),
    }

    print("\nLegacy")
    print(s_old)
    print("\nTheta engine")
    print(s_new)
    print("\nDifference")
    print(diff)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "residual_check.txt", "w") as f:
        f.write("legacy\n")
        f.write(str(s_old))
        f.write("\n\n")
        f.write("theta_engine\n")
        f.write(str(s_new))
        f.write("\n\n")
        f.write("difference\n")
        f.write(str(diff))
        f.write("\n")

    print(f"\n[wrote] {outdir / 'residual_check.txt'}")


if __name__ == "__main__":
    main()