#!/usr/bin/env python3
"""
23_diagnostics_theta_check.py

Validate the shared theta diagnostics layer against the public theta engine.
"""

import argparse
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.theta_engine.engine import advance_theta_cn
from lc_soliton.diagnostics.theta import (
    summarize_theta_state,
    summarize_theta_step,
    theta_history_row,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    ctx.dt = float(args.dt)
    ctx.picard_iters = 20
    ctx.picard_tol_up = 1e-14
    ctx.picard_tol_cn = 1e-10
    ctx.theta_clamp = None

    theta1, info = advance_theta_cn(
        theta0,
        I,
        ctx,
        dt=float(args.dt),
        dtype=cp.float64,
        xp=cp,
    )

    st0 = summarize_theta_state(theta0, I, ctx, xp=cp)
    st1 = summarize_theta_step(theta1, theta0, I, ctx, dt=float(args.dt), xp=cp)
    row = theta_history_row(
        step=1,
        theta=theta1,
        theta_old=theta0,
        intensity=I,
        ctx=ctx,
        dt=float(args.dt),
        cn_info=info,
        xp=cp,
    )

    print()
    print("Theta diagnostics validation")
    print("-" * 72)
    print("state 0:", st0.as_dict())
    print("step  1:", st1.as_dict())
    print("history row:", row)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "theta_diagnostics_check.txt", "w") as f:
        f.write("state0\n")
        f.write(str(st0.as_dict()) + "\n\n")
        f.write("step1\n")
        f.write(str(st1.as_dict()) + "\n\n")
        f.write("row\n")
        f.write(str(row) + "\n")

    print(f"\n[wrote] {outdir / 'theta_diagnostics_check.txt'}")


if __name__ == "__main__":
    main()
