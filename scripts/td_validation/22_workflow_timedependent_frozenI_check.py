#!/usr/bin/env python3
"""
22_workflow_timedependent_frozenI_check.py

Validate the time-dependent theta workflow using frozen intensity.

Expected:
    PDE residual should decrease similarly to script 19, while each
    accepted CN step has small CN residual.
"""

import argparse
import csv
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile
from lc_soliton.workflows.timedependent import run_theta_td_frozen_intensity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--record-every", type=int, default=1)
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

    result = run_theta_td_frozen_intensity(
        theta0,
        I,
        ctx,
        dt=float(args.dt),
        nsteps=int(args.steps),
        dtype=cp.float64,
        xp=cp,
        record_every=int(args.record_every),
    )

    print()
    print("Workflow TD frozen-I validation")
    print("-" * 72)

    milestones = {0, 1, 2, 5, 10, 20, 50, 100, int(args.steps)}
    for row in result.history:
        if row["step"] in milestones:
            print(
                f"{row['step']:4d}  "
                f"PDE={row['pde_rms']:.6e}  "
                f"CN={row['cn_rms']:.3e}  "
                f"dtheta={row['dtheta_rms']:.3e}  "
                f"picard={row['picard_iters']}"
            )

    print()
    print("Info")
    print(result.info)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    csvfile = outdir / "workflow_td_frozenI_history.csv"
    with open(csvfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.history[0].keys()))
        w.writeheader()
        w.writerows(result.history)

    with open(outdir / "workflow_td_frozenI_info.txt", "w") as f:
        f.write(str(result.info))
        f.write("\n")

    print()
    print("[wrote]", csvfile)


if __name__ == "__main__":
    main()
