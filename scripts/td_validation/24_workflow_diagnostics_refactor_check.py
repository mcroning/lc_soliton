#!/usr/bin/env python3
"""
24_workflow_diagnostics_refactor_check.py

Validate that the static and time-dependent workflow layers use the shared
theta diagnostics and preserve the numerical behavior established by tests
20-22.
"""

import argparse
import csv
from pathlib import Path

import cupy as cp

from load_profile import load_ctx_and_profile

from lc_soliton.workflows.static import run_static_theta
from lc_soliton.workflows.timedependent import run_timedependent_theta_frozenI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx, mode, theta0, I = load_ctx_and_profile(args.profile)

    theta0 = theta0.astype(cp.float64, copy=False)
    I = I.astype(cp.float64, copy=False)

    # Minimal context values used by the engine/workflows.
    ctx.dt = float(args.dt)
    ctx.dtau_static = float(args.dt)
    ctx.static_max_steps = int(args.steps)
    ctx.static_tol_rms = 5e-3
    ctx.static_tol_max = 5e-2

    ctx.picard_iters = 20
    ctx.picard_tol_up = 1e-14
    ctx.picard_tol_cn = 1e-10
    ctx.theta_clamp = None

    static_result = run_static_theta(
        theta0,
        I,
        ctx,
        dtau=float(args.dt),
        max_steps=int(args.steps),
        residual_tol_rms=5e-3,
        residual_tol_max=5e-2,
        dtype=cp.float64,
        xp=cp,
    )

    td_result = run_timedependent_theta_frozenI(
        theta0,
        I,
        ctx,
        dt=float(args.dt),
        steps=int(args.steps),
        dtype=cp.float64,
        xp=cp,
    )

    print()
    print("Workflow diagnostics refactor validation")
    print("-" * 72)
    print("Static initial:", static_result.diagnostics_initial)
    print("Static final  :", static_result.diagnostics_final)
    print("Static info   :", {k: v for k, v in static_result.info.items() if k != "last_cn_info"})
    print()
    print("TD first row  :", td_result.history[0])
    print("TD final row  :", td_result.history[-1])
    print("TD info       :", {k: v for k, v in td_result.info.items() if k != "last_cn_info"})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "td_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(td_result.history[0].keys()))
        w.writeheader()
        w.writerows(td_result.history)

    with open(outdir / "summary.txt", "w") as f:
        f.write("static_initial\n")
        f.write(str(static_result.diagnostics_initial))
        f.write("\n\nstatic_final\n")
        f.write(str(static_result.diagnostics_final))
        f.write("\n\nstatic_info\n")
        f.write(str(static_result.info))
        f.write("\n\ntd_info\n")
        f.write(str(td_result.info))
        f.write("\n")

    print()
    print("[wrote]", outdir / "summary.txt")
    print("[wrote]", outdir / "td_history.csv")


if __name__ == "__main__":
    main()
