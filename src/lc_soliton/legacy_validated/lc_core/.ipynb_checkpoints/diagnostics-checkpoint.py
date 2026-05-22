# lc_core/diagnostics.py
from __future__ import annotations

import json
from pathlib import Path

from .residuals import (
    lc_residual2d_ctx,
    lc_residual3d_ctx,
    residual_stats_2d,
    residual_stats_3d,
)


def initial_residual_report(ctx, cp):
    Izero2 = cp.zeros_like(ctx.theta_bias_2d)
    R2 = lc_residual2d_ctx(ctx, ctx.theta_bias_2d, Izero2)

    Izero3 = cp.zeros_like(ctx.theta_full)
    R3 = lc_residual3d_ctx(ctx, ctx.theta_full, Izero3)

    return dict(
        theta_bias_2d=residual_stats_2d(R2),
        theta_full_3d=residual_stats_3d(R3),
        theta_z_gamma=float(ctx.theta_z_gamma),
        b=float(ctx.b),
        bi=float(ctx.bi),
        du=float(ctx.du),
        dv=float(ctx.dv),
        dz=float(ctx.dz),
    )


def write_initial_residual_report(ctx, cp, run_dir):
    run_dir = Path(run_dir)
    report = initial_residual_report(ctx, cp)

    path = run_dir / "initial_residual_report.json"
    path.write_text(json.dumps(report, indent=2))

    return report