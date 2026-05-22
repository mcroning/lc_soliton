# lc_core/io.py
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import RunConfig, config_to_dict


def make_run_dir(cfg: RunConfig, label: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")

    base = Path(cfg.save.base_dir).expanduser()
    run_name = f"{label}_{cfg.save.run_name}_{stamp}"

    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "figures").mkdir()
    (run_dir / "slices").mkdir()
    (run_dir / "profiles").mkdir()

    return run_dir


def write_repro_json(cfg: RunConfig, run_dir: Path) -> Path:
    path = run_dir / "repro.json"
    path.write_text(json.dumps(config_to_dict(cfg), indent=2))
    return path


def write_json(obj: dict, path: Path) -> Path:
    path.write_text(json.dumps(obj, indent=2))
    return path


def write_gpu_smoke_summary(cfg: RunConfig, ctx, launch_arrays, run_dir: Path) -> Path:
    cp = ctx.xp

    P0 = float(cp.sum(launch_arrays.I0).get() * ctx.dx * ctx.dy)
    Imax = float(cp.max(launch_arrays.I0).get())

    summary = dict(
        Nx=ctx.Nx,
        Ny=ctx.Ny,
        Nz=ctx.Nz,
        dx_um=ctx.dx,
        dy_um=ctx.dy,
        dz_um=ctx.dz,
        refin=ctx.refin,
        b=ctx.b,
        bi=ctx.bi,
        coh=ctx.coh,
        amp0_shape=list(ctx.amp0.shape),
        theta_shape=list(ctx.theta_full.shape),
        normalized_power=P0,
        Imax=Imax,
    )

    return write_json(summary, run_dir / "gpu_smoke_summary.json")