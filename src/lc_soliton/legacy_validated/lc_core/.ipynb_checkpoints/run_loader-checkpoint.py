from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import config_from_prdata, validate_config
from .gpu_context import build_gpu_context


def load_config_dict(run_dir):
    run_dir = Path(run_dir)
    return json.loads((run_dir / "config.json").read_text())


def load_prdata(run_dir):
    cfg_dict = load_config_dict(run_dir)
    return cfg_dict["legacy_prdata"]


def rebuild_cfg(run_dir, *, validate=True):
    prdata = load_prdata(run_dir)
    cfg = config_from_prdata(prdata)

    if validate:
        validate_config(cfg)

    return cfg


def rebuild_ctx(run_dir, *, build_context_kwargs):
    cfg = rebuild_cfg(run_dir)

    ctx, stores, launch_arrays = build_gpu_context(
        cfg,
        **build_context_kwargs,
    )

    return ctx, stores, launch_arrays


def load_array(run_dir, name, *, xp=None):
    run_dir = Path(run_dir)
    path = run_dir / "arrays" / f"{name}.npy"

    arr = np.load(path)

    if xp is not None:
        arr = xp.asarray(arr)

    return arr


def load_saved_result(run_dir, *, xp=None):
    run_dir = Path(run_dir)

    theta = load_array(run_dir, "theta_full", xp=xp)
    I_mid = load_array(run_dir, "I_mid_store", xp=xp)

    return SimpleNamespace(
        theta_full=theta,
        I_mid_store=I_mid,
        run_dir=run_dir,
    )


def load_movie(run_dir, name, *, xp=None):
    run_dir = Path(run_dir)
    path = run_dir / "movies" / f"{name}.npy"

    arr = np.load(path)

    if xp is not None:
        arr = xp.asarray(arr)

    return arr


def list_run_contents(run_dir):
    run_dir = Path(run_dir)

    out = dict(
        run_dir=str(run_dir),
        files=sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("*") if p.is_file()),
        arrays=sorted(str(p.relative_to(run_dir)) for p in (run_dir / "arrays").glob("*"))
        if (run_dir / "arrays").exists() else [],
        movies=sorted(str(p.relative_to(run_dir)) for p in (run_dir / "movies").glob("*"))
        if (run_dir / "movies").exists() else [],
        comparisons=sorted(str(p.relative_to(run_dir)) for p in (run_dir / "comparisons").glob("*"))
        if (run_dir / "comparisons").exists() else [],
        diagnostics=sorted(str(p.relative_to(run_dir)) for p in (run_dir / "pde_defect_diagnostics").glob("*"))
        if (run_dir / "pde_defect_diagnostics").exists() else [],
    )

    return out


def rebuild_td_ctx_from_run(
    run_dir,
    *,
    build_context_kwargs,
    tend,
    tsteps,
    t_stride=1,
    save_full_I_mid_store=True,
):
    """
    Rebuild a TD-capable ctx/stores/launch_arrays from any saved run.

    Useful for:
      - TD continuation from strict static
      - temporal stability probes
      - testing alternative TD integrators
    """
    prdata = load_prdata(run_dir)

    prdata["time_behavior"] = "Time Dependent"
    prdata["tend"] = float(tend)
    prdata["tsteps"] = int(tsteps)
    prdata["t_stride"] = int(t_stride)
    prdata["save_full_I_mid_store"] = bool(save_full_I_mid_store)

    cfg = config_from_prdata(prdata)
    validate_config(cfg)

    ctx, stores, launch_arrays = build_gpu_context(
        cfg,
        **build_context_kwargs,
    )

    return ctx, stores, launch_arrays