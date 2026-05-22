# lc_core/pipeline.py
from __future__ import annotations

from .run_manager import make_run_dir, save_config_json, save_environment_json
from .gpu_context import build_gpu_context
from .static_z_march import run_static_strict_z_march_bridge, write_static_z_march_outputs
from .td_runner import run_true_td_bridge, write_td_outputs
from .residual_quality import save_defect_diagnostics
from .io_arrays import save_static_arrays


def run_static_experiment(
    cfg,
    *,
    run_root=None,
    build_context_kwargs,
    static_bridge_kwargs,
    tag="strict_static",
    save_arrays=True,
):
    run_dir = make_run_dir(root=run_root, tag=tag)
    save_config_json(cfg, run_dir)
    save_environment_json(run_dir)

    ctx, stores, launch_arrays = build_gpu_context(
        cfg,
        **build_context_kwargs,
    )

    res = run_static_strict_z_march_bridge(
        ctx,
        **static_bridge_kwargs,
    )

    write_static_z_march_outputs(res, run_dir)
    save_defect_diagnostics(ctx, res, run_dir)

    if save_arrays:
        save_static_arrays(
            res,
            run_dir,
            save_theta=True,
            save_I_mid=True,
            save_final_amp=False,
        )

    return ctx, stores, launch_arrays, res, run_dir


def run_td_experiment(
    cfg,
    *,
    run_root=None,
    build_context_kwargs,
    td_bridge_kwargs,
    tag="true_td",
    save_arrays=True,
):
    run_dir = make_run_dir(root=run_root, tag=tag)
    save_config_json(cfg, run_dir)
    save_environment_json(run_dir)

    ctx, stores, launch_arrays = build_gpu_context(
        cfg,
        **build_context_kwargs,
    )

    res = run_true_td_bridge(
        ctx,
        stores,
        **td_bridge_kwargs,
    )

    write_td_outputs(res, run_dir)

    if save_arrays:
        save_static_arrays(
            res,
            run_dir,
            save_theta=True,
            save_I_mid=True,
            save_final_amp=False,
        )

    return ctx, stores, launch_arrays, res, run_dir