from __future__ import annotations

from pathlib import Path

from .run_manager import make_run_dir, save_config_json, save_environment_json

from .run_loader import load_prdata, load_saved_result

from .config import config_from_prdata, validate_config
from .gpu_context import build_gpu_context

from .td_runner import run_true_td_bridge, write_td_outputs
from .io_arrays import save_static_arrays
from .io_movies import save_td_movies


def continue_td_from_run(
    source_run_dir,
    *,
    build_context_kwargs,
    td_bridge_kwargs,
    run_root=None,
    tag=None,
    save_arrays=True,
    save_movies=True,
    xp=None,
    tend=None,
    tsteps=None,
    t_stride=None,
):
    source_run_dir = Path(source_run_dir)
    
    if tag is None:
        tag = f"td_from_{source_run_dir.name}"
    
    # Load source run parameters, then force continuation to be TD
    prdata = load_prdata(source_run_dir)
    
    prdata["time_behavior"] = "Time Dependent"
    prdata["save_full_I_mid_store"] = True
    
    if tend is not None:
        prdata["tend"] = float(tend)
    
    if tsteps is not None:
        prdata["tsteps"] = int(tsteps)
    
    if t_stride is not None:
        prdata["t_stride"] = int(t_stride)
    
    cfg = config_from_prdata(prdata)
    validate_config(cfg)
    
    cont_run_dir = make_run_dir(
        root=run_root,
        tag=tag,
    )
    
    save_config_json(cfg, cont_run_dir)
    save_environment_json(cont_run_dir)
    
    ctx, stores, launch_arrays = build_gpu_context(
        cfg,
        **build_context_kwargs,
    )
    
    if xp is None:
        xp = ctx.xp
    
    saved = load_saved_result(
        source_run_dir,
        xp=xp,
    )
    
    if saved.theta_full.shape != ctx.theta_full.shape:
        raise ValueError(
            "Saved theta_full shape does not match rebuilt context: "
            f"{saved.theta_full.shape} vs {ctx.theta_full.shape}"
        )
    
    ctx.theta_full[...] = saved.theta_full
    
    td_kwargs = dict(td_bridge_kwargs)
    td_kwargs["restart_theta"] = False

    td_res = run_true_td_bridge(
        ctx,
        stores,
        **td_kwargs,
    )

    write_td_outputs(td_res, cont_run_dir)

    if save_arrays:
        save_static_arrays(
            td_res,
            cont_run_dir,
            save_theta=True,
            save_I_mid=True,
            save_final_amp=False,
        )

    if save_movies:
        save_td_movies(
            td_res,
            cont_run_dir,
        )

    return ctx, stores, launch_arrays, td_res, cont_run_dir