# lc_core/td_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TDRunResult:
    """
    Result container for a true-time LC run.
    """
    theta_full: Any
    I_mid_store: Any
    stores: Any
    summary: dict


def patch_legacy_td_ctx_defaults(ctx):
    """
    Compatibility defaults required by the legacy monolithic TD/static runner.
    """
    defaults = dict(
        use_dual_grid=False,

        dn_max_est=0.02,
        dz_opt_max_phi=0.5,
        max_substeps=8,

        theta_clamp=None,
        true_td_full_pred_optics=False,

        save_full_I_mid_store=True,
        store_slice_movies=True,

        runtime_every_k=0,
        linearized_td=False,

        strict_selfcons_passes=3,
        strict_selfcons_tol_theta=1e-4,
        strict_selfcons_tol_I=1e-4,
    )

    for k, v in defaults.items():
        if not hasattr(ctx, k):
            setattr(ctx, k, v)

    if not hasattr(ctx, "_h_cache"):
        ctx._h_cache = {}

    if not hasattr(ctx, "h_cache"):
        ctx.h_cache = ctx._h_cache

    return ctx


def run_true_td_bridge(
    ctx,
    stores,
    *,
    run_unified_td_static,
    restart_theta=True,
    store_I_dtype=None,
    static_mode="strict_relax",
    use_quasiglobal_td=False,
    report_runtime=True,
    tqdm_timedep=True,
    tqdm_static_z=False,
    true_td=True,
    save_slices=True,
    strict_residual_tol_max=5e-2,
    strict_residual_tol_rms=1e-2,
    strict_max_outer_passes=8,
):
    """
    Run the legacy true-TD branch through a clean new-core wrapper.
    """
    cp = ctx.xp
    patch_legacy_td_ctx_defaults(ctx)

    if store_I_dtype is None:
        store_I_dtype = cp.float32

    save_slices_fn = stores.save_slices if save_slices else None

    I_mid_store = run_unified_td_static(
        ctx,
        timedep=True,
        time_steps=stores.time_steps,
        tsteps=len(stores.time_steps),
        t_stride=stores.t_stride,
        restart_theta=bool(restart_theta),
        store_I_dtype=store_I_dtype,
        save_slices_fn=save_slices_fn,
        true_td=bool(true_td),
        static_mode=static_mode,
        use_quasiglobal_td=bool(use_quasiglobal_td),
        report_runtime=bool(report_runtime),
        tqdm_timedep=bool(tqdm_timedep),
        tqdm_static_z=bool(tqdm_static_z),
        strict_residual_tol_max=float(strict_residual_tol_max),
        strict_residual_tol_rms=float(strict_residual_tol_rms),
        strict_max_outer_passes=int(strict_max_outer_passes),
    )

    summary = summarize_td_result(ctx, stores, I_mid_store)

    return TDRunResult(
        theta_full=ctx.theta_full,
        I_mid_store=I_mid_store,
        stores=stores,
        summary=summary,
    )


def summarize_td_result(ctx, stores, I_mid_store=None) -> dict:
    cp = ctx.xp

    theta = ctx.theta_full
    t_total = float(cp.sum(stores.time_steps).get())

    summary = dict(
        mode="true_td_bridge",
        Nt=int(len(stores.time_steps)),
        dt_first=float(stores.time_steps[0].get() if hasattr(stores.time_steps[0], "get") else stores.time_steps[0]),
        t_total=t_total,
        Nz=int(ctx.Nz),
        dz_um=float(ctx.dz),
        theta_shape=list(theta.shape),
        theta_min=float(cp.min(theta).get()),
        theta_max=float(cp.max(theta).get()),
        theta_mean=float(cp.mean(theta).get()),
        has_I_mid_store=I_mid_store is not None,
        has_Ixz=stores.Ixz is not None,
        has_Iyz=stores.Iyz is not None,
        has_thetaxz=stores.thetaxz is not None,
        has_thetayz=stores.thetayz is not None,
    )

    if I_mid_store is not None:
        summary.update(
            dict(
                I_mid_shape=list(I_mid_store.shape),
                I_mid_min=float(cp.min(I_mid_store).get()),
                I_mid_max=float(cp.max(I_mid_store).get()),
                I_mid_mean=float(cp.mean(I_mid_store).get()),
            )
        )

    return summary


def write_td_outputs(td_result: TDRunResult, run_dir):
    """
    Save TD JSON summary only.

    Large arrays should be saved with lc_core.io_arrays, not here.
    """
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "td_summary.json"
    path.write_text(json.dumps(td_result.summary, indent=2))

    return td_result.summary
