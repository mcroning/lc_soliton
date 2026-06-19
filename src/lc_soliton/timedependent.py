"""
Public time-dependent propagation API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import json
import time
from dataclasses import asdict

import numpy as np

from .core.context import LCParams, make_context, apply_legacy_context_aliases
from .core.storage import LightStore
from .legacy_validated.runner_utils import _prepare_substeps
from .legacy_validated.td_runner import _run_td_predictor
from .environment import write_environment_json


def run_td(
    params: LCParams,
    *,
    run_dir: str | Path,
    Nt: int,
    dt: float,
    t_stride: int = 1,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Callable[[str], None] | None = print,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Run a validated predictor-only time-dependent LC propagation calculation.

    This public entry point builds the LC context, prepares storage/metadata,
    and calls the TD workflow runner directly.
    """
    t0 = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx, dg = make_context(params)
###
    print("use_dual_grid =", getattr(params, "use_dual_grid", None))
    print("dual_grid_factor =", getattr(params, "dual_grid_factor", None))
    print("dg =", dg)
    if dg is not None:
        print("DG factor/shape =", dg.factor, dg.Nx_c, dg.Ny_c)
###
    xp = ctx.xp
    apply_legacy_context_aliases(ctx, params)

    launch_profile_path = getattr(params, "launch_profile_path", None)
    if launch_profile_path:
        raise NotImplementedError(
            "Direct TD launch from saved eigensoliton profiles has not yet been "
            "migrated from the legacy profile-loading path."
        )

    ctx.theta_full = xp.stack(
        [ctx.theta_bias_2d.copy() for _ in range(ctx.Nz)],
        axis=0,
    ).astype(xp.float32, copy=False)

    Nt_eff = int(Nt)
    t_stride = max(1, int(t_stride))
    Nt_out = (Nt_eff + t_stride - 1) // t_stride

    store = LightStore(
        run_dir,
        ctx,
        Nt_out=Nt_out,
        save_slices=save_slices,
        save_full=save_full,
    )

    nsub_meta, dz_sub_meta, phi_meta, _, _ = _prepare_substeps(
        ctx,
        params,
        use_core=False,
    )

    meta = asdict(params)
    meta.update(
        dict(
            mode="td_predictor_only",
            Nt=Nt_eff,
            dt=float(dt),
            t_stride=t_stride,
            Nt_out=Nt_out,
            nsub=nsub_meta,
            dz_sub=dz_sub_meta,
            phi_est=phi_meta,
            backend="cupy" if getattr(xp, "__name__", "") == "cupy" else "numpy",
            td_kernel="validated_core_direct",
        )
    )

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("use_dual_grid =", getattr(params, "use_dual_grid", None))
    print("dual_grid_factor =", getattr(params, "dual_grid_factor", None))
    stopped = False
    try:
        stopped = _run_td_predictor(
            ctx=ctx,
            dg=dg,
            params=params,
            store=store,
            Nt=Nt_eff,
            dt=float(dt),
            t_stride=t_stride,
            progress=progress,
            should_stop=should_stop,
        )

        np.savez(
            run_dir / "final_summary.npz",
            elapsed_s=time.time() - t0,
            stopped=stopped,
        )
        if stopped and progress:
            progress("run stopped by user")
    finally:
        store.close()

    write_environment_json(run_dir)

    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "metadata.json"),
        "scalar_log": str(run_dir / "scalar_log.csv"),
    }


__all__ = [
    "LCParams",
    "run_td",
]
