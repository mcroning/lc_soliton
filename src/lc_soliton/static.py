"""
Public static-propagation API.

This module defines the stable user-facing names for static LC propagation.
The current implementation delegates to validated legacy code while the
internals are being cleaned up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import time
from dataclasses import asdict

import numpy as np
from .core.context import LCParams, make_context, apply_legacy_context_aliases
from .legacy_validated.static_runner import _run_static
from .core.storage import LightStore
from .legacy_validated.runner_utils import _prepare_substeps
from .environment import write_environment_json


def run_static(
    params: LCParams,
    *,
    run_dir: str | Path,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Callable[[str], None] | None = print,
    should_stop: Callable[[], bool] | None = None,
    strict_max_outer_passes: int = 8,
) -> dict[str, Any]:
    t0 = time.time()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx, _dg = make_context(params)
    xp = ctx.xp
    apply_legacy_context_aliases(ctx, params)

    ctx.theta_full = xp.stack(
        [ctx.theta_bias_2d.copy() for _ in range(ctx.Nz)],
        axis=0,
    ).astype(xp.float32, copy=False)

    if False:   ####temp
        launch_profile_path = getattr(params, "launch_profile_path", None)
        if launch_profile_path:
            mode_profile = load_verified_profile(launch_profile_path)
            theta_source = getattr(params, "launch_profile_theta_source", "Saved eigensoliton θ")
            install_launch_profile_into_ctx(
                ctx,
                mode_profile,
                use_saved_theta=(theta_source == "Saved eigensoliton θ"),
            )

    Nt_out = 1
    store = LightStore(run_dir, ctx, Nt_out=Nt_out, save_slices=save_slices, save_full=save_full)

    nsub_meta, dz_sub_meta, phi_meta, _, _ = _prepare_substeps(
        ctx,
        params,
        use_core=False,   # CPU/GPU static_runner now handles core path
    )

    meta = asdict(params)
    meta.update(
        dict(
            mode="strict_static",
            Nt=1,
            dt=0.0,
            t_stride=1,
            Nt_out=1,
            nsub=nsub_meta,
            dz_sub=dz_sub_meta,
            phi_est=phi_meta,
            backend="cupy" if getattr(ctx.xp, "__name__", "") == "cupy" else "numpy",
            strict_static_kernel="validated_core",
        )
    )

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    try:
        stopped = _run_static(
            ctx=ctx,
            params=params,
            store=store,
            progress=progress,
            should_stop=should_stop,
            use_legacy_static=False,
            save_full=save_full,
            strict_max_outer_passes=int(strict_max_outer_passes),
        )

        np.savez(run_dir / "final_summary.npz", elapsed_s=time.time() - t0, stopped=stopped)
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
    "run_static",
]
