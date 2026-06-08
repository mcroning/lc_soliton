"""
lc_engine_validated.py

Thin orchestration layer for the LC soliton validated runtime.

The trusted GPU physics kernels live in:
    lc_soliton.validated_core.runner_core

This module owns:
    - backend selection
    - parameter/context construction
    - lightweight scalar/movie storage
    - mode dispatch and progress reporting

It deliberately avoids re-implementing the trusted static/TD kernels.
"""
from __future__ import annotations

#from dataclasses import asdict, dataclass
from pathlib import Path
#import csv
import json
#import math
import time
from typing import  Callable, Dict, Optional
from dataclasses import asdict
import numpy as np

try:
    import cupy as _cupy  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cupy = None
    _HAS_CUPY = False

from ..core.backend import (
    _HAS_CUPY,
    _cupy,
    asnumpy,
    free_backend_memory,
    get_backend,
    is_cupy_array,
)
if False:
    from ..core.bias import (
        compute_neff,
        lc_b_from_voltage,
        voltage_from_lc_b,
        theta0_from_b_zero_bc,
        b_from_theta0_zero_bc,
        theta_bias_1d_exact_zero_bc,
        theta_bias_1d_relax_bc,
        theta_bias_2d_from_params,
    )

from ..core.bias import (    
    lc_b_from_voltage,
    voltage_from_lc_b,
    theta0_from_b_zero_bc,
    b_from_theta0_zero_bc,

)

from ..core.context import (
    LCParams,
    LCContext,
    DualGrid,    
    apply_legacy_context_aliases,    
    make_context,    
)

from ..core.storage import LightStore
from ..core.viewer import load_run_arrays
from .static_runner import _run_static

from .td_runner import (
    _run_td_predictor,
    _run_dg_td_experimental,
)

from ..helpers import _prepare_substeps

# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------


def run_lc_validated(
    params: LCParams,
    *,
    run_dir: str | Path,
    mode: str = "strict_static",
    Nt: int = 100,
    dt: float = 5e-4,
    t_stride: int = 1,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Optional[Callable[[str], None]] = print,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    mode = mode.lower().strip()
    if mode not in {"static_fast", "strict_static", "td_predictor_only", "dg_td_predictor"}:
        raise ValueError("mode must be static_fast, strict_static, td_predictor_only, or dg_td_predictor")

    if mode == "dg_td_predictor":
        params = LCParams(**{**asdict(params), "use_dual_grid": True})

    ctx, dg = make_context(params)
    xp = ctx.xp
    apply_legacy_context_aliases(ctx, params)

    ctx.theta_full = xp.stack([ctx.theta_bias_2d.copy() for _ in range(ctx.Nz)], axis=0).astype(xp.float32, copy=False)

    use_legacy_static = _HAS_CUPY and xp is _cupy and mode == "strict_static"
    use_legacy_td = _HAS_CUPY and xp is _cupy and mode == "td_predictor_only"
    use_legacy_dg_td = False
    if mode == "strict_static" and not use_legacy_static:   
        raise RuntimeError(    
            "Validated strict_static currently requires CuPy/GPU. "    
            "CPU physics execution is not validated yet."   
        )
    
    if mode in {"strict_static", "td_predictor_only"} and not (_HAS_CUPY and xp is _cupy):
        if mode == "td_predictor_only":
            raise RuntimeError("Validated TD predictor requires CuPy/GPU. Use backend='auto' on a GPU node.")

    Nt_eff = int(Nt) if mode in {"td_predictor_only", "dg_td_predictor"} else 1
    t_stride = max(1, int(t_stride))
    Nt_out = (Nt_eff + t_stride - 1) // t_stride

    run_dir = Path(run_dir)
    store = LightStore(run_dir, ctx, Nt_out=Nt_out, save_slices=save_slices, save_full=save_full)

    # Metadata uses the same substep selector as the branch that will run.
    nsub_meta, dz_sub_meta, phi_meta, _, _ = _prepare_substeps(ctx, params, use_core=(_HAS_CUPY and xp is _cupy and mode in {"strict_static", "td_predictor_only"}))
    meta = asdict(params)
    meta.update(
        dict(
            mode=mode,
            Nt=Nt_eff,
            dt=dt,
            t_stride=t_stride,
            Nt_out=Nt_out,
            nsub=nsub_meta,
            dz_sub=dz_sub_meta,
            phi_est=phi_meta,
            backend="cupy" if (_HAS_CUPY and xp is _cupy) else "numpy",
            strict_static_kernel="validated_core" if use_legacy_static else "package_smoke",
            td_kernel="validated_core" if use_legacy_td else "package_smoke",
            dg_td_kernel=(
                "experimental_dual_grid"
                if use_legacy_dg_td
                else "disabled"
            ),
        )
    )

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    stopped = False
    try:
        if mode in {"static_fast", "strict_static"}:
            stopped = _run_static(
                ctx=ctx,
                params=params,
                store=store,
                progress=progress,
                should_stop=should_stop,
                use_legacy_static=use_legacy_static,
                save_full=save_full,
            )
        elif mode == "td_predictor_only":
            stopped = _run_td_predictor(
                ctx=ctx,
                params=params,
                store=store,
                Nt=Nt_eff,
                dt=dt,
                t_stride=t_stride,
                progress=progress,
                should_stop=should_stop,
            )
        else:
            if dg is None:
                raise RuntimeError("Dual-grid mode requested but no dual-grid context was built.")
            stopped = _run_dg_td_experimental(
                ctx=ctx,
                dg=dg,
                params=params,
                store=store,
                Nt=Nt_eff,
                dt=dt,
                t_stride=t_stride,
                progress=progress,
                should_stop=should_stop,
            )

        np.savez(run_dir / "final_summary.npz", elapsed_s=time.time() - t0, stopped=stopped)
        if stopped and progress:
            progress("run stopped by user")
    finally:
        store.close()

    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "metadata.json"),
        "scalar_log": str(run_dir / "scalar_log.csv"),
    }




__all__ = [
    "LCParams",
    "LCContext",
    "DualGrid",
    "get_backend",
    "run_lc_validated",
    "load_run_arrays",
    "lc_b_from_voltage",
    "voltage_from_lc_b",
    "theta0_from_b_zero_bc",
    "b_from_theta0_zero_bc",
]
