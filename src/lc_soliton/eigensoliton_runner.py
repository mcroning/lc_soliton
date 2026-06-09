
"""
Public eigensoliton execution API.

This module wraps the validated nonlinear LC optical eigensoliton solver
without changing the trusted numerical core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .request import SimulationRequest, save_request
from .request_translate import request_to_lcparams_kwargs
from .core.context import LCParams, make_context, apply_legacy_context_aliases
from .validated_core.eigenmode_core import (
    make_mode_seed,
    lc_eigensoliton_existence_curve_v2,
)


def _prdata_from_request(request: SimulationRequest) -> dict[str, Any]:
    """
    Minimal physical parameter dictionary expected by the validated
    eigensoliton continuation code.
    """
    return {
        "d": float(request.geometry.xaper_um),
        "K": float(request.material.K),
        "ne": float(request.material.ne),
        "no": float(request.material.no),
        "De": float(request.material.De),
        "wavelength_um": float(request.geometry.wavelength_um),
    }


def run_eigensoliton_existence_curve(
    request: SimulationRequest,
    powers_mW: Sequence[float],
    *,
    progress_callback=None,
    run_dir: str | Path | None = None,
    branch_name: str = "fundamental",
    mode_seed: str = "00",
    w0_um: float | None = None,
    checkpoint_prefix: str = "lc_eigensoliton",
    save_profiles: bool = True,
    live_plot: bool = False,
    solve_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a nonlinear LC optical eigensoliton continuation over input power.

    This is not a static propagation sweep. It solves the coupled nonlinear
    eigenproblem for A(x,y), theta(x,y), and beta at each branch point.

    Returns
    -------
    dict
        Lightweight run summary including the CSV path and final dataframe.
    """
    if run_dir is None:
        run_dir = request.output.run_dir

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    powers = np.asarray([float(p) for p in powers_mW], dtype=float)

    param_data = request_to_lcparams_kwargs(request)
    params = LCParams(**param_data)

    # The eigensoliton continuation is transverse/eigenmode machinery.
    # It does not need a propagation Nz, but keeping request Nz is harmless.
    ctx, _dg = make_context(params)
    apply_legacy_context_aliases(ctx, params)

    prdata = _prdata_from_request(request)

    if w0_um is None:
        w0_um = float(getattr(request.launch, "waist_x_um", None) or 4.0)

    A_seed, theta_seed = make_mode_seed(ctx, mode=mode_seed, w0_um=float(w0_um))

    save_request(request, run_dir / "request.json")

    kwargs = dict(solve_kwargs or {})

    df, state = lc_eigensoliton_existence_curve_v2(
        ctx,
        prdata,
        powers,
        A_seed,
        theta_seed,
        branch_name=branch_name,
        save_dir=run_dir,
        checkpoint_prefix=checkpoint_prefix,
        save_profiles=save_profiles,
        live_plot=live_plot,
        progress_callback=progress_callback,
        **kwargs,
    )

    csv_path = run_dir / f"{checkpoint_prefix}_partial.csv"

    return {
        "workflow": "eigensoliton_existence_curve",
        "run_dir": str(run_dir),
        "csv": str(csv_path),
        "n_completed": int(len(df)),
        "powers_mW": powers.tolist(),
        "dataframe": df,
        "state": state,
    }


def run_eigensoliton_case(*args, **kwargs):
    """
    Backward-compatible placeholder name.
    """
    return run_eigensoliton_existence_curve(*args, **kwargs)


__all__ = [
    "run_eigensoliton_existence_curve",
    "run_eigensoliton_case",
]
