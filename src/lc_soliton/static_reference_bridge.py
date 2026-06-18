"""
Faithful wrapper for the legacy strict static z-march bridge used to
generate the stored trusted reference case.

Legacy-only reference bridge.

This module intentionally imports lc_offload_tools to reproduce the original
trusted strict-static reference path. It is not used by the active static
runner.
"""

from __future__ import annotations

from pathlib import Path

from .legacy_validated.lc_core.pipeline import run_static_experiment
from .legacy_validated.lc_core.launch import build_launch_field_gpu
from .legacy_validated.lc_offload_tools import (
    build_theta_bias_IC,
    build_theta_bias_IC_dirichlet_value,
    compute_n_bg_from_bias,
    genrot,
    build_amp_pair,
    intens,
    choose_optics_substeps,
    get_h_for_dz,
    prepare_cn_ky_operator,
    strict_static_relax_slice_selfconsistent,
)


def run_static_reference_bridge(
    cfg,
    *,
    run_root: str | Path,
    save_arrays: bool = False,
):
    """
    Run the original validated strict-static bridge path.

    This reproduces the path used by the trusted strict-static reference.
    """

    return run_static_experiment(
        cfg,
        run_root=run_root,
        tag="strict_static_replay",
        save_arrays=save_arrays,

        build_context_kwargs=dict(
            build_theta_bias_IC=build_theta_bias_IC,
            build_theta_bias_IC_dirichlet_value=build_theta_bias_IC_dirichlet_value,
            compute_n_bg_from_bias=compute_n_bg_from_bias,
            build_launch_field_gpu=build_launch_field_gpu,
            genrot=genrot,
            build_amp_pair=build_amp_pair,
            intens=intens,
        ),

        static_bridge_kwargs=dict(
            choose_optics_substeps=choose_optics_substeps,
            get_h_for_dz_legacy=get_h_for_dz,
            prepare_cn_ky_operator=prepare_cn_ky_operator,
            strict_static_relax_slice_selfconsistent=strict_static_relax_slice_selfconsistent,
            strict_residual_tol_max=6e-2,
            strict_residual_tol_rms=1e-2,
            strict_max_outer_passes=8,
            verbose_every=25,
            tqdm_z=False,
        ),
    )


__all__ = ["run_static_reference_bridge"]
