import json

from lc_soliton import load_reference_case
from lc_soliton.static_reference_bridge import run_static_reference_bridge


from lc_soliton.legacy_validated.lc_core.config import config_from_prdata

import pytest

try:
    import cupy
    cupy.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip(
        "CUDA runtime unavailable; legacy strict-static bridge replay is GPU-only",
        allow_module_level=True,
    )
@pytest.mark.xfail(
    reason="Current run_engine static path does not reproduce trusted optical confinement.",
    strict=True,

)
def test_legacy_static_bridge_replays_trusted_reference(tmp_path):
    reference_data = load_reference_case("strict_static_centroid_drift")
    trusted = reference_data["trusted_metrics"]

    cfg = config_from_prdata(reference_data["config"]["legacy_prdata"])

    ctx, stores, launch, res, run_dir = run_static_reference_bridge(
        cfg,
        run_root=tmp_path,
        save_arrays=False,
    )

    summary = res.summary

    assert summary["converged_count"] == 50
    assert summary["failed_count"] == 0

    assert abs(summary["P_final"] - trusted["P_final"]) < 1e-6
    assert abs(summary["max_residual_rms"] - trusted["max_residual_rms"]) < 1e-6
    assert abs(summary["max_residual_max"] - trusted["max_residual_max"]) < 1e-6
