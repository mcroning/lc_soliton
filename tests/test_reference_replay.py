import json

from lc_soliton import load_reference_case, run_engine
from lc_soliton.reference_translate import reference_config_to_request


def test_strict_static_reference_replays_through_engine(tmp_path):
    reference_data = load_reference_case("strict_static_centroid_drift")
    trusted = reference_data["trusted_metrics"]

    run_dir = tmp_path / "strict_static_replay"

    req = reference_config_to_request(
        reference_data,
        run_dir=run_dir,
    )

    req.runtime.progress = True
    req.runtime.backend = "auto"

    result = run_engine(req)

    import pandas as pd

    scalar_log = run_dir / "scalar_log.csv"
    assert scalar_log.exists()
    
    df = pd.read_csv(scalar_log)
    
    print("SCALAR COLUMNS:", list(df.columns))
    print(df.tail())
    
    summary = {
        "converged_count": int(df["converged"].sum()) if "converged" in df.columns else len(df),
        "failed_count": int((~df["converged"]).sum()) if "converged" in df.columns else 0,
        "P_final": float(df["power"].iloc[-1]) if "power" in df.columns else float(df["P"].iloc[-1]),
        "max_residual_rms": float(df["rrms"].max()) if "rrms" in df.columns else float(df["rms_interior"].max()),
        "max_residual_max": float(df["rmax"].max()) if "rmax" in df.columns else float(df["max_interior"].max()),
    }
    assert df["Imax"].iloc[-1] / df["Imax"].iloc[0] > 0.5
