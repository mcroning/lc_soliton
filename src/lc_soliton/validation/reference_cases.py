"""
Reference-case validation helpers.
"""

from __future__ import annotations

from pathlib import Path
import json
from ..engine import run_engine
def reference_config_to_request(reference_data, run_dir):
    cfg = reference_data["config"]

    return SimulationRequest(
        mode="static",
        grid=GridRequest(
            Nx=int(cfg["grid"]["Nx"]),
            Ny=int(cfg["grid"]["Ny"]),
            Nz=int(cfg["grid"]["Nz"]),
        ),
        geometry=GeometryRequest(
            xaper_um=float(cfg["grid"]["xaper_um"]),
            yaper_um=float(cfg["grid"]["yaper_um"]),
            dz_um=float(cfg["grid"]["dz_um"]),
            wavelength_um=float(cfg["grid"]["lm_um"]),
        ),
        material=MaterialRequest(
            ne=float(cfg["material"]["ne"]),
            no=float(cfg["material"]["no"]),
            b=float(cfg["material"]["b"]),
            bi=float(cfg["material"]["bi"]),
            mobility=float(cfg["material"].get("mobility", 1.0)),
            theta_bc=float(cfg["material"].get("theta_bc", 0.0)),
            theta_z_gamma=float(cfg["material"].get("theta_z_gamma", 0.0)),
        ),
        launch=LaunchRequest(
            waist_x_um=float(cfg["launch"]["w0x1_um"]),
            waist_y_um=float(cfg["launch"]["w0y1_um"]),
            separation_um=float(cfg["launch"].get("soliton_pair_sep_um", 0.0)),
            coherent=bool(cfg["launch"].get("coh", False)),
        ),
        boundary=BoundaryRequest(
            use_sponge=bool(cfg["boundary"].get("use_sponge", True)),
            windowedge=float(cfg["boundary"].get("windowedge", 0.1)),
        ),
        solver=SolverRequest(
            static_max_steps=int(cfg["static"].get("static_max_steps", 2000)),
            Nt=1,
            dt=float(cfg["time"].get("dt", 0.02)),
            t_stride=int(cfg["time"].get("t_stride", 1)),
        ),
        output=OutputRequest(
            run_dir=str(run_dir),
            save_slices=True,
            save_full=False,
        ),
        runtime=RuntimeRequest(
            backend="auto",
            progress=True,
        ),
    )


def validate_strict_static_centroid_drift(
    repo_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> dict:
    """
    Re-run the strict-static centroid-drift reference case through engine.py
    and compare the new result against the stored trusted metrics.
    """
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    case_dir = (
        repo_root
        / "validation"
        / "reference_cases"
        / "strict_static_centroid_drift"
    )

    required = [
        "README.md",
        "trusted_metrics.json",
        "xz_reference.png",
        "centroid_reference.png",
        "config.json",
        "static_z_summary.json",
        "static_z_reports.json",
    ]

    missing = [name for name in required if not (case_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing reference files: {missing}")

    trusted = json.loads((case_dir / "trusted_metrics.json").read_text())
    reference_data = {
        "config": json.loads((case_dir / "config.json").read_text()),
        "trusted_metrics": trusted,
        "case_dir": str(case_dir),
    }

    if run_dir is None:
        run_dir = repo_root / "runs" / "reference_validation" / "strict_static_centroid_drift"
    else:
        run_dir = Path(run_dir)

    request = reference_config_to_request(reference_data, run_dir=run_dir)

    result = run_engine(request)

    summary_path = run_dir / "static_z_summary.json"
    metadata_path = run_dir / "metadata.json"

    if summary_path.exists():
        new_summary = json.loads(summary_path.read_text())
    elif isinstance(result, dict):
        new_summary = result
    else:
        raise RuntimeError(
            "Reference validation run completed, but no static_z_summary.json "
            "was found and engine result was not a dict."
        )

    checks = {
        "max_residual_rms": (
            new_summary.get("max_residual_rms"),
            trusted["max_residual_rms"],
        ),
        "max_residual_max": (
            new_summary.get("max_residual_max"),
            trusted["max_residual_max"],
        ),
        "P_final": (
            new_summary.get("P_final"),
            trusted["P_final"],
        ),
    }

    tolerances = {
        "max_residual_rms": 5e-6,
        "max_residual_max": 5e-5,
        "P_final": 5e-5,
    }

    failures = {}

    for key, (actual, expected) in checks.items():
        if actual is None:
            failures[key] = {
                "actual": None,
                "expected": expected,
                "reason": "missing from new summary",
            }
            continue

        abs_err = abs(float(actual) - float(expected))

        if abs_err > tolerances[key]:
            failures[key] = {
                "actual": actual,
                "expected": expected,
                "abs_err": abs_err,
                "tol": tolerances[key],
            }

    if failures:
        raise AssertionError(
            "Strict-static centroid-drift reference validation failed: "
            + json.dumps(failures, indent=2)
        )

    return {
        "case": "strict_static_centroid_drift",
        "passed": True,
        "run_dir": str(run_dir),
        "trusted_metrics": trusted,
        "new_summary": new_summary,
        "metadata_path": str(metadata_path) if metadata_path.exists() else None,
    }
