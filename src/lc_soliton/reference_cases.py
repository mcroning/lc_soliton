"""
Public reference-case API.
"""

from __future__ import annotations

from pathlib import Path
import json


def get_reference_case_dir(name: str, repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path.cwd()
    else:
        repo_root = Path(repo_root)

    case_dir = repo_root / "validation" / "reference_cases" / name

    if not case_dir.exists():
        raise FileNotFoundError(f"Reference case not found: {case_dir}")

    return case_dir


def load_reference_case(name: str, repo_root: str | Path | None = None) -> dict:
    """
    Load stored reference-case metadata and trusted metrics.
    """
    case_dir = get_reference_case_dir(name, repo_root=repo_root)

    data = {
        "name": name,
        "case_dir": str(case_dir),
    }

    for filename, key in [
        ("trusted_metrics.json", "trusted_metrics"),
        ("static_z_summary.json", "summary"),
        ("static_z_reports.json", "reports"),
        ("config.json", "config"),
    ]:
        path = case_dir / filename
        if path.exists():
            data[key] = json.loads(path.read_text())

    data["figures"] = {
        "xz_reference": str(case_dir / "xz_reference.png"),
        "centroid_reference": str(case_dir / "centroid_reference.png"),
    }

    return data


__all__ = [
    "get_reference_case_dir",
    "load_reference_case,
    "summarize_reference_case",
]


def summarize_reference_case(name: str, repo_root: str | Path | None = None) -> dict:
    """
    Return a compact summary of a stored reference case.
    """
    data = load_reference_case(name, repo_root=repo_root)

    summary = {
        "name": data["name"],
        "case_dir": data["case_dir"],
        "has_trusted_metrics": "trusted_metrics" in data,
        "has_summary": "summary" in data,
        "has_reports": "reports" in data,
        "has_config": "config" in data,
        "figures": data.get("figures", {}),
    }

    if "trusted_metrics" in data:
        tm = data["trusted_metrics"]
        for key in ["max_residual_rms", "max_residual_max", "P_final"]:
            if key in tm:
                summary[key] = tm[key]

    return summary
