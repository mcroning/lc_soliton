"""Persistence helpers for LC results.

This module saves compact run summaries, sample tables, metadata, and optional
arrays. It does not run algorithms or construct experiments.

The functions are intentionally small and conservative:
* JSON for human-readable metadata/summaries.
* CSV for scalar sample rows.
* NPZ for numerical arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from ..numerics.backend import asnumpy

Array = Any


def _json_default(obj: Any) -> Any:
    """JSON serializer for NumPy scalars/arrays and Paths."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        return asnumpy(obj).tolist()
    except Exception:
        return str(obj)


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Any) -> Path:
    """Save JSON with stable formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")
    return p


def load_json(path: str | Path) -> Any:
    """Load JSON."""
    return json.loads(Path(path).read_text())


def save_samples_csv(path: str | Path, samples: list[dict]) -> Path:
    """Save list-of-dict scalar samples to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not samples:
        p.write_text("")
        return p

    keys: list[str] = []
    for row in samples:
        for k in row:
            if k not in keys:
                keys.append(k)

    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in samples:
            writer.writerow({k: row.get(k, "") for k in keys})

    return p


def save_arrays_npz(path: str | Path, **arrays: Array) -> Path:
    """Save arrays to compressed NPZ after converting backend arrays to NumPy."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {k: asnumpy(v) for k, v in arrays.items()}
    np.savez_compressed(p, **out)
    return p


def save_run_summary(
    run_dir: str | Path,
    result: Any,
    *,
    metadata: dict | None = None,
    arrays: dict[str, Array] | None = None,
) -> dict[str, str]:
    """Save a RunSummary-like object to a run directory.

    Expected result attributes:
        kind
        metrics
        samples
    """
    d = ensure_dir(run_dir)

    files: dict[str, str] = {}

    summary_data = {
        "kind": getattr(result, "kind", "RunSummary"),
        "metrics": getattr(result, "metrics", {}),
    }
    files["summary_json"] = str(save_json(d / "summary.json", summary_data))

    samples = list(getattr(result, "samples", []))
    files["samples_csv"] = str(save_samples_csv(d / "samples.csv", samples))

    if metadata is not None:
        files["metadata_json"] = str(save_json(d / "metadata.json", metadata))

    if arrays:
        files["arrays_npz"] = str(save_arrays_npz(d / "arrays.npz", **arrays))

    return files


__all__ = [
    "Array",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_samples_csv",
    "save_arrays_npz",
    "save_run_summary",
]
