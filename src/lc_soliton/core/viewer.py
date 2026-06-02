from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

# -----------------------------------------------------------------------------
# Viewer helpers
# -----------------------------------------------------------------------------


def load_run_arrays(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    Nx, Ny, Nz = int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])
    Nt_out = int(meta.get("Nt_out", (int(meta.get("Nt", 1)) + int(meta.get("t_stride", 1)) - 1) // int(meta.get("t_stride", 1))))
    out = {"metadata": meta, "run_dir": run_dir}
    for name, shape in {
        "Ixz": (Nt_out, Nz, Nx),
        "Iyz": (Nt_out, Nz, Ny),
        "dthetaxz": (Nt_out, Nz, Nx),
        "dthetayz": (Nt_out, Nz, Ny),
    }.items():
        path = run_dir / f"{name}.dat"
        if path.exists():
            out[name] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    return out

__all__ = ["load_run_arrays"]