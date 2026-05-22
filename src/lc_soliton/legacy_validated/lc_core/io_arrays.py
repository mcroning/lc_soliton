# lc_core/io_arrays.py
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def save_array(arr, path, *, dtype="float32"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = _to_numpy(arr)

    if dtype is not None:
        a = a.astype(dtype, copy=False)

    np.save(path, a)
    return path


def save_static_arrays(
    res_z,
    run_dir,
    *,
    save_theta=True,
    save_I_mid=True,
    save_final_amp=False,
    theta_dtype="float32",
    I_dtype="float16",
    amp_dtype="complex64",
):
    run_dir = Path(run_dir)
    arr_dir = run_dir / "arrays"
    arr_dir.mkdir(parents=True, exist_ok=True)

    manifest = dict(arrays={})

    if save_theta:
        path = save_array(
            res_z.theta_full,
            arr_dir / "theta_full.npy",
            dtype=theta_dtype,
        )
        manifest["arrays"]["theta_full"] = dict(
            path=str(path.relative_to(run_dir)),
            dtype=theta_dtype,
            description="Static solved theta_full[z,x,y]",
        )

    if save_I_mid:
        path = save_array(
            res_z.I_mid_store,
            arr_dir / "I_mid_store.npy",
            dtype=I_dtype,
        )
        manifest["arrays"]["I_mid_store"] = dict(
            path=str(path.relative_to(run_dir)),
            dtype=I_dtype,
            description="Self-consistent midpoint intensity I_mid[z,x,y]",
        )

    if save_final_amp and hasattr(res_z, "final_amp"):
        path = save_array(
            res_z.final_amp,
            arr_dir / "final_amp.npy",
            dtype=amp_dtype,
        )
        manifest["arrays"]["final_amp"] = dict(
            path=str(path.relative_to(run_dir)),
            dtype=amp_dtype,
            description="Final optical field after static z march",
        )

    manifest_path = arr_dir / "array_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest