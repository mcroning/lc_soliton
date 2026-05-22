from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def save_td_movies(td_res, run_dir, *, dtype="float32"):
    run_dir = Path(run_dir)
    movie_dir = run_dir / "movies"
    movie_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}

    for name in ("Ixz", "Iyz", "thetaxz", "thetayz"):
        arr = getattr(td_res.stores, name, None)
        if arr is None:
            continue

        path = movie_dir / f"{name}.npy"
        np.save(path, _to_numpy(arr).astype(dtype, copy=False))

        manifest[name] = dict(
            path=str(path.relative_to(run_dir)),
            dtype=dtype,
            shape=list(arr.shape),
        )

    manifest_path = movie_dir / "movie_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest