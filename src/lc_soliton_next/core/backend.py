"""Backend selection utilities for :mod:`lc_soliton_next`.

This module owns the NumPy/CuPy choice.  Numerical engines should accept an
``xp`` module from context/workflows rather than importing CuPy directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

BackendName = Literal["auto", "numpy", "cpu", "cupy", "gpu"]

try:  # optional dependency
    import cupy as _cupy  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cupy = None
    _HAS_CUPY = False

xp_default = _cupy if _HAS_CUPY else np


@dataclass(frozen=True)
class BackendInfo:
    """Description of the selected numerical backend."""

    backend: str
    device: str
    gpu_name: str | None = None
    gpu_mem_gb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_mem_gb": self.gpu_mem_gb,
        }


def _print_backend(info: BackendInfo, *, note: str | None = None) -> None:
    print("=" * 60)
    print("LC backend")
    print(f"  backend : {info.backend}")
    print(f"  device  : {info.device}")
    if info.gpu_name is not None:
        print(f"  GPU     : {info.gpu_name}")
    if info.gpu_mem_gb is not None:
        print(f"  memory  : {info.gpu_mem_gb:.1f} GB")
    if note:
        print(f"  note    : {note}")
    print("=" * 60)


def get_backend(name: BackendName = "auto", *, verbose: bool = True):
    """Return ``(xp, info)`` for the requested array backend.

    Parameters
    ----------
    name:
        ``"numpy"``/``"cpu"`` forces NumPy.  ``"cupy"``/``"gpu"`` requires a
        CUDA device.  ``"auto"`` tries CuPy first and falls back to NumPy.
    verbose:
        Print a short backend banner.

    Returns
    -------
    xp, info:
        ``xp`` is either :mod:`numpy` or :mod:`cupy`.  ``info`` is a plain dict
        for convenient JSON serialization.
    """

    name = str(name).lower()

    if name in ("numpy", "cpu"):
        info = BackendInfo(backend="numpy", device="CPU")
        if verbose:
            _print_backend(info)
        return np, info.as_dict()

    if name not in ("auto", "cupy", "gpu"):
        raise ValueError(f"unknown backend {name!r}; use auto, cupy/gpu, or numpy/cpu")

    try:
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("CuPy imported but no CUDA devices were found")

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        raw_name = props.get("name", b"GPU")
        gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        mem_gb = float(props.get("totalGlobalMem", 0)) / 1024**3

        info = BackendInfo(
            backend="cupy",
            device=f"GPU:{dev.id}",
            gpu_name=gpu_name,
            gpu_mem_gb=mem_gb,
        )
        if verbose:
            _print_backend(info)
        return cp, info.as_dict()

    except Exception as exc:
        if name in ("cupy", "gpu"):
            raise RuntimeError(f"requested CuPy backend but it is unavailable: {exc}") from exc

        info = BackendInfo(backend="numpy", device="CPU")
        if verbose:
            _print_backend(info, note=f"CuPy unavailable: {exc}")
        return np, info.as_dict()


def is_cupy_array(a: Any) -> bool:
    """Return True when ``a`` is a CuPy ndarray."""

    return bool(_HAS_CUPY and _cupy is not None and isinstance(a, _cupy.ndarray))


def asnumpy(a: Any) -> np.ndarray:
    """Return ``a`` as a NumPy array, copying from GPU when needed."""

    if is_cupy_array(a):
        return _cupy.asnumpy(a)  # type: ignore[union-attr]
    return np.asarray(a)


def free_backend_memory(xp: Any) -> None:
    """Release cached GPU memory if ``xp`` is CuPy; no-op for NumPy."""

    if _HAS_CUPY and _cupy is not None and xp is _cupy:
        _cupy.get_default_memory_pool().free_all_blocks()
        _cupy.get_default_pinned_memory_pool().free_all_blocks()


__all__ = [
    "BackendInfo",
    "BackendName",
    "_HAS_CUPY",
    "_cupy",
    "xp_default",
    "get_backend",
    "is_cupy_array",
    "asnumpy",
    "free_backend_memory",
]
