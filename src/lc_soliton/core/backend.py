from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as _cupy  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cupy = None
    _HAS_CUPY = False
xp_default = _cupy if _HAS_CUPY else np

def get_backend(name="auto", verbose=True):
    if name in ("numpy", "cpu"):
        info = dict(backend="numpy", device="CPU", gpu_name=None, gpu_mem_gb=None)

        if verbose:
            print("=" * 60)
            print("LC backend")
            print("  backend : NumPy")
            print("  device  : CPU")
            print("=" * 60)

        return np, info

    if name in ("auto", "cupy", "gpu"):
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                dev = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                gpu_name = props["name"].decode()
                mem_gb = props["totalGlobalMem"] / 1024**3

                info = dict(
                    backend="cupy",
                    device=f"GPU:{dev.id}",
                    gpu_name=gpu_name,
                    gpu_mem_gb=float(mem_gb),
                )

                if verbose:
                    print("=" * 60)
                    print("LC backend")
                    print("  backend : CuPy")
                    print(f"  device  : GPU {dev.id}")
                    print(f"  GPU     : {gpu_name}")
                    print(f"  memory  : {mem_gb:.1f} GB")
                    print("=" * 60)

                return cp, info

        except Exception as exc:
            if verbose:
                print(f"[backend fallback] CuPy unavailable: {exc}")

        info = dict(backend="numpy", device="CPU", gpu_name=None, gpu_mem_gb=None)

        if verbose:
            print("=" * 60)
            print("LC backend")
            print("  backend : NumPy")
            print("  device  : CPU")
            print("  note    : GPU backend unavailable")
            print("=" * 60)

        return np, info

    raise ValueError(f"Unknown backend {name!r}. Use auto, cupy/gpu, or numpy/cpu.")


def is_cupy_array(a: Any) -> bool:
    return _HAS_CUPY and isinstance(a, _cupy.ndarray)


def asnumpy(a: Any) -> np.ndarray:
    if is_cupy_array(a):
        return _cupy.asnumpy(a)
    return np.asarray(a)


def free_backend_memory(xp) -> None:
    if _HAS_CUPY and xp is _cupy:
        _cupy.get_default_memory_pool().free_all_blocks()
        _cupy.get_default_pinned_memory_pool().free_all_blocks()


__all__ = [
    "_HAS_CUPY",
    "_cupy",
    "xp_default",
    "get_backend",
    "is_cupy_array",
    "asnumpy",
    "free_backend_memory",
]