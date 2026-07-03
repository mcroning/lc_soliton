"""Backend and precision helpers.

This module chooses NumPy or CuPy and provides small backend-independent
conversion helpers. It contains no LC physics, beam parameters, workflows, or
algorithm logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = Any


@dataclass(frozen=True)
class BackendSpec:
    """Backend and precision choices."""

    backend: str = "auto"      # "auto", "numpy", or "cupy"
    precision: str = "float32" # "float32" or "float64"
    verbose: bool = True

    def validate(self) -> None:
        if self.backend not in ("auto", "numpy", "cupy"):
            raise ValueError("backend must be 'auto', 'numpy', or 'cupy'")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")


@dataclass(frozen=True)
class Backend:
    """Resolved runtime backend."""

    xp: Any
    name: str
    real_dtype: Any
    complex_dtype: Any
    is_gpu: bool = False

    def summary(self) -> dict[str, str | bool]:
        return {
            "backend": self.name,
            "real_dtype": str(np.dtype(self.real_dtype)),
            "complex_dtype": str(np.dtype(self.complex_dtype)),
            "is_gpu": bool(self.is_gpu),
        }


def dtype_pair(precision: str) -> tuple[Any, Any]:
    """Return real/complex dtype pair for precision string."""
    if precision == "float32":
        return np.float32, np.complex64
    if precision == "float64":
        return np.float64, np.complex128
    raise ValueError("precision must be 'float32' or 'float64'")


def get_backend(spec: BackendSpec | str = BackendSpec(), *, precision: str | None = None) -> Backend:
    """Resolve NumPy/CuPy backend.

    Parameters
    ----------
    spec
        BackendSpec or backend name string.
    precision
        Optional precision override when ``spec`` is a string.
    """

    if isinstance(spec, str):
        spec = BackendSpec(backend=spec, precision="float32" if precision is None else precision)
    elif precision is not None:
        spec = BackendSpec(backend=spec.backend, precision=precision, verbose=spec.verbose)

    spec.validate()
    real_dtype, complex_dtype = dtype_pair(spec.precision)

    if spec.backend == "numpy":
        if spec.verbose:
            print("[lc.backend] using numpy")
        return Backend(np, "numpy", real_dtype, complex_dtype, False)

    if spec.backend in ("auto", "cupy"):
        try:
            import cupy as cp  # type: ignore

            # Allocation test catches broken GPU sessions.
            _ = cp.arange(1)
            if spec.verbose:
                dev = cp.cuda.Device()
                print(f"[lc.backend] using cupy device={dev.id}")
            return Backend(cp, "cupy", real_dtype, complex_dtype, True)
        except Exception as e:
            if spec.backend == "cupy":
                raise RuntimeError(f"requested cupy backend but CuPy/GPU is unavailable: {e}") from e
            if spec.verbose:
                print(f"[lc.backend] cupy unavailable, falling back to numpy: {e}")

    return Backend(np, "numpy", real_dtype, complex_dtype, False)


def asnumpy(a: Array) -> np.ndarray:
    """Convert NumPy/CuPy array or scalar to NumPy."""
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def scalar_float(x: Any) -> float:
    """Backend-independent conversion to Python float."""
    return float(asnumpy(x))


def synchronize(xp: Any) -> None:
    """Synchronize backend if it is CuPy."""
    if getattr(xp, "__name__", "") == "cupy":
        xp.cuda.Stream.null.synchronize()


def free_memory(xp: Any) -> None:
    """Best-effort backend memory cleanup."""
    if getattr(xp, "__name__", "") == "cupy":
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()


__all__ = [
    "Array",
    "BackendSpec",
    "Backend",
    "dtype_pair",
    "get_backend",
    "asnumpy",
    "scalar_float",
    "synchronize",
    "free_memory",
]
