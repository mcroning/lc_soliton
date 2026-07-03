#!/usr/bin/env python
"""53: equivalence and timing for lc.algorithms.thomas_fast."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def asnumpy(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def rel_l2(a, b):
    a = asnumpy(a)
    b = asnumpy(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def max_abs(a, b):
    return float(np.max(np.abs(asnumpy(a) - asnumpy(b))))


def sync(xp):
    if getattr(xp, "__name__", "") == "cupy":
        xp.cuda.Stream.null.synchronize()


def time_call(fn, xp, reps: int):
    sync(xp)
    t0 = time.perf_counter()
    out = None
    for _ in range(int(reps)):
        out = fn()
    sync(xp)
    return time.perf_counter() - t0, out


def main():
    _ensure_src_on_path()

    from lc.algorithms.thomas import solve_const_offdiag_batched
    from lc.algorithms.thomas_fast import solve_const_offdiag_batched_fast

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n", type=int, default=254)
    ap.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--tol", type=float, default=None)
    args = ap.parse_args()

    import cupy as cp

    dtype = cp.complex64 if args.dtype == "complex64" else cp.complex128
    tol = args.tol
    if tol is None:
        tol = 5e-6 if args.dtype == "complex64" else 5e-12

    rng = np.random.default_rng(53)
    batch = int(args.batch)
    n = int(args.n)

    lower = -0.08 + 0.0j
    upper = -0.08 + 0.0j

    diag_np = (1.2 + 0.2 * rng.random(batch)).astype(np.float32 if args.dtype == "complex64" else np.float64)
    diag_np = diag_np.astype(np.complex64 if args.dtype == "complex64" else np.complex128)

    rhs_np = rng.normal(size=(batch, n)) + 1j * rng.normal(size=(batch, n))
    rhs_np = rhs_np.astype(np.complex64 if args.dtype == "complex64" else np.complex128)

    diag = cp.asarray(diag_np)
    rhs = cp.asarray(rhs_np)

    ref = solve_const_offdiag_batched(lower, diag, upper, rhs, xp=cp)
    fast = solve_const_offdiag_batched_fast(lower, diag, upper, rhs)

    err_rel = rel_l2(fast, ref)
    err_abs = max_abs(fast, ref)

    if err_rel > tol:
        raise AssertionError(f"fast Thomas differs from reference: rel={err_rel}, tol={tol}")

    t_ref, _ = time_call(lambda: solve_const_offdiag_batched(lower, diag, upper, rhs, xp=cp), cp, args.reps)
    t_fast, _ = time_call(lambda: solve_const_offdiag_batched_fast(lower, diag, upper, rhs), cp, args.reps)

    print("53_thomas_fast_equivalence")
    print(f"  batch              : {batch}")
    print(f"  n                  : {n}")
    print(f"  dtype              : {args.dtype}")
    print(f"  reps               : {args.reps}")
    print(f"  rel_l2             : {err_rel:.16e}")
    print(f"  max_abs            : {err_abs:.16e}")
    print(f"  reference_s        : {t_ref:.9g}")
    print(f"  fast_s             : {t_fast:.9g}")
    print(f"  speedup            : {t_ref / max(t_fast, 1e-300):.6g}")
    print("53_thomas_fast_equivalence: PASS")


if __name__ == "__main__":
    main()
