"""Fast CuPy Thomas solver for batched constant-offdiagonal systems."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .thomas import solve_const_offdiag_batched

Array = Any


def _is_cupy_array(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "cupy"


_CUDA_SRC_COMPLEX64 = r"""
__device__ inline float2 c64_mul(float2 x, float2 y) {
    return make_float2(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x);
}
__device__ inline float2 c64_sub(float2 x, float2 y) {
    return make_float2(x.x-y.x, x.y-y.y);
}
__device__ inline float2 c64_div(float2 x, float2 y) {
    float den = y.x*y.x + y.y*y.y;
    return make_float2((x.x*y.x + x.y*y.y)/den, (x.y*y.x - x.x*y.y)/den);
}

extern "C" __global__
void thomas_const_complex64(
    const float lower_r, const float lower_i,
    const float upper_r, const float upper_i,
    const float2* __restrict__ diag,
    const float2* __restrict__ rhs,
    float2* __restrict__ cprime,
    float2* __restrict__ dprime,
    float2* __restrict__ out,
    const int batch, const int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch || n <= 0) return;
    #define IDX(j) (i * n + (j))

    float2 a = make_float2(lower_r, lower_i);
    float2 c = make_float2(upper_r, upper_i);
    float2 b = diag[i];

    float2 denom = b;
    cprime[IDX(0)] = (n > 1) ? c64_div(c, denom) : make_float2(0.0f, 0.0f);
    dprime[IDX(0)] = c64_div(rhs[IDX(0)], denom);

    for (int j = 1; j < n; ++j) {
        denom = c64_sub(b, c64_mul(a, cprime[IDX(j-1)]));
        cprime[IDX(j)] = (j < n - 1) ? c64_div(c, denom) : make_float2(0.0f, 0.0f);
        dprime[IDX(j)] = c64_div(c64_sub(rhs[IDX(j)], c64_mul(a, dprime[IDX(j-1)])), denom);
    }

    out[IDX(n-1)] = dprime[IDX(n-1)];
    for (int j = n - 2; j >= 0; --j) {
        out[IDX(j)] = c64_sub(dprime[IDX(j)], c64_mul(cprime[IDX(j)], out[IDX(j+1)]));
    }

    #undef IDX
}
"""

_CUDA_SRC_COMPLEX128 = r"""
__device__ inline double2 c128_mul(double2 x, double2 y) {
    return make_double2(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x);
}
__device__ inline double2 c128_sub(double2 x, double2 y) {
    return make_double2(x.x-y.x, x.y-y.y);
}
__device__ inline double2 c128_div(double2 x, double2 y) {
    double den = y.x*y.x + y.y*y.y;
    return make_double2((x.x*y.x + x.y*y.y)/den, (x.y*y.x - x.x*y.y)/den);
}

extern "C" __global__
void thomas_const_complex128(
    const double lower_r, const double lower_i,
    const double upper_r, const double upper_i,
    const double2* __restrict__ diag,
    const double2* __restrict__ rhs,
    double2* __restrict__ cprime,
    double2* __restrict__ dprime,
    double2* __restrict__ out,
    const int batch, const int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch || n <= 0) return;
    #define IDX(j) (i * n + (j))

    double2 a = make_double2(lower_r, lower_i);
    double2 c = make_double2(upper_r, upper_i);
    double2 b = diag[i];

    double2 denom = b;
    cprime[IDX(0)] = (n > 1) ? c128_div(c, denom) : make_double2(0.0, 0.0);
    dprime[IDX(0)] = c128_div(rhs[IDX(0)], denom);

    for (int j = 1; j < n; ++j) {
        denom = c128_sub(b, c128_mul(a, cprime[IDX(j-1)]));
        cprime[IDX(j)] = (j < n - 1) ? c128_div(c, denom) : make_double2(0.0, 0.0);
        dprime[IDX(j)] = c128_div(c128_sub(rhs[IDX(j)], c128_mul(a, dprime[IDX(j-1)])), denom);
    }

    out[IDX(n-1)] = dprime[IDX(n-1)];
    for (int j = n - 2; j >= 0; --j) {
        out[IDX(j)] = c128_sub(dprime[IDX(j)], c128_mul(cprime[IDX(j)], out[IDX(j+1)]));
    }

    #undef IDX
}
"""


@lru_cache(maxsize=2)
def _kernel(dtype_name: str):
    import cupy as cp

    if dtype_name == "complex64":
        return cp.RawKernel(_CUDA_SRC_COMPLEX64, "thomas_const_complex64")
    if dtype_name == "complex128":
        return cp.RawKernel(_CUDA_SRC_COMPLEX128, "thomas_const_complex128")
    raise TypeError(dtype_name)


def _complex_parts(z: Any, real_dtype: Any):
    zz = complex(z)
    return real_dtype(zz.real), real_dtype(zz.imag)


def solve_const_offdiag_batched_fast(
    lower: Any,
    diag: Array,
    upper: Any,
    rhs: Array,
    *,
    threads_per_block: int = 128,
) -> Array:
    """Fast CuPy batched Thomas solve, falling back to the reference otherwise."""

    if not (_is_cupy_array(rhs) and _is_cupy_array(diag)):
        return solve_const_offdiag_batched(lower, diag, upper, rhs)

    import cupy as cp

    if rhs.ndim != 2:
        raise ValueError("rhs must have shape (batch, n)")
    if diag.ndim != 1:
        raise ValueError("diag must have shape (batch,)")
    if rhs.shape[0] != diag.shape[0]:
        raise ValueError("rhs.shape[0] must equal diag.shape[0]")

    if rhs.dtype not in (cp.complex64, cp.complex128):
        return solve_const_offdiag_batched(lower, diag, upper, rhs, xp=cp)

    batch, n = map(int, rhs.shape)
    if n == 0:
        return cp.empty_like(rhs)

    dtype = rhs.dtype
    real_dtype = np.float32 if dtype == cp.complex64 else np.float64

    diag_c = diag.astype(dtype, copy=False)
    rhs_c = cp.ascontiguousarray(rhs.astype(dtype, copy=False))

    cprime = cp.empty_like(rhs_c)
    dprime = cp.empty_like(rhs_c)
    out = cp.empty_like(rhs_c)

    lower_r, lower_i = _complex_parts(lower, real_dtype)
    upper_r, upper_i = _complex_parts(upper, real_dtype)

    block = int(threads_per_block)
    grid = ((batch + block - 1) // block,)

    kern = _kernel("complex64" if dtype == cp.complex64 else "complex128")
    kern(
        grid,
        (block,),
        (
            lower_r, lower_i, upper_r, upper_i,
            diag_c, rhs_c, cprime, dprime, out,
            np.int32(batch), np.int32(n),
        ),
    )
    return out


__all__ = ["solve_const_offdiag_batched_fast"]
