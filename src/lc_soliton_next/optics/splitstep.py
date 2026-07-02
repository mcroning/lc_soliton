"""Precision-aware split-step optics for z-stack TD propagation.

This is the clean next-package version of the optics slice logic validated
against runner_core in test 45.  The algorithmic ordering is unchanged:

    phase(theta_k) -> linear hop

repeated for each optics substep, with midpoint intensity

    I_mid = 0.5 * (I_before + I_after)

returned for the theta z-slice update.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

try:
    import cupy as _cp  # type: ignore
except Exception:  # pragma: no cover
    _cp = None

try:
    import cupyx.scipy.fft as _cupy_fft  # type: ignore
except Exception:  # pragma: no cover
    _cupy_fft = None

import scipy.fft as _scipy_fft

Array = Any


def array_module(*arrays: Array, xp: Any | None = None):
    if xp is not None:
        return xp
    if _cp is not None:
        for a in arrays:
            if isinstance(a, _cp.ndarray):
                return _cp
    return np


def fft_module(xp):
    return _cupy_fft if (_cp is not None and xp is _cp and _cupy_fft is not None) else _scipy_fft


def asnumpy(a):
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def intensity(amp: Array, *, coherent: bool = False, out: Optional[Array] = None, xp: Any | None = None) -> Array:
    """Optical intensity using the trusted one/two-channel convention."""
    xp = array_module(amp, xp=xp)
    if getattr(amp, "ndim", 0) == 2:
        a = amp
        if out is None:
            out = xp.empty(a.shape, dtype=a.real.dtype)
        out[...] = a.real * a.real + a.imag * a.imag
        return out

    if amp.shape[0] == 1:
        a = amp[0]
        if out is None:
            out = xp.empty(a.shape, dtype=a.real.dtype)
        out[...] = a.real * a.real + a.imag * a.imag
        return out

    a0, a1 = amp[0], amp[1]
    if out is None:
        out = xp.empty(a0.shape, dtype=a0.real.dtype)
    if coherent:
        s = a0 + a1
        out[...] = s.real * s.real + s.imag * s.imag
    else:
        out[...] = (
            a0.real * a0.real + a0.imag * a0.imag
            + a1.real * a1.real + a1.imag * a1.imag
        )
    return out


def intens(amp: Array, coh: bool, out: Optional[Array] = None) -> Array:
    return intensity(amp, coherent=coh, out=out)


def intens_into(out_I: Array, amp: Array, *, coh: bool, xp: Any | None = None) -> Array:
    out = intensity(amp, coherent=coh, out=out_I, xp=xp)
    out_I[...] = out.astype(out_I.dtype, copy=False)
    return out_I


def normalize_power_inplace(amp: Array, power: float, dx: float, dy: float, *, coherent: bool = False, xp: Any | None = None) -> Array:
    xp = array_module(amp, xp=xp)
    I = intensity(amp, coherent=coherent, xp=xp)
    p = xp.sum(I) * float(dx) * float(dy)
    scale = xp.sqrt(float(power) / p) if float(power) > 0 else 0.0
    amp *= scale.astype(amp.real.dtype, copy=False) if hasattr(scale, "astype") else scale
    return amp


def hop_linear_inplace(
    amp: Array,
    hker: Array,
    windowxy: Optional[Array] = None,
    Ahat: Optional[Array] = None,
    *,
    plan_f: Any = None,
    plan_i: Any = None,
    xp: Any | None = None,
) -> Array:
    """One in-place angular-spectrum linear hop."""
    xp = array_module(amp, hker, xp=xp)
    fft = fft_module(xp)
    if plan_f is None:
        A = fft.fft2(amp, axes=(-2, -1))
    else:  # pragma: no cover; plan path depends on backend
        with plan_f:
            A = fft.fft2(amp, axes=(-2, -1))
    A *= hker
    if plan_i is None:
        out = fft.ifft2(A, axes=(-2, -1))
    else:  # pragma: no cover
        with plan_i:
            out = fft.ifft2(A, axes=(-2, -1))
    amp[...] = out.astype(amp.dtype, copy=False)
    if windowxy is not None:
        amp *= windowxy
    return amp


def hop_linear(amp: Array, hker: Array, windowxy: Optional[Array] = None, *, xp: Any | None = None) -> Array:
    xp = array_module(amp, hker, xp=xp)
    out = xp.array(amp, copy=True)
    return hop_linear_inplace(out, hker, windowxy, xp=xp)


def lc_dn_from_theta(theta: Array, ne: float, no: float, refin: float, *, xp: Any | None = None, dtype: Any | None = None) -> Array:
    """Return n_eff(theta)-refin in the requested real precision."""
    xp = array_module(theta, xp=xp)
    real_dtype = dtype or theta.dtype
    th = theta.astype(real_dtype, copy=False)
    ct = xp.cos(th)
    st = xp.sin(th)
    ne_f = float(ne)
    no_f = float(no)
    n_eff = (ne_f * no_f) / xp.sqrt((ne_f * ct) ** 2 + (no_f * st) ** 2)
    return (n_eff - float(refin)).astype(real_dtype, copy=False)


def apply_nonlinear_phase_inplace(
    amp: Array,
    theta_xy: Array,
    *,
    dz_step_um: float,
    kout: float,
    ne: float,
    no: float,
    refin: float,
    xp: Any | None = None,
) -> Array:
    """Apply the LC phase step in the precision of ``amp``/``theta``."""
    xp = array_module(amp, theta_xy, xp=xp)
    dn = lc_dn_from_theta(theta_xy, ne, no, refin, xp=xp, dtype=theta_xy.dtype)
    phase = xp.exp((1j * float(kout) * float(dz_step_um)) * dn)
    amp *= phase.astype(amp.dtype, copy=False)
    return amp


def fill_mid_intensity(I_mid: Array, I_b: Array, I_a: Array) -> Array:
    I_mid[...] = 0.5 * (I_b + I_a)
    return I_mid


def choose_optics_substeps(
    dz_um: float,
    *,
    kout: float,
    dn_max_est: float,
    dz_opt_max_phi: float,
    max_substeps: int,
) -> Tuple[int, float, float]:
    phi_est = abs(float(kout)) * abs(float(dz_um)) * abs(float(dn_max_est))
    nsub = int(np.ceil(phi_est / float(dz_opt_max_phi))) if phi_est > 0 else 1
    nsub = max(1, min(int(max_substeps), nsub))
    return nsub, float(dz_um) / nsub, phi_est


def propagation_kernel(ctx: Any, dz_step_um: float, *, xp: Any | None = None) -> Array:
    """Angular-spectrum propagation kernel for one linear hop."""
    xp = xp or getattr(ctx, "xp", None) or array_module(getattr(ctx, "fxy2", None))
    if not hasattr(ctx, "_h_cache") or ctx._h_cache is None:
        ctx._h_cache = {}
    key = (float(np.round(float(dz_step_um), 12)), str(getattr(ctx, "complex_dtype", "")))
    if key in ctx._h_cache:
        return ctx._h_cache[key]
    lm = float(ctx.lm)
    refin = float(ctx.refin)
    fxy2 = ctx.fxy2
    real_dtype = getattr(ctx, "real_dtype", xp.float32 if hasattr(xp, "float32") else np.float32)
    complex_dtype = getattr(ctx, "complex_dtype", xp.complex64 if hasattr(xp, "complex64") else np.complex64)
    arg = 1.0 - (lm / refin) ** 2 * fxy2
    h = xp.where(
        arg >= 0.0,
        xp.exp(2.0j * xp.pi * refin * real_dtype(dz_step_um) / lm * xp.sqrt(xp.maximum(arg, 0.0))),
        0.0,
    ).astype(complex_dtype, copy=False)
    ctx._h_cache[key] = h
    return h


def get_h_for_dz(ctx: Any, dz_step_um: float) -> Array:
    return propagation_kernel(ctx, dz_step_um, xp=getattr(ctx, "xp", None))


def advance_optics_slice(
    ctx: Any,
    amp: Array,
    theta_use: Array,
    *,
    I_b: Optional[Array] = None,
    I_a: Optional[Array] = None,
    I_mid: Optional[Array] = None,
    Nsub: Optional[int] = None,
    dz_sub: Optional[float] = None,
    h_sub: Optional[Array] = None,
    Ahat: Optional[Array] = None,
    plan_f: Any = None,
    plan_i: Any = None,
) -> tuple[Array, Array, Array, Array]:
    """Advance one z-slice and return ``amp, I_b, I_a, I_mid``."""
    xp = getattr(ctx, "xp", None) or array_module(amp, theta_use)
    shape = amp.shape[-2:]
    real_dtype = getattr(ctx, "real_dtype", amp.real.dtype)
    if I_b is None:
        I_b = xp.empty(shape, dtype=real_dtype)
    if I_a is None:
        I_a = xp.empty(shape, dtype=real_dtype)
    if I_mid is None:
        I_mid = xp.empty(shape, dtype=real_dtype)
    intens_into(I_b, amp, coh=bool(getattr(ctx, "coh", False)), xp=xp)
    if Nsub is None or dz_sub is None:
        Nsub, dz_sub, _ = choose_optics_substeps(
            float(ctx.dz), kout=float(ctx.kout),
            dn_max_est=float(getattr(ctx, "dn_max_est", 0.02)),
            dz_opt_max_phi=float(getattr(ctx, "dz_opt_max_phi", 0.3)),
            max_substeps=int(getattr(ctx, "max_substeps", 16)),
        )
    if h_sub is None:
        h_sub = propagation_kernel(ctx, float(dz_sub), xp=xp)
    for _ in range(int(Nsub)):
        apply_nonlinear_phase_inplace(
            amp, theta_use,
            dz_step_um=float(dz_sub), kout=float(ctx.kout),
            ne=float(ctx.ne), no=float(ctx.no), refin=float(ctx.refin), xp=xp,
        )
        hop_linear_inplace(amp, h_sub, getattr(ctx, "windowxy", None), Ahat, plan_f=plan_f, plan_i=plan_i, xp=xp)
    intens_into(I_a, amp, coh=bool(getattr(ctx, "coh", False)), xp=xp)
    fill_mid_intensity(I_mid, I_b, I_a)
    return amp, I_b, I_a, I_mid


__all__ = [
    "array_module", "asnumpy", "intensity", "intens", "intens_into",
    "normalize_power_inplace", "hop_linear", "hop_linear_inplace",
    "lc_dn_from_theta", "apply_nonlinear_phase_inplace", "fill_mid_intensity",
    "choose_optics_substeps", "propagation_kernel", "get_h_for_dz", "advance_optics_slice",
]
