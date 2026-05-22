# lc_core/optics.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpticsStepInfo:
    dz_um: float
    used_cache: bool
    support_fraction: float | None = None


def intensity(amp, coh: bool = True):
    """
    Intensity convention for two-component launches.

    amp can be:
        (Nx, Ny) complex field
    or:
        (2, Nx, Ny) two-component field

    coherent:
        |E1 + E2|^2

    incoherent:
        |E1|^2 + |E2|^2
    """
    xp = _xp_of(amp)

    if amp.ndim == 2:
        return xp.abs(amp) ** 2

    if amp.ndim == 3:
        if coh:
            return xp.abs(xp.sum(amp, axis=0)) ** 2
        return xp.sum(xp.abs(amp) ** 2, axis=0)

    raise ValueError(f"Expected amp ndim 2 or 3, got shape {amp.shape}")


def total_power(Ixy, dx_um: float, dy_um: float) -> float:
    """
    Return integral I dx dy using microns as the transverse length unit,
    matching the existing normalization convention.
    """
    xp = _xp_of(Ixy)
    val = xp.sum(Ixy) * xp.asarray(dx_um * dy_um, dtype=Ixy.dtype)
    return float(val.get() if hasattr(val, "get") else val)


def normalize_amp_power(amp, dx_um: float, dy_um: float, coh: bool = True, target_power: float = 1.0):
    """
    Normalize optical field so integral I dx dy = target_power.
    """
    xp = _xp_of(amp)
    I = intensity(amp, coh=coh)
    P = xp.sum(I) * xp.asarray(dx_um * dy_um, dtype=I.dtype)
    P = xp.where(P == 0, 1.0, P)
    return (amp * xp.sqrt(xp.asarray(target_power, dtype=I.dtype) / P)).astype(amp.dtype, copy=False)


def spectral_support_mask(fxy2, lm_um: float, refin: float | None = None):
    """
    Legacy-compatible support mask:
        lm**2 * fxy2 < 1.0
    """
    return (float(lm_um) ** 2 * fxy2 < 1.0)


def get_h_for_dz(ctx, dz_um: float):
    """
    Cached nonparaxial angular-spectrum transfer function.

    Legacy-compatible form:
        q = kout * refin * (sqrt(1 - (lm/refin)^2 fxy2) - 1)
        h = exp(i q dz)

    Frequencies outside the propagating support are zeroed.
    """
    cp = ctx.xp
    dz_key = float(dz_um)

    if not hasattr(ctx, "h_cache") or ctx.h_cache is None:
        ctx.h_cache = {}

    if dz_key in ctx.h_cache:
        return ctx.h_cache[dz_key], OpticsStepInfo(dz_um=dz_key, used_cache=True)

    lm = float(ctx.lm)
    refin = float(ctx.refin)
    kout = float(ctx.kout)

    arg = 1.0 - (lm / refin) ** 2 * ctx.fxy2
    support = arg >= 0.0

    root = cp.sqrt(cp.maximum(arg, 0.0)).astype(cp.float32, copy=False)
    q = (kout * refin * (root - 1.0)).astype(cp.float32, copy=False)

    h = cp.exp(1j * q * cp.float32(dz_key)).astype(cp.complex64, copy=False)
    h = cp.where(support, h, cp.complex64(0.0))

    ctx.h_cache[dz_key] = h

    support_fraction = float(cp.mean(support.astype(cp.float32)).get())
    return h, OpticsStepInfo(dz_um=dz_key, used_cache=False, support_fraction=support_fraction)


def linear_step(ctx, amp, dz_um: float, *, apply_window: bool = False):
    """
    One angular-spectrum linear propagation step.

    amp may be shape:
        (Nx, Ny)
    or:
        (2, Nx, Ny)
    """
    cp = ctx.xp
    h, info = get_h_for_dz(ctx, dz_um)

    A = cp.fft.fft2(amp, axes=(-2, -1))
    A = A * h
    out = cp.fft.ifft2(A, axes=(-2, -1)).astype(cp.complex64, copy=False)

    if apply_window:
        out = apply_sponge(ctx, out)

    return out, info


def apply_sponge(ctx, amp):
    """
    Apply transverse sponge/window to field amplitude.
    """
    if getattr(ctx, "windowxy", None) is None:
        return amp
    return (amp * ctx.windowxy).astype(amp.dtype, copy=False)


def n_eff_from_theta(theta, ne: float, no: float):
    """
    Effective extraordinary index mapping:
        n_eff = ne*no / sqrt((ne cos theta)^2 + (no sin theta)^2)
    """
    xp = _xp_of(theta)
    ne = float(ne)
    no = float(no)
    den = xp.sqrt((ne * xp.cos(theta)) ** 2 + (no * xp.sin(theta)) ** 2)
    return (ne * no / den).astype(theta.dtype, copy=False)


def dn_from_theta(ctx, theta):
    """
    dn(theta) = n_eff(theta) - refin.
    """
    return (n_eff_from_theta(theta, ctx.ne, ctx.no) - ctx.refin).astype(theta.dtype, copy=False)


def nonlinear_phase_step(ctx, amp, theta, dz_um: float):
    """
    Apply local nonlinear phase:
        amp -> amp * exp(i * kout * dn(theta) * dz)
    """
    cp = ctx.xp
    dn = dn_from_theta(ctx, theta)
    phase = cp.exp(1j * cp.float32(ctx.kout * float(dz_um)) * dn).astype(cp.complex64, copy=False)
    return (amp * phase).astype(cp.complex64, copy=False)


def strang_step_with_fixed_theta(ctx, amp, theta, dz_um: float, *, apply_window: bool = False):
    """
    Optics-only Strang step with fixed theta:
        L(dz/2) -> N(theta, dz) -> L(dz/2)
    """
    half = 0.5 * float(dz_um)

    a, info1 = linear_step(ctx, amp, half, apply_window=False)
    a = nonlinear_phase_step(ctx, a, theta, dz_um)
    a, info2 = linear_step(ctx, a, half, apply_window=apply_window)

    return a, (info1, info2)


def optics_smoke_test(ctx, launch_arrays, *, dz_um: float | None = None, apply_window: bool = False):
    """
    Minimal GPU optics smoke test.

    Returns JSON-safe diagnostics. Does not modify ctx.amp0.
    """
    cp = ctx.xp

    if dz_um is None:
        dz_um = float(ctx.dz)

    amp0 = ctx.amp0
    I0 = intensity(amp0, coh=ctx.coh).astype(cp.float32, copy=False)
    P0 = total_power(I0, ctx.dx, ctx.dy)

    amp1, info = linear_step(ctx, amp0, dz_um, apply_window=apply_window)
    I1 = intensity(amp1, coh=ctx.coh).astype(cp.float32, copy=False)
    P1 = total_power(I1, ctx.dx, ctx.dy)

    return dict(
        dz_um=float(dz_um),
        apply_window=bool(apply_window),
        amp0_shape=list(amp0.shape),
        amp1_shape=list(amp1.shape),
        P0=float(P0),
        P1=float(P1),
        rel_power_change=float((P1 - P0) / P0) if P0 != 0 else None,
        I0_max=float(cp.max(I0).get()),
        I1_max=float(cp.max(I1).get()),
        h_used_cache=bool(info.used_cache),
        support_fraction=info.support_fraction,
    )


def _xp_of(arr):
    """
    Return numpy or cupy module for an array-like object.
    """
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy as cp
        return cp
    import numpy as np
    return np
