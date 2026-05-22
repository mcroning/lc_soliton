"""
lc_engine_validated.py

A compact, platform-conscious LC propagation demonstrator.

Validated public/demo branches included:
  - strict static slice relaxation
  - true-time TD predictor-only update
  - optional dual-grid TD predictor-only
  - lightweight xz/yz slice storage and scalar logs

Backends:
  - NumPy works everywhere.
  - CuPy is used when requested and available.

This module is intentionally smaller than the research notebook. It is meant
for independent users to explore behavior without exposing experimental branches.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import math
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import cupy as _cupy  # type: ignore
    import cupyx.scipy.fft as _cupy_fft  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover
    _cupy = None
    _cupy_fft = None
    _HAS_CUPY = False


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------

def get_backend(name="auto", verbose=True):
    """
    Backend selector.

    Returns
    -------
    xp : module
        cupy or numpy

    backend_info : dict
        information for display/logging
    """

    # --------------------------------------------------------
    # Explicit CPU
    # --------------------------------------------------------
    if name in ("numpy", "cpu"):
        import numpy as np

        info = dict(
            backend="numpy",
            device="CPU",
            gpu_name=None,
            gpu_mem_gb=None,
        )

        if verbose:
            print("=" * 60)
            print("LC backend")
            print("  backend : NumPy")
            print("  device  : CPU")
            print("=" * 60)

        return np, info

    # --------------------------------------------------------
    # GPU or auto
    # --------------------------------------------------------
    if name in ("auto", "cupy", "gpu"):

        try:
            import cupy as cp

            ndev = cp.cuda.runtime.getDeviceCount()

            if ndev > 0:
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

        except Exception as e:

            if verbose:
                print(f"[backend fallback] CuPy unavailable: {e}")

        import numpy as np

        info = dict(
            backend="numpy",
            device="CPU",
            gpu_name=None,
            gpu_mem_gb=None,
        )

        if verbose:
            print("=" * 60)
            print("LC backend")
            print("  backend : NumPy")
            print("  device  : CPU")
            print("  note    : GPU backend unavailable")
            print("=" * 60)

        return np, info

    raise ValueError(
        f"Unknown backend '{name}'. "
        "Use auto, cupy/gpu, or numpy/cpu."
    )


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


def fft2(xp, a, axes=(-2, -1)):
    if _HAS_CUPY and xp is _cupy:
        return _cupy_fft.fft2(a, axes=axes)
    return np.fft.fft2(a, axes=axes)


def ifft2(xp, a, axes=(-2, -1)):
    if _HAS_CUPY and xp is _cupy:
        return _cupy_fft.ifft2(a, axes=axes)
    return np.fft.ifft2(a, axes=axes)


def fft(xp, a, axis=-1):
    return xp.fft.fft(a, axis=axis)


def ifft(xp, a, axis=-1):
    return xp.fft.ifft(a, axis=axis)


# -----------------------------------------------------------------------------
# Parameters and context
# -----------------------------------------------------------------------------

@dataclass
class LCParams:
    # grid and propagation
    Nx: int = 256
    Ny: int = 256
    Nz: int = 300
    xaper_um: float = 75.0
    yaper_um: float = 500.0
    dz_um: float = 5.0
    wavelength_um: float = 0.633

    # material / director model
    ne: float = 1.70
    no: float = 1.50
    b: float = 1.0
    bi: float = 0.5
    mobility: float = 1.0
    theta_bc: float = 0.0
    theta_bias_amp: float = 0.10
    theta_clamp_min: float = -1.2
    theta_clamp_max: float = 1.2
    theta_z_gamma: float = 0.0

    # optical input
    power_norm: float = 1.0
    waist_x_um: float = 8.0
    waist_y_um: float = 8.0
    y_sep_um: float = 0.0
    coherent: bool = True

    # stepping controls
    dz_opt_max_phi: float = 0.30
    dn_max_est: float = 0.20
    max_substeps: int = 16
    dtau_static: float = 0.02
    static_max_steps: int = 250
    static_tol_rms: float = 1e-3
    static_tol_max: float = 1e-2
    static_selfcons_passes: int = 3
    static_mix: float = 0.6

    # dual grid
    use_dual_grid: bool = False
    dual_grid_factor: int = 2

    # backend
    backend: str = "auto"


@dataclass
class LCContext:
    p: LCParams
    xp: Any
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    du_lc: float
    dv_lc: float
    x: Any
    y: Any
    fxy2: Any
    amp0: Any
    theta_bias_2d: Any
    refin: float
    kout: float
    windowxy: Any
    theta_clamp: Tuple[float, float]
    _h_cache: Dict[float, Any]


@dataclass
class DualGrid:
    factor: int
    Nx_c: int
    Ny_c: int
    theta_bias_c: Any


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

def compute_neff(theta, *, ne: float, no: float, xp):
    ct = xp.cos(theta)
    st = xp.sin(theta)
    return (ne * no) / xp.sqrt((ne * ct) ** 2 + (no * st) ** 2)


def theta_bias_profile(p: LCParams, xp):
    x01 = xp.linspace(-1.0, 1.0, p.Nx, dtype=xp.float32)
    # Smooth symmetric trial bias satisfying Dirichlet rows.
    th = p.theta_bc + p.theta_bias_amp * xp.cos(0.5 * xp.pi * x01)
    th = xp.asarray(th, dtype=xp.float32)
    th[0] = p.theta_bc
    th[-1] = p.theta_bc
    return xp.tile(th[:, None], (1, p.Ny)).astype(xp.float32)


def make_input_field(p: LCParams, x, y, xp):
    X = x[:, None]
    Y = y[None, :]
    w0x = float(p.waist_x_um)
    w0y = float(p.waist_y_um)
    sep = float(p.y_sep_um)

    g1 = xp.exp(-(X ** 2) / (w0x ** 2) - ((Y - 0.5 * sep) ** 2) / (w0y ** 2))
    if abs(sep) > 0:
        g2 = xp.exp(-(X ** 2) / (w0x ** 2) - ((Y + 0.5 * sep) ** 2) / (w0y ** 2))
    else:
        g2 = xp.zeros_like(g1)

    amp = xp.stack([g1, g2], axis=0).astype(xp.complex64)
    I = intensity(amp, p.coherent, xp=xp)
    norm = xp.sum(I) * xp.float32((p.xaper_um / p.Nx) * (p.yaper_um / p.Ny))
    norm = xp.where(norm == 0, xp.float32(1.0), norm)
    amp = (amp / xp.sqrt(norm / xp.float32(p.power_norm))).astype(xp.complex64)
    return amp


def make_context(params: LCParams) -> Tuple[LCContext, Optional[DualGrid]]:
    xp, backend_info = get_backend(params.backend)
    Nx, Ny, Nz = int(params.Nx), int(params.Ny), int(params.Nz)
    dx = float(params.xaper_um) / Nx
    dy = float(params.yaper_um) / Ny
    dz = float(params.dz_um)

    x = (xp.arange(Nx, dtype=xp.float32) - Nx / 2) * dx + 0.5 * dx
    y = (xp.arange(Ny, dtype=xp.float32) - Ny / 2) * dy + 0.5 * dy

    theta_bias_2d = theta_bias_2d_from_params(params, xp)
    refin = float(asnumpy(compute_neff(theta_bias_2d, ne=params.ne, no=params.no, xp=xp)).mean())
    kout = float(2.0 * math.pi / params.wavelength_um)

    fx = xp.fft.fftfreq(Nx, d=dx).astype(xp.float32)
    fy = xp.fft.fftfreq(Ny, d=dy).astype(xp.float32)
    fxy2 = (fx[:, None] ** 2 + fy[None, :] ** 2).astype(xp.float32)

    amp0 = make_input_field(params, x, y, xp)
    windowxy = xp.ones((Nx, Ny), dtype=xp.float32)
    du_lc = 2.0 / max(1, Nx - 1)
    dv_lc = du_lc * (dy / dx)
    ctx = LCContext(
        p=params,
        xp=xp,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=dx,
        dy=dy,
        dz=dz,
        du_lc=du_lc,
        dv_lc=dv_lc,
        x=x,
        y=y,
        fxy2=fxy2,
        amp0=amp0,
        theta_bias_2d=theta_bias_2d,
        refin=refin,
        kout=kout,
        windowxy=windowxy,
        theta_clamp=(params.theta_clamp_min, params.theta_clamp_max),
        _h_cache={},
    )

    dg = None
    if params.use_dual_grid:
        f = int(params.dual_grid_factor)
        if Nx % f or Ny % f:
            raise ValueError("Nx and Ny must be divisible by dual_grid_factor")
        theta_bias_c = restrict_block_mean(theta_bias_2d, f, xp=xp)
        dg = DualGrid(factor=f, Nx_c=Nx // f, Ny_c=Ny // f, theta_bias_c=theta_bias_c)

    return ctx, dg


# -----------------------------------------------------------------------------
# Core numerics
# -----------------------------------------------------------------------------




def lc_b_from_voltage(V, K=7e-12, De=13.0):
    eps0 = 8.8541878128e-12
    return float(eps0 * float(De) * float(V)**2 / (8.0 * float(K)))


def voltage_from_lc_b(b, K=7e-12, De=13.0):
    eps0 = 8.8541878128e-12
    return float((8.0 * float(K) * float(b) / (eps0 * float(De))) ** 0.5)


def theta0_from_b_zero_bc(b):
    import numpy as np
    from scipy import special as spspec

    b = float(b)
    bc = np.pi**2 / 8.0
    if b <= bc:
        return 0.0

    lo, hi = 1e-14, 1.0 - 1e-14

    for _ in range(80):
        m = 0.5 * (lo + hi)
        val = spspec.ellipk(m)**2 - 2.0 * b
        if val > 0:
            hi = m
        else:
            lo = m

    m = 0.5 * (lo + hi)
    return float(np.arcsin(np.sqrt(m)))


def b_from_theta0_zero_bc(theta0):
    import numpy as np
    from scipy import special as spspec

    th = abs(float(theta0))
    if th <= 0:
        return float(np.pi**2 / 8.0)

    m = np.sin(th)**2
    return float(0.5 * spspec.ellipk(m)**2)


def theta_bias_1d_exact_zero_bc(Nx, b, xp):
    import numpy as np
    from scipy import special as spspec

    b = float(b)
    bc = np.pi**2 / 8.0

    if b <= bc:
        return xp.zeros(int(Nx), dtype=xp.float32)

    theta0 = theta0_from_b_zero_bc(b)
    m = np.sin(theta0)**2

    u_cpu = np.linspace(-1.0, 1.0, int(Nx), dtype=np.float64)
    arg_cpu = np.sqrt(2.0 * b) * u_cpu

    _, cn, dn, _ = spspec.ellipj(arg_cpu, m)
    s = np.sin(theta0) * cn / dn
    s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)

    theta_cpu = np.arcsin(s).astype(np.float32)
    theta_cpu[0] = 0.0
    theta_cpu[-1] = 0.0

    return xp.asarray(theta_cpu, dtype=xp.float32)


def theta_bias_1d_relax_bc(Nx, b, theta_bc, xp, *, max_iter=200000, tol=1e-8):
    """
    Portable 1D bias-profile relaxation for nonzero theta_bc.

    Solves:
        theta_xx + b sin(2 theta) = 0
    on normalized x in [-1, 1], with theta=theta_bc at both boundaries.
    """
    import numpy as np

    Nx = int(Nx)
    b = float(b)
    theta_bc = float(theta_bc)

    u = np.linspace(-1.0, 1.0, Nx, dtype=np.float64)
    du = float(u[1] - u[0])

    # Smooth symmetric initial condition.
    if abs(theta_bc) < 1e-14:
        th0 = theta_bias_1d_exact_zero_bc(Nx, b, np)
        theta = np.asarray(th0, dtype=np.float64)
    else:
        center_boost = max(0.0, theta0_from_b_zero_bc(b))
        theta = theta_bc + center_boost * np.cos(0.5 * np.pi * u)
        theta[0] = theta_bc
        theta[-1] = theta_bc

    dt = 0.2 * du * du

    for _ in range(int(max_iter)):
        old = theta.copy()

        lap = np.zeros_like(theta)
        lap[1:-1] = (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / (du * du)

        R = lap + b * np.sin(2.0 * theta)
        theta[1:-1] += dt * R[1:-1]
        theta[0] = theta_bc
        theta[-1] = theta_bc

        err = np.sqrt(np.mean((theta - old)**2))
        if err < tol:
            break

    return xp.asarray(theta.astype(np.float32), dtype=xp.float32)


def theta_bias_2d_from_params(params, xp):
    """
    Authoritative bias-profile builder for the validated engine.
    Uses exact elliptic solution for theta_bc=0 and relaxation otherwise.
    """
    Nx = int(params.Nx)
    Ny = int(params.Ny)
    b = float(params.b)
    theta_bc = float(params.theta_bc)

    if abs(theta_bc) < 1e-14:
        th1 = theta_bias_1d_exact_zero_bc(Nx, b, xp)
    else:
        th1 = theta_bias_1d_relax_bc(Nx, b, theta_bc, xp)

    th2 = xp.tile(th1[:, None], (1, Ny)).astype(xp.float32, copy=False)
    th2[0, :] = xp.float32(theta_bc)
    th2[-1, :] = xp.float32(theta_bc)
    return th2


def intensity(amp, coherent: bool, *, xp):
    a0 = amp[0]
    a1 = amp[1]
    if coherent:
        s = a0 + a1
        return (s.real * s.real + s.imag * s.imag).astype(xp.float32)
    return (a0.real * a0.real + a0.imag * a0.imag + a1.real * a1.real + a1.imag * a1.imag).astype(xp.float32)


def lc_dn(theta, ctx: LCContext):
    xp = ctx.xp
    return (compute_neff(theta, ne=ctx.p.ne, no=ctx.p.no, xp=xp) - ctx.refin).astype(xp.float32)


def get_h_for_dz(ctx: LCContext, dz_um: float):
    key = round(float(dz_um), 12)
    if key in ctx._h_cache:
        return ctx._h_cache[key]
    xp = ctx.xp
    lm = float(ctx.p.wavelength_um)
    refin = float(ctx.refin)
    arg = 1.0 - (lm / refin) ** 2 * ctx.fxy2
    h = xp.where(arg > 0, xp.exp(2j * xp.pi * refin * dz_um / lm * xp.sqrt(xp.maximum(arg, 0))), 0)
    h = h.astype(xp.complex64)
    ctx._h_cache[key] = h
    return h


def hop_linear(ctx: LCContext, amp, h):
    xp = ctx.xp
    A = fft2(xp, amp, axes=(-2, -1))
    A *= h
    out = ifft2(xp, A, axes=(-2, -1)).astype(xp.complex64)
    if ctx.windowxy is not None:
        out *= ctx.windowxy[None, :, :]
    return out


def apply_nonlinear_phase(ctx: LCContext, amp, theta, dz_um: float):
    xp = ctx.xp
    phase = xp.exp((1j * xp.float32(ctx.kout * dz_um)) * lc_dn(theta, ctx)).astype(xp.complex64)
    return (amp * phase[None, :, :]).astype(xp.complex64)


def choose_optics_substeps(ctx: LCContext):
    p = ctx.p
    phi = abs(ctx.kout * ctx.dz * p.dn_max_est)
    nsub = max(1, min(int(p.max_substeps), int(math.ceil(phi / max(p.dz_opt_max_phi, 1e-12)))))
    return nsub, ctx.dz / nsub, phi


def propagate_slice(ctx: LCContext, amp, theta, *, nsub: int, dz_sub: float, h_sub):
    for _ in range(int(nsub)):
        amp = apply_nonlinear_phase(ctx, amp, theta, dz_sub)
        amp = hop_linear(ctx, amp, h_sub)
    return amp


def laplacian(theta, du: float, dv: float, xp, theta_bc: float = 0.0):
    th = theta.astype(xp.float32, copy=False)
    lap = xp.zeros_like(th)
    yplus = xp.roll(th, -1, axis=1)
    yminus = xp.roll(th, 1, axis=1)
    lap[1:-1, :] = ((th[2:, :] - 2 * th[1:-1, :] + th[:-2, :]) / (du * du)
                    + (yplus[1:-1, :] - 2 * th[1:-1, :] + yminus[1:-1, :]) / (dv * dv))
    lap[0, :] = 0
    lap[-1, :] = 0
    return lap


def residual(theta, I, ctx_like, *, du: float, dv: float, b: float, bi: float, mobility: float, theta_bc: float):
    xp = ctx_like.xp if hasattr(ctx_like, "xp") else ctx_like["xp"]
    R = (laplacian(theta, du, dv, xp, theta_bc=theta_bc) + (b + bi * I) * xp.sin(2 * theta)) / mobility
    R[0, :] = 0
    R[-1, :] = 0
    return R.astype(xp.float32)

def _extract_theta_b_1d(theta_bias, xp):
    th = xp.asarray(theta_bias)
    if th.ndim == 1:
        return th
    if th.ndim == 2:
        return xp.mean(th, axis=1)
    raise ValueError("theta_bias must have shape (Nx,) or (Nx, Ny)")


def thomas_batched_modevarying(off_lower, diag, off_upper, rhs, xp):
    rhs = xp.asarray(rhs)
    diag = xp.asarray(diag)

    Nx, Ny = rhs.shape
    dtype = xp.result_type(diag.dtype, rhs.dtype)

    a_in = xp.asarray(off_lower, dtype=dtype)
    c_in = xp.asarray(off_upper, dtype=dtype)

    if a_in.ndim == 1:
        a = xp.broadcast_to(a_in[:, None], (Nx - 1, Ny)).copy()
    else:
        a = xp.broadcast_to(a_in, (Nx - 1, Ny)).copy()

    if c_in.ndim == 1:
        c = xp.broadcast_to(c_in[:, None], (Nx - 1, Ny)).copy()
    else:
        c = xp.broadcast_to(c_in, (Nx - 1, Ny)).copy()

    bdiag = xp.asarray(diag, dtype=dtype).copy()
    d = xp.asarray(rhs, dtype=dtype).copy()

    for i in range(1, Nx):
        w = a[i - 1] / bdiag[i - 1]
        bdiag[i] = bdiag[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    x = xp.empty_like(d)
    x[-1] = d[-1] / bdiag[-1]

    for i in range(Nx - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / bdiag[i]

    return x


def solve_theta_perturbation_dirichletx_periody(
    theta_bias,
    Ixy,
    *,
    b,
    bi,
    du,
    dv,
    xp,
    return_theta=True,
):
    """
    Linear perturbation solve around theta_bias.

    Solves approximately:
        [L + 2 b cos(2 theta_b)] delta
        = - bi I sin(2 theta_b)

    and returns theta = theta_b + delta.
    """
    Ixy = xp.asarray(Ixy, dtype=xp.float64)
    theta_b_1d = _extract_theta_b_1d(theta_bias, xp).astype(xp.float64, copy=False)

    Nx, Ny = Ixy.shape

    qx = 2.0 * float(b) * xp.cos(2.0 * theta_b_1d)
    rhs = -float(bi) * Ixy * xp.sin(2.0 * theta_b_1d)[:, None]

    qx_i = qx[1:-1]
    rhs_i = rhs[1:-1, :]

    rhs_hat_i = xp.fft.fft(rhs_i, axis=1)

    k = xp.arange(Ny, dtype=xp.float64)
    lam_y = (2.0 * xp.cos(2.0 * xp.pi * k / Ny) - 2.0) / (float(dv) * float(dv))

    off = xp.full((Nx - 3,), 1.0 / (float(du) * float(du)), dtype=xp.float64)
    diag = (-2.0 / (float(du) * float(du))) + qx_i[:, None] + lam_y[None, :]

    delta_hat_i = thomas_batched_modevarying(off, diag, off, rhs_hat_i, xp)
    delta_i = xp.fft.ifft(delta_hat_i, axis=1).real

    delta = xp.zeros((Nx, Ny), dtype=xp.float64)
    delta[1:-1, :] = delta_i

    out = {"delta": delta}

    if return_theta:
        theta2d = theta_b_1d[:, None] + delta
        theta2d[0, :] = theta_b_1d[0]
        theta2d[-1, :] = theta_b_1d[-1]
        out["theta"] = theta2d

    return out



def enforce_theta(theta, ctx_or_params, xp=None):
    if xp is None:
        xp = ctx_or_params.xp
    if hasattr(ctx_or_params, "p"):
        p = ctx_or_params.p
    else:
        p = ctx_or_params
    theta = xp.clip(theta, p.theta_clamp_min, p.theta_clamp_max).astype(xp.float32)
    theta[0, :] = xp.float32(p.theta_bc)
    theta[-1, :] = xp.float32(p.theta_bc)
    return theta


def thomas_const_batched(a, b, c, d, xp):
    """Solve tridiagonal systems with constant off diagonal a,c and batched RHS d=(batch,n)."""
    d = d.copy()
    batch, n = d.shape
    cpv = xp.empty((n,), dtype=d.dtype)
    bp = xp.empty((n,), dtype=d.dtype)
    bp[0] = b[0] if getattr(b, "ndim", 0) else b
    cpv[0] = c / bp[0]
    for i in range(1, n):
        bi = b[i] if getattr(b, "ndim", 0) else b
        denom = bi - a * cpv[i - 1]
        bp[i] = denom
        cpv[i] = c / denom if i < n - 1 else 0
    d[:, 0] /= bp[0]
    for i in range(1, n):
        d[:, i] = (d[:, i] - a * d[:, i - 1]) / bp[i]
    x = xp.empty_like(d)
    x[:, -1] = d[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d[:, i] - cpv[i] * x[:, i + 1]
    return x


def prepare_ie_operator(dt: float, mobility: float, du: float, dv: float, Ny: int, xp):
    a_ie = float(dt) / float(mobility)
    ky = xp.arange(Ny, dtype=xp.float32)
    lam_y = (-4.0 * xp.sin(xp.pi * ky / Ny) ** 2) / (dv * dv)
    off = xp.float32(-a_ie / (du * du))
    diag = (1.0 + 2.0 * a_ie / (du * du) - a_ie * lam_y).astype(xp.float32)
    return xp.float32(a_ie), off, diag, lam_y


def solve_implicit_xy(rhs, *, off, diag, xp, theta_bc: float = 0.0):
    rhs2 = rhs.astype(xp.float32, copy=True)
    rhs2[0, :] = 0.0
    rhs2[-1, :] = 0.0

    rhs_hat = fft(xp, rhs2, axis=1).astype(xp.complex64)
    d_hat = rhs_hat[1:-1, :].T.copy()  # (Ny, Nx-2)

    theta_bc32 = xp.float32(theta_bc)
    if abs(float(theta_bc)) > 0.0:
        Ny_local = rhs.shape[1]
        bc_vec = xp.full((Ny_local,), theta_bc32, dtype=xp.float32)
        bc_hat = fft(xp, bc_vec, axis=0).astype(xp.complex64)

        d_hat[:, 0] -= off * bc_hat
        d_hat[:, -1] -= off * bc_hat

    x_hat = thomas_const_batched(off, diag.astype(xp.complex64), off, d_hat, xp)

    theta_hat = xp.zeros_like(rhs_hat)
    theta_hat[1:-1, :] = x_hat.T

    out = ifft(xp, theta_hat, axis=1).real.astype(xp.float32)
    out[0, :] = theta_bc32
    out[-1, :] = theta_bc32
    return out


def theta_td_predictor_only(theta_n, I_mid, ctx_like, *, dt: float, theta_prev=None, theta_next=None,
                            du: float, dv: float, b: float, bi: float, mobility: float,
                            theta_bc: float, gamma_z: float = 0.0):
    xp = ctx_like.xp if hasattr(ctx_like, "xp") else ctx_like["xp"]
    a_ie, off, diag, _ = prepare_ie_operator(dt, mobility, du, dv, theta_n.shape[1], xp)
    drive = (b + bi * I_mid) * xp.sin(2 * theta_n)
    rhs = theta_n + xp.float32(dt / mobility) * drive
    if gamma_z and theta_prev is not None and theta_next is not None:
        inv_dz2 = xp.float32(1.0 / (ctx_like.dz * ctx_like.dz)) if hasattr(ctx_like, "dz") else xp.float32(1.0)
        gam = xp.float32(gamma_z) * inv_dz2
        rhs += xp.float32(dt / mobility) * gam * (theta_prev + theta_next)
        diag = diag + xp.float32(2.0 * dt / mobility) * gam
    out = solve_implicit_xy(rhs, off=off, diag=diag, xp=xp, theta_bc=theta_bc)
    return enforce_theta(out, ctx_like, xp)


def strict_theta_solve(theta_seed, I_mid, ctx: LCContext, *, max_steps=None):
    xp = ctx.xp
    p = ctx.p
    max_steps = int(max_steps or p.static_max_steps)
    theta = theta_seed.astype(xp.float32, copy=True)
    theta = enforce_theta(theta, ctx)
    dt = float(p.dtau_static)
    for it in range(max_steps):
        theta_new = theta_td_predictor_only(
            theta,
            I_mid,
            ctx,
            dt=dt,
            du=ctx.du_lc,
            dv=ctx.dv_lc,
            b=p.b,
            bi=p.bi,
            mobility=p.mobility,
            theta_bc=p.theta_bc,
            gamma_z=0.0,
        )
        theta = ((1 - p.static_mix) * theta + p.static_mix * theta_new).astype(xp.float32)
        theta = enforce_theta(theta, ctx)
        if (it % 10) == 0 or it == max_steps - 1:
            R = residual(theta, I_mid, ctx, du=ctx.du_lc, dv=ctx.dv_lc, b=p.b, bi=p.bi,
                         mobility=p.mobility, theta_bc=p.theta_bc)
            Rint = R[1:-1, :]
            rmax = float(asnumpy(xp.max(xp.abs(Rint))))
            rrms = float(asnumpy(xp.sqrt(xp.mean(Rint * Rint))))
            if rrms < p.static_tol_rms and rmax < p.static_tol_max:
                return theta, {"iterations": it + 1, "rmax": rmax, "rrms": rrms, "converged": True}
    return theta, {"iterations": max_steps, "rmax": rmax, "rrms": rrms, "converged": False}


def static_fast_slice_selfconsistent(ctx: LCContext, amp_in, *, nsub: int, dz_sub: float, h_sub):
    xp = ctx.xp
    p = ctx.p

    I_b = intensity(amp_in, p.coherent, xp=xp)

    # First estimate from entrance intensity.
    out_lin = solve_theta_perturbation_dirichletx_periody(
        theta_bias=ctx.theta_bias_2d,
        Ixy=I_b,
        b=float(p.b),
        bi=float(p.bi),
        du=float(ctx.du_lc),
        dv=float(ctx.dv_lc),
        xp=xp,
        return_theta=True,
    )

    theta = out_lin["theta"].astype(xp.float32, copy=False)
    theta = enforce_theta(theta, ctx)

    # One optical self-consistency pass.
    amp_work = propagate_slice(
        ctx,
        amp_in.copy(),
        theta,
        nsub=nsub,
        dz_sub=dz_sub,
        h_sub=h_sub,
    )

    I_a = intensity(amp_work, p.coherent, xp=xp)
    I_mid = 0.5 * (I_b + I_a)

    out_lin = solve_theta_perturbation_dirichletx_periody(
        theta_bias=ctx.theta_bias_2d,
        Ixy=I_mid,
        b=float(p.b),
        bi=float(p.bi),
        du=float(ctx.du_lc),
        dv=float(ctx.dv_lc),
        xp=xp,
        return_theta=True,
    )

    theta = out_lin["theta"].astype(xp.float32, copy=False)
    theta = enforce_theta(theta, ctx)

    # Re-propagate with final theta for this slice.
    amp_out = propagate_slice(
        ctx,
        amp_in.copy(),
        theta,
        nsub=nsub,
        dz_sub=dz_sub,
        h_sub=h_sub,
    )

    R = residual(
        theta,
        I_mid,
        ctx,
        du=ctx.du_lc,
        dv=ctx.dv_lc,
        b=p.b,
        bi=p.bi,
        mobility=p.mobility,
        theta_bc=p.theta_bc,
    )
    Rint = R[1:-1, :]

    info = {
        "rrms": float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
        "rmax": float(asnumpy(xp.max(xp.abs(Rint)))),
        "converged": True,
        "method": "static_fast_linear_perturbation",
    }

    return theta, I_mid, amp_out, info



def strict_static_slice_selfconsistent(ctx: LCContext, amp_in, theta_seed, *, nsub: int, dz_sub: float, h_sub):
    xp = ctx.xp
    p = ctx.p

    I_b = intensity(amp_in, p.coherent, xp=xp)

    # validated seed: perturbation solve about theta_bias_2d
    try:
        out_lin = solve_theta_perturbation_dirichletx_periody(
            theta_bias=ctx.theta_bias_2d,
            Ixy=I_b,
            b=float(p.b),
            bi=float(p.bi),
            du=float(ctx.du_lc),
            dv=float(ctx.dv_lc),
            xp=xp,
            return_theta=True,
        )
        theta = out_lin["theta"].astype(xp.float32, copy=False)
        theta = enforce_theta(theta, ctx)
    except Exception:
        theta = theta_seed.astype(xp.float32, copy=True)
        theta = enforce_theta(theta, ctx)

    info = {}
    amp_out = amp_in
    I_mid = I_b

    for _ in range(int(p.static_selfcons_passes)):
        amp_work = propagate_slice(
            ctx,
            amp_in.copy(),
            theta,
            nsub=nsub,
            dz_sub=dz_sub,
            h_sub=h_sub,
        )

        I_a = intensity(amp_work, p.coherent, xp=xp)
        I_mid = 0.5 * (I_b + I_a)

        theta_new, info = strict_theta_solve(theta, I_mid, ctx)

        dtheta = float(asnumpy(xp.max(xp.abs(theta_new - theta))))
        theta = theta_new
        amp_out = amp_work

        if dtheta < 1e-4:
            break

    return theta, I_mid, amp_out, info

def lc_grid_spacings(ctx: LCContext):
    """
    Director PDE coordinates.

    theta_bias_1d_exact_zero_bc solves on u in [-1,1].
    Use the same normalized transverse coordinate for LC updates.
    """
    du = 2.0 / max(1, ctx.Nx - 1)
    dv = du * (ctx.dy / ctx.dx)
    return float(du), float(dv)

# -----------------------------------------------------------------------------
# Dual-grid helpers
# -----------------------------------------------------------------------------

def restrict_block_mean(arr, factor: int, *, xp):
    Nx, Ny = arr.shape
    f = int(factor)
    return arr.reshape(Nx // f, f, Ny // f, f).mean(axis=(1, 3)).astype(xp.float32)


def prolong_repeat(arr_c, factor: int, *, target_shape: Tuple[int, int], xp):
    f = int(factor)
    arr = xp.repeat(xp.repeat(arr_c, f, axis=0), f, axis=1)
    return arr[:target_shape[0], :target_shape[1]].astype(xp.float32)


# -----------------------------------------------------------------------------
# Lightweight storage
# -----------------------------------------------------------------------------

class LightStore:
    def __init__(self, run_dir: Path, ctx: LCContext, *, Nt_out: int, save_slices=True, save_full=False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = ctx
        self.Nt_out = int(Nt_out)
        self.save_slices = bool(save_slices)
        self.save_full = bool(save_full)
        self.log_path = self.run_dir / "scalar_log.csv"
        self.log_file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(self.log_file, fieldnames=[
            "jt", "k", "z_um", "Imax", "power", "theta_min", "theta_max", "dtheta_max", "rrms", "rmax", "converged"
        ])
        self.writer.writeheader()
        if self.save_slices:
            self.Ixz = np.memmap(self.run_dir / "Ixz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Nx))
            self.Iyz = np.memmap(self.run_dir / "Iyz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Ny))
            self.dthetaxz = np.memmap(self.run_dir / "dthetaxz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Nx))
            self.dthetayz = np.memmap(self.run_dir / "dthetayz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Ny))
        if self.save_full:
            self.theta_final = np.memmap(self.run_dir / "theta_final.dat", dtype=np.float32, mode="w+", shape=(ctx.Nz, ctx.Nx, ctx.Ny))
    
    def save(self, slot: int, k: int, I, theta, info: Dict[str, Any]):
        ctx = self.ctx
        xp = ctx.xp
        Icpu = asnumpy(I).astype(np.float32, copy=False)
        thcpu = asnumpy(theta).astype(np.float32, copy=False)
        bcpu = asnumpy(ctx.theta_bias_2d).astype(np.float32, copy=False)
        if self.save_slices:
            self.Ixz[slot, k, :] = Icpu[:, ctx.Ny // 2]
            self.Iyz[slot, k, :] = Icpu[ctx.Nx // 2, :]
            dth = thcpu - bcpu
            self.dthetaxz[slot, k, :] = dth[:, ctx.Ny // 2]
            self.dthetayz[slot, k, :] = dth[ctx.Nx // 2, :]
        if self.save_full:
            self.theta_final[k] = thcpu
        self.writer.writerow(dict(
            jt=slot, k=k, z_um=k * ctx.dz,
            Imax=float(Icpu.max()), power=float(Icpu.sum() * ctx.dx * ctx.dy),
            theta_min=float(thcpu.min()), theta_max=float(thcpu.max()),
            dtheta_max=float(np.max(np.abs(thcpu - bcpu))),
            rrms=float(info.get("rrms", np.nan)), rmax=float(info.get("rmax", np.nan)),
            converged=bool(info.get("converged", False)),
        ))

    def close(self):
        self.log_file.flush()
        self.log_file.close()
        if self.save_slices:
            self.Ixz.flush(); self.Iyz.flush(); self.dthetaxz.flush(); self.dthetayz.flush()
        if self.save_full:
            self.theta_final.flush()


# -----------------------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------------------

def run_lc_validated(
    params: LCParams,
    *,
    run_dir: str | Path,
    mode: str = "strict_static",
    Nt: int = 100,
    dt: float = 5e-4,
    t_stride: int = 1,
    save_slices: bool = True,
    save_full: bool = False,
    progress: Optional[Callable[[str], None]] = print,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run a validated LC demo calculation and write lightweight outputs."""
    t0 = time.time()

    mode = mode.lower().strip()
    if mode not in ("static_fast", "strict_static", "td_predictor_only", "dg_td_predictor"):
        raise ValueError("mode must be static_fast, strict_static, td_predictor_only, or dg_td_predictor")

    if mode == "dg_td_predictor":
        params = LCParams(**{**asdict(params), "use_dual_grid": True})

    ctx, dg = make_context(params)
    xp = ctx.xp

    #du_lc, dv_lc = lc_grid_spacings(ctx)
    
    nsub, dz_sub, phi = choose_optics_substeps(ctx)
    h_sub = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)

    timedep = mode in ("td_predictor_only", "dg_td_predictor")
    Nt_eff = int(Nt) if timedep else 1
    t_stride = max(1, int(t_stride))
    Nt_out = (Nt_eff + t_stride - 1) // t_stride

    run_dir = Path(run_dir)
    store = LightStore(
        run_dir,
        ctx,
        Nt_out=Nt_out,
        save_slices=save_slices,
        save_full=save_full,
    )

    meta = asdict(params)
    meta.update(
        dict(
            mode=mode,
            Nt=Nt_eff,
            dt=dt,
            t_stride=t_stride,
            nsub=nsub,
            dz_sub=dz_sub,
            phi_est=phi,
            backend="cupy" if (_HAS_CUPY and xp is _cupy) else "numpy",
        )
    )

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    theta_stack = [ctx.theta_bias_2d.copy() for _ in range(ctx.Nz)]

    if dg is not None:
        theta_c_stack = [dg.theta_bias_c.copy() for _ in range(ctx.Nz)]
    else:
        theta_c_stack = None
    stopped = False
    try:
        for jt in range(Nt_eff):
            if should_stop is not None and should_stop():
                stopped = True
                if progress:
                    progress("run stopped by user")
                break
            amp = hop_linear(ctx, ctx.amp0.copy(), h_half)
            theta_prev = list(theta_stack)
            slot = jt // t_stride

            last_Imax = np.nan
            last_Rrms = np.nan

            for k in range(ctx.Nz):
                if should_stop is not None and should_stop():
                    stopped = True
                    if progress:
                        progress("run stopped by user")
                    break




                
                # --------------------------------------------------
                #  STATIC
                # --------------------------------------------------
                if mode in ("static_fast", "strict_static"):
                    if mode == "static_fast":
                        theta, I_mid, amp, info = static_fast_slice_selfconsistent(
                            ctx,
                            amp,
                            nsub=nsub,
                            dz_sub=dz_sub,
                            h_sub=h_sub,
                        )
                    
                    else:  # strict_static
                        theta_seed = theta_stack[k - 1] if k > 0 else ctx.theta_bias_2d.copy()
                    
                        theta, I_mid, amp, info = strict_static_slice_selfconsistent(
                            ctx,
                            amp,
                            theta_seed,
                            nsub=nsub,
                            dz_sub=dz_sub,
                            h_sub=h_sub,
                        )
                    
                    theta_stack[k] = theta
                    I_save = I_mid

                # --------------------------------------------------
                # FULL-GRID TD PREDICTOR ONLY
                # --------------------------------------------------
                elif mode == "td_predictor_only":
                    theta_old = theta_prev[k]

                    I_b = intensity(amp, params.coherent, xp=xp)
                    amp = propagate_slice(
                        ctx,
                        amp,
                        theta_old,
                        nsub=nsub,
                        dz_sub=dz_sub,
                        h_sub=h_sub,
                    )
                    I_a = intensity(amp, params.coherent, xp=xp)
                    I_mid = 0.5 * (I_b + I_a)

                    tp = theta_prev[k - 1] if k > 0 else theta_old
                    tn = theta_prev[k + 1] if k + 1 < ctx.Nz else theta_old

                    if False:
                        theta = theta_td_predictor_only(
                            theta_old,
                            I_mid,
                            ctx,
                            dt=dt,
                            theta_prev=tp,
                            theta_next=tn,
                            du=ctx.du_lc,
                            dv=ctx.dv_lc,
                            b=params.b,
                            bi=params.bi,
                            mobility=params.mobility,
                            theta_bc=params.theta_bc,
                            gamma_z=params.theta_z_gamma,
                        )
                    

                    bias = ctx.theta_bias_2d
                    eta_old = theta_old - bias
                    
                    drive = params.bi * I_mid * xp.sin(2.0 * bias)
                    
                    a_ie, off, diag, _ = prepare_ie_operator(
                        dt,
                        params.mobility,
                        ctx.du_lc,
                        ctx.dv_lc,
                        ctx.Ny,
                        xp,
                    )
                    
                    rhs = eta_old + xp.float32(dt / params.mobility) * drive
                    
                    eta_new = solve_implicit_xy(
                        rhs,
                        off=off,
                        diag=diag,
                        xp=xp,
                        theta_bc=0.0,
                    )
                        
                    theta = enforce_theta(bias + eta_new, ctx)

                    
                    R = residual(
                        theta,
                        I_mid,
                        ctx,
                        du=ctx.du_lc,
                        dv=ctx.dv_lc,
                        b=params.b,
                        bi=params.bi,
                        mobility=params.mobility,
                        theta_bc=params.theta_bc,
                    )
                    Rint = R[1:-1, :]

                    info = {
                        "rrms": float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
                        "rmax": float(asnumpy(xp.max(xp.abs(Rint)))),
                        "converged": True,
                    }

                    theta_stack[k] = theta
  
                    I_save = I_mid

                # --------------------------------------------------
                # DUAL-GRID TD PREDICTOR ONLY
                # --------------------------------------------------
                else:
                    assert dg is not None
                    assert theta_c_stack is not None

                    theta_old_c = theta_c_stack[k]
                    theta_old_f = prolong_repeat(
                        theta_old_c,
                        dg.factor,
                        target_shape=(ctx.Nx, ctx.Ny),
                        xp=xp,
                    )

                    I_b = intensity(amp, params.coherent, xp=xp)
                    amp = propagate_slice(
                        ctx,
                        amp,
                        theta_old_f,
                        nsub=nsub,
                        dz_sub=dz_sub,
                        h_sub=h_sub,
                    )
                    I_a = intensity(amp, params.coherent, xp=xp)

                    I_mid_f = 0.5 * (I_b + I_a)
                    I_mid_c = restrict_block_mean(I_mid_f, dg.factor, xp=xp)

                    class C:
                        pass

                    c = C()
                    c.xp = xp
                    c.dz = ctx.dz
                    c.p = params
                    c.du_lc = ctx.du_lc * dg.factor
                    c.dv_lc = ctx.dv_lc * dg.factor
                    theta_c = theta_td_predictor_only(
                        theta_old_c,
                        I_mid_c,
                        c,
                        dt=dt,
                        du=ctx.du_lc * dg.factor,
                        dv=ctx.dv_lc * dg.factor,
                        b=params.b,
                        bi=params.bi,
                        mobility=params.mobility,
                        theta_bc=params.theta_bc,
                        gamma_z=0.0,
                    )

                    theta_c_stack[k] = theta_c

                    theta_f = prolong_repeat(
                        theta_c,
                        dg.factor,
                        target_shape=(ctx.Nx, ctx.Ny),
                        xp=xp,
                    )

                    theta_stack[k] = theta_f

                    R = residual(
                        theta_f,
                        I_mid_f,
                        ctx,
                        du=ctx.du_lc,
                        dv=ctx.dv_lc,
                        b=params.b,
                        bi=params.bi,
                        mobility=params.mobility,
                        theta_bc=params.theta_bc,
                    )
                    Rint = R[1:-1, :]

                    info = {
                        "rrms": float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
                        "rmax": float(asnumpy(xp.max(xp.abs(Rint)))),
                        "converged": True,
                    }

                    I_save = I_mid_f

                # --------------------------------------------------
                # Save lightweight outputs
                # --------------------------------------------------
                if (not timedep) or (jt % t_stride == 0):
                    store.save(slot, k, I_save, theta_stack[k], info)

                last_Imax = float(asnumpy(xp.max(I_save)))
                last_Rrms = float(info.get("rrms", np.nan))

                # --------------------------------------------------
                # Static progress: z-status only
                # --------------------------------------------------
                if (
                    progress
                    and not timedep
                    and (
                        k == 0
                        or (k + 1) % max(1, ctx.Nz // 20) == 0
                        or k == ctx.Nz - 1
                    )
                ):
                    progress(
                        f"static z {k+1}/{ctx.Nz}  "
                        f"Imax={last_Imax:.3e}  "
                        f"Rrms={last_Rrms:.3e}"
                    )
            if stopped:
                break
            # ------------------------------------------------------
            # TD progress: time-step status only
            # ------------------------------------------------------
            if progress and timedep:
                progress(
                    f"time step {jt+1}/{Nt_eff}  "
                    f"mode={mode}  "
                    f"t={(jt+1)*dt:.4g}  "
                    f"Imax={last_Imax:.3e}  "
                    f"Rrms={last_Rrms:.3e}"
                )

            free_backend_memory(xp)

        np.savez(
            run_dir / "final_summary.npz",
            elapsed_s=time.time() - t0,
            stopped=stopped,
        )
        if stopped and progress:
            progress("run stopped by user")

    finally:
        store.close()

    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "metadata.json"),
        "scalar_log": str(run_dir / "scalar_log.csv"),
    }


# -----------------------------------------------------------------------------
# Viewer helpers
# -----------------------------------------------------------------------------

def load_run_arrays(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    Nx, Ny, Nz = int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])
    Nt_out = (int(meta.get("Nt", 1)) + int(meta.get("t_stride", 1)) - 1) // int(meta.get("t_stride", 1))
    out = {"metadata": meta, "run_dir": run_dir}
    for name, shape in {
        "Ixz": (Nt_out, Nz, Nx), "Iyz": (Nt_out, Nz, Ny),
        "dthetaxz": (Nt_out, Nz, Nx), "dthetayz": (Nt_out, Nz, Ny),
    }.items():
        path = run_dir / f"{name}.dat"
        if path.exists():
            out[name] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
    return out
