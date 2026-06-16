"""Clean extracted LC soliton core code from the latest working notebook.

This module intentionally keeps the numerics close to the trusted notebook while
removing notebook UI, ad-hoc execution cells, and Streamlit/package wrappers.
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from lc_soliton.core.backend import xp_default as cp

try:
    import cupyx.scipy.fft as spfft
except ImportError:
    import scipy.fft as spfft

try:
    from cupyx.scipy.ndimage import gaussian_filter
except ImportError:
    from scipy.ndimage import gaussian_filter
import scipy.special as spspec
from scipy.optimize import root_scalar
from scipy.signal.windows import tukey
from scipy.ndimage import zoom as sp_zoom

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

j = 1j


def genrot(rlen, thout, phi, x, y, *, refin, z_focus=0.0):
    x = cp.asarray(x)
    y = cp.asarray(y)
    th = cp.arcsin(cp.sin(thout) / refin)
    sP, cP = cp.sin(phi), cp.cos(phi)
    st, ct = cp.sin(th), cp.cos(th)
    el = cp.asarray(z_focus, dtype=cp.float32)

    xp = x[:, None] + cP * (1 - ct) * (-x[:, None] * cP + y[None, :] * sP) - el * cP * st
    yp = y[None, :] + sP * (1 - ct) * (x[:, None] * cP - y[None, :] * sP) + el * sP * st
    zp = (-el) * ct + (-x[:, None] * cP + y[None, :] * sP) * st
    return xp, yp, zp

def getx0y0(coord):
    xp1, yp1 = coord[0], coord[1]
    shape = xp1.shape
    x0, y0 = cp.unravel_index(cp.argmin(xp1**2 + yp1**2), shape)
    return int(x0), int(y0)

def gaus(coord, beam, arrin, w0x, w0y, prdata):
    lm = prdata['lm']
    refin = prdata['refin']
    xsamp = int(prdata['xsamp'])
    ysamp = int(prdata['ysamp'])
    x, y, z = coord

    kin = refin * 2 * cp.pi / lm
    z0x = w0x**2 * kin / 2.0
    z0y = w0y**2 * kin / 2.0
    eta = (cp.arctan(z / z0x) + cp.arctan(z / z0y)) / 2
    rlxinv = z / (z**2 + z0x**2)
    rlyinv = z / (z**2 + z0y**2)
    wl2x = w0x**2 * (1.0 + (z / z0x)**2)
    wl2y = w0y**2 * (1.0 + (z / z0y)**2)
    wlxy = cp.sqrt((1.0 + (z / z0x)**2) * (1.0 + (z / z0y)**2))
    argx = (x**2) * (1.0 / wl2x - 1j * kin * rlxinv / 2.0)
    argy = (y**2) * (1.0 / wl2y - 1j * kin * rlyinv / 2.0)
    arg = argx + argy

    if w0x > 0:
        amp1 = cp.exp(-arg + 1j * kin * z - 1j * eta) / cp.sqrt(wlxy)
    else:
        amp1 = cp.exp(1j * kin * z)

    transp = prdata['transp1'] if beam == 1 else prdata['transp2']
    phasetransp = prdata['realimage'] == "phase image"

    if transp:
        lenx, leny = cp.shape(arrin)
        xindex1, yindex1 = getx0y0(coord)
        ampxlo = max(xindex1 - lenx // 2, 0)
        ampylo = max(yindex1 - leny // 2, 0)
        ampxhi = min(xindex1 + lenx // 2, xsamp - 1)
        ampyhi = min(yindex1 + leny // 2, ysamp - 1)
        arrxlo = lenx // 2 - (xindex1 - ampxlo)
        arrxhi = lenx // 2 + ampxhi - xindex1
        arrylo = leny // 2 - (yindex1 - ampylo)
        arryhi = leny // 2 + ampyhi - yindex1

        if not phasetransp:
            amp1[ampxlo:ampxhi, ampylo:ampyhi] = (
                cp.sqrt(arrin[arrxlo:arrxhi, arrylo:arryhi]) *
                amp1[ampxlo:ampxhi, ampylo:ampyhi]
            )
        else:
            amp1[ampxlo:ampxhi, ampylo:ampyhi] = (
                cp.exp(1j * cp.pi * arrin[arrxlo:arrxhi, arrylo:arryhi]) *
                amp1[ampxlo:ampxhi, ampylo:ampyhi]
            )

    return amp1

def build_amp_pair(arrin, coord1, coord2, prdata):
    rat = float(prdata.get("rat", 1.0))
    a1r = cp.sqrt(1.0 / (1.0 + rat))
    a2r = cp.sqrt(rat / (1.0 + rat))

    w0x1 = float(prdata.get("w01x", 100.0))
    w0y1 = float(prdata.get("w01y", 100.0))
    w0x2 = float(prdata.get("w02x", 100.0))
    w0y2 = float(prdata.get("w02y", 100.0))

    image_on_beam = prdata.get("image_on_beam", "No Image")
    prdata["transp1"] = image_on_beam in ("Beam 1", "Beams 1 & 2")
    prdata["transp2"] = image_on_beam in ("Beam 2", "Beams 1 & 2")

    A1 = gaus(coord1, 1, arrin, w0x1, w0y1, prdata)
    A2 = gaus(coord2, 2, arrin, w0x2, w0y2, prdata)
    return a1r * A1, a2r * A2

def compute_n_bg_from_bias(theta_bias, ne, no, xp_mod, reducer="median"):
    xp = xp_mod
    ct = xp.cos(theta_bias)
    st = xp.sin(theta_bias)
    inv_neff2 = (ct * ct) / (no * no) + (st * st) / (ne * ne)
    neff_bias = 1.0 / xp.sqrt(inv_neff2)
    return float(xp.mean(neff_bias)) if reducer == "mean" else float(xp.median(neff_bias))

def build_theta_bias_IC(
    Nx, Ny, Nz, b,
    *,
    eps_clip=1e-12,
    return_1d=False,
    return_2d=False,
    dtype=cp.float32,
    kick_eps=1e-4,
    kick_seed=1234,
):
    xp = cp
    b_c = xp.pi**2 / 8.0

    if b <= float(b_c):
        theta_1d = xp.zeros(Nx, dtype=dtype)
        if return_1d:
            return theta_1d
        theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)
        if float(kick_eps) > 0.0:
            rng = xp.random.default_rng(int(kick_seed))
            kick = (float(kick_eps) * rng.standard_normal((Nx, Ny), dtype=cp.float32)).astype(dtype, copy=False)
            kick[0, :] = 0.0
            kick[-1, :] = 0.0
            theta_2d = (theta_2d + kick).astype(dtype, copy=False)
        theta_2d[0, :] = 0.0
        theta_2d[-1, :] = 0.0
        if return_2d:
            return theta_2d
        return xp.repeat(theta_2d[:, :, None], Nz, axis=2)

    u = xp.linspace(-1.0, 1.0, Nx, dtype=cp.float64)
    two_b = 2.0 * float(b)

    def Ksq_minus_2b(m):
        Km = spspec.ellipk(m)
        return float(Km * Km - two_b)

    m_lo, m_hi = 1e-12, 1.0 - 1e-12
    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)
        if Ksq_minus_2b(m_lo) * Ksq_minus_2b(m_mid) <= 0.0:
            m_hi = m_mid
        else:
            m_lo = m_mid
        if abs(m_hi - m_lo) < 1e-14:
            break
    m = 0.5 * (m_lo + m_hi)

    theta0 = xp.arcsin(xp.sqrt(m))
    arg = xp.sqrt(2.0 * b) * u
    arg_cpu = cp.asnumpy(arg)
    _, cn_cpu, dn_cpu, _ = spspec.ellipj(arg_cpu, float(m))
    cn = xp.asarray(cn_cpu)
    dn = xp.asarray(dn_cpu)
    cd = cn / dn
    s = xp.sin(theta0) * cd
    s = xp.clip(s, -1.0 + eps_clip, 1.0 - eps_clip)
    theta_1d = xp.arcsin(s).astype(dtype, copy=False)

    if return_1d:
        return theta_1d
    theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)
    theta_2d[0, :] = 0.0
    theta_2d[-1, :] = 0.0
    if return_2d:
        return theta_2d
    return xp.repeat(theta_2d[:, :, None], Nz, axis=2)


def build_theta_bias_IC_dirichlet_value(
    Nx, Ny, Nz, b,
    *,
    theta_bc=0.0,
    eps_clip=1e-12,
    return_1d=False,
    return_2d=False,
    dtype=cp.float32,
):
  
    xp = cp

    u = xp.linspace(-1.0, 1.0, Nx, dtype=cp.float64)
    two_b = 2.0 * float(b)

    sbc = float(np.sin(theta_bc))

    # ---------------------------------------
    # solve for modulus m
    # ---------------------------------------

    if abs(theta_bc) < 1e-14:

        def f(m):
            Km = spspec.ellipk(m)
            return Km * Km - two_b

    else:

        sqrt2b = np.sqrt(two_b)

        def f(m):
            sn, cn, dn, ph = spspec.ellipj(sqrt2b, m)
            cd = cn / dn
            return np.sqrt(m) * cd - sbc

    m_lo = 1e-12
    m_hi = 1.0 - 1e-12

    flo = f(m_lo)
    fhi = f(m_hi)

    if flo * fhi > 0:
        raise RuntimeError(
            f"Could not bracket bias modulus: "
            f"f({m_lo})={flo}, f({m_hi})={fhi}"
        )

    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)

        if f(m_lo) * f(m_mid) <= 0:
            m_hi = m_mid
        else:
            m_lo = m_mid

        if abs(m_hi - m_lo) < 1e-14:
            break

    m = 0.5 * (m_lo + m_hi)

    # ---------------------------------------
    # evaluate profile
    # ---------------------------------------

    theta0 = xp.arcsin(xp.sqrt(m))

    arg = xp.sqrt(two_b) * u

    arg_cpu = cp.asnumpy(arg)

    _, cn_cpu, dn_cpu, _ = spspec.ellipj(arg_cpu, float(m))

    cn = xp.asarray(cn_cpu)
    dn = xp.asarray(dn_cpu)

    cd = cn / dn

    s = xp.sin(theta0) * cd

    s = xp.clip(
        s,
        -1.0 + eps_clip,
        1.0 - eps_clip,
    )

    theta_1d = xp.arcsin(s).astype(dtype, copy=False)

    if return_1d:
        return theta_1d

    theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)

    theta_2d[0, :] = cp.float32(theta_bc)
    theta_2d[-1, :] = cp.float32(theta_bc)

    if return_2d:
        return theta_2d

    return xp.repeat(theta_2d[:, :, None], Nz, axis=2)


    
def build_theta_bias_IC_dirichlet_value_original(
    Nx, Ny, Nz, b,
    *,
    theta_bc=0.0,
    du=None,
    dv=None,
    mobility=1.0,
    max_iter=5000,
    tol_rms=1e-8,
    tol_max=1e-6,
    dtau=0.05,
    report_every=100,
    verbose=True,
    return_1d=False,
    return_2d=False,
    dtype=cp.float32,
):


# Fast y-uniform bias path
    if False:
        theta_x = cp.full((Nx,), cp.float32(theta_bc), dtype=cp.float32)
        theta_x[0] = cp.float32(theta_bc)
        theta_x[-1] = cp.float32(theta_bc)
    
        a_ie = cp.float32(float(dtau) / float(mobility))
        off = cp.float32((-a_ie) / (du * du))
        diag = cp.full(
            (Nx - 2,),
            cp.float32(1.0 + 2.0 * float(a_ie) / (du * du)),
            dtype=cp.float32,
        )
    
        last_stats = None
    
        for it in range(int(max_iter)):
            drive = cp.float32(b) * cp.sin(2.0 * theta_x)
            rhs = theta_x + a_ie * drive
    
            d = rhs[1:-1].astype(cp.float32, copy=True)
            d[0]  -= off * cp.float32(theta_bc)
            d[-1] -= off * cp.float32(theta_bc)
    
            theta_inner = thomas_const_tridiag_1d(off, diag, off, d)
    
            theta_x[1:-1] = theta_inner
            theta_x[0] = cp.float32(theta_bc)
            theta_x[-1] = cp.float32(theta_bc)
    
            if (it % report_every == 0) or (it == max_iter - 1):
                lap = (theta_x[2:] - 2.0 * theta_x[1:-1] + theta_x[:-2]) / cp.float32(du * du)
                Rint = lap + cp.float32(b) * cp.sin(2.0 * theta_x[1:-1])
    
                stats = {
                    "max_interior": float(cp.max(cp.abs(Rint))),
                    "rms_interior": float(cp.sqrt(cp.mean(Rint.astype(cp.float64) ** 2))),
                }
                last_stats = stats
    
                if verbose:
                    print(
                        f"[bias-1d theta_bc={theta_bc:.4g}] it={it:6d} "
                        f"Rmax={stats['max_interior']:.3e} "
                        f"Rrms={stats['rms_interior']:.3e}"
                    )
    
                if stats["rms_interior"] < tol_rms and stats["max_interior"] < tol_max:
                    break
    
        if return_1d:
            return theta_x.astype(dtype, copy=False)
    
        theta_2d = cp.repeat(theta_x[:, None], int(Ny), axis=1)
    
        if return_2d:
            return theta_2d.astype(dtype, copy=False)
    
        return cp.repeat(theta_2d[:, :, None], int(Nz), axis=2).astype(dtype, copy=False)
    
    
    def prepare_ie_ky_operator(*, dt, mobility, du, dv, Ny):
        a_ie = float(dt) / float(mobility)
        lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
        off = cp.float32((-a_ie) * (1.0 / (du * du)))
        diag = (1.0 + (2.0 * a_ie) * (1.0 / (du * du)) - a_ie * lam_y).astype(cp.float32, copy=False)
        return cp.float32(a_ie), off, diag, lam_y
    theta_bc = float(theta_bc)

    if du is None:
        du = 2.0 / (Nx - 1)
    if dv is None:
        dv = 1.0

    theta = cp.full((Nx, Ny), cp.float32(theta_bc), dtype=cp.float32)
    theta[0, :] = cp.float32(theta_bc)
    theta[-1, :] = cp.float32(theta_bc)

    a_ie, off, diag, lam_y = prepare_ie_ky_operator(
        dt=float(dtau),
        mobility=float(mobility),
        du=float(du),
        dv=float(dv),
        Ny=int(Ny),
    )

    bc_hat = cp.fft.fft(
        cp.full((Ny,), cp.float32(theta_bc), dtype=cp.float32)
    ).astype(cp.complex64)

    def solve_ie(rhs):
        rhs2 = rhs.astype(cp.float32, copy=True)
        rhs2[0, :] = 0.0
        rhs2[-1, :] = 0.0

        rhs_hat = cp.fft.fft(rhs2, axis=1).astype(cp.complex64, copy=False)
        d_hatB = rhs_hat[1:-1, :].T.copy(order="C")

        # nonzero Dirichlet boundary contribution
        d_hatB[:, 0]  -= off * bc_hat
        d_hatB[:, -1] -= off * bc_hat

        x_hatB = thomas_batched_const_tridiag(off, diag, off, d_hatB)

        theta_hat = cp.zeros_like(rhs_hat)
        theta_hat[1:-1, :] = x_hatB.T

        out = cp.fft.ifft(theta_hat, axis=1).real.astype(cp.float32, copy=False)
        out[0, :] = cp.float32(theta_bc)
        out[-1, :] = cp.float32(theta_bc)
        return out

    I0 = cp.zeros((Nx, Ny), dtype=cp.float32)

    last_stats = None

    for it in range(int(max_iter)):
        drive = cp.float32(b) * cp.sin(2.0 * theta)

        rhs = theta + cp.float32(float(dtau) / float(mobility)) * drive
        theta = solve_ie(rhs)

        if (it % report_every == 0) or (it == max_iter - 1):
            R = lc_residual2d_dirichletx_periody(
                theta, I0,
                b=float(b),
                bi=0.0,
                du=float(du),
                dv=float(dv),
                mobility=float(mobility),
            )
            Rint = R[1:-1, :].astype(cp.float64, copy=False)

            stats = {
                "max_interior": float(cp.max(cp.abs(Rint))),
                "rms_interior": float(cp.sqrt(cp.mean(Rint * Rint))),
            }
            last_stats = stats

            if verbose:
                print(
                    f"[bias theta_bc={theta_bc:.4g}] it={it:6d} "
                    f"Rmax={stats['max_interior']:.3e} "
                    f"Rrms={stats['rms_interior']:.3e}"
                )

            if stats["rms_interior"] < tol_rms and stats["max_interior"] < tol_max:
                break

    if return_1d:
        return theta[:, 0].astype(dtype, copy=False)

    if return_2d:
        return theta.astype(dtype, copy=False)

    return cp.repeat(theta[:, :, None], int(Nz), axis=2).astype(dtype, copy=False)

