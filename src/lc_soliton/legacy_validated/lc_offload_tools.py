
# Auto-scraped from notebook namespace.
# First production-offload version.

from pathlib import Path
from dataclasses import dataclass

import os, gc, json, time, math, traceback

import numpy as np
import pandas as pd
import cupy as cp

from PIL import Image, ImageChops

from scipy.ndimage import zoom as sp_zoom
from scipy.signal.windows import tukey

import cupyx.scipy.fft as spfft
import scipy.special as spspec
from tqdm.auto import tqdm


@dataclass
class LCContext:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




# ================================================================================

# initialize_lc_from_prdata

def initialize_lc_from_prdata(prdata, restart_theta=True):
    eps0 = 8.854e-12
    c0 = 3e8

    lm = float(prdata.get("lm", 0.633))
    Nx = int(prdata.get("xsamp", 512))
    Ny = int(prdata.get("ysamp", 512))
    yaper = float(prdata.get("yaper", 1000.0))
    rlen = float(prdata.get("rlen", 4000.0))
    dz = float(prdata.get("dz", 20.0))
    windowedge = float(prdata.get("windowedge", 0.1))

    d_um = float(prdata.get("d", 75.0))
    K_SI = float(prdata.get("K", 7e-12))
    De_rel = float(prdata.get("De", 13.0))
    bias_V = float(prdata.get("bias_voltage", 4.0))
    ne = float(prdata.get("ne", 1.7))
    no = float(prdata.get("no", 1.5))

    P_mW = float(prdata.get("P", 1.0))
    P_W = P_mW * 1e-3

    thout1 = float(prdata.get("thout1", 0.1))
    thout2 = float(prdata.get("thout2", -0.1))
    phi1 = float(prdata.get("phi1", 0.0))
    phi2 = float(prdata.get("phi2", 0.0))
    xoffset = float(prdata.get("xoffset", 0.0))
    yoffset = float(prdata.get("yoffset", 0.0))
    soliton_pair_sep = float(prdata.get("soliton_pair_sep", 0.0))
    soliton_pair_angle = float(prdata.get("soliton_pair_angle", 0.0))
    coh = bool(prdata.get("coh", True))

    timedep = (prdata.get("time_behavior", "Static") == "Time Dependent")
    tend = float(prdata.get("tend", 10.0))
    tsteps = int(prdata.get("tsteps", 120)) if timedep else 1
    use_cons_tsteps = bool(prdata.get("use_cons_tsteps", False))
    t_stride = int(prdata.get("t_stride", 2))

    xaper = float(d_um)
    prdata["xaper"] = xaper

    Nz = max(1, int(round(rlen / dz)))
    prdata["niter"] = Nz

    du = 2.0 / (Nx - 1)
    dv = (2.0 * (yaper / d_um)) / Ny
    prdata["du"] = float(du)
    prdata["dv"] = float(dv)

    dx = xaper / Nx
    dy = yaper / Ny
    x = (cp.arange(Nx, dtype=cp.float32) - Nx / 2) * dx + 0.5 * dx
    y = (cp.arange(Ny, dtype=cp.float32) - Ny / 2) * dy + 0.5 * dy
    z = cp.arange(Nz, dtype=cp.float32) * dz

    d_m = d_um * 1e-6
    De_SI = De_rel * eps0
    E_SI = bias_V / d_m
    b = float((De_SI * (E_SI**2) * (d_m**2)) / (8.0 * K_SI))
    prdata["b"] = b

    na2 = ne**2 - no**2
    bi = float((na2 * (d_m**2) * 1e12 * P_W) / (8.0 * c0 * K_SI))
    prdata["bi"] = bi

    fx = cp.fft.fftfreq(Nx, d=dx).astype(cp.float32, copy=False)
    fy = cp.fft.fftfreq(Ny, d=dy).astype(cp.float32, copy=False)
    fxy2 = (fx[:, None]**2 + fy[None, :]**2).astype(cp.float32, copy=False)
    kout = float(2.0 * math.pi / lm)

    if prdata.get("theta_bc", 0.0) == 0.0:

        theta_bias_2d = build_theta_bias_IC(
            Nx, Ny, 1, b,
            kick_eps=0.01,
            eps_clip=1e-12,
            return_2d=True
        ).astype(cp.float32, copy=False)
        theta_bias_2d[0, :] = 0.0
        theta_bias_2d[-1, :] = 0.0
    else:

        theta_bc = float(prdata.get("theta_bc", 0.0))


        if theta_bc == 0.0:
            theta_bias_2d = build_theta_bias_IC(
                Nx, Ny, 1, b,
                kick_eps=0.01,
                eps_clip=1e-12,
                return_2d=True
            ).astype(cp.float32, copy=False)
        else:
            theta_bias_2d = build_theta_bias_IC_dirichlet_value(
                Nx, Ny, 1, b,
                theta_bc=theta_bc,
                du=du,
                dv=dv,
                mobility=prdata.get("mobility", 1.0),
                max_iter=int(prdata.get("theta_bias_max_iter", 50000)),
                tol_rms=float(prdata.get("theta_bias_tol_rms", 1e-8)),
                tol_max=float(prdata.get("theta_bias_tol_max", 1e-6)),
                report_every=int(prdata.get("theta_bias_report_every", 1000)),
                verbose=bool(prdata.get("theta_bias_verbose", True)),
                return_2d=True,
                dtype=cp.float32,
            ).astype(cp.float32, copy=False)

        # 🔴 CRITICAL: enforce BC unconditionally
        theta_bias_2d[0, :]  = cp.float32(theta_bc)
        theta_bias_2d[-1, :] = cp.float32(theta_bc)
        edge0 = float(cp.mean(theta_bias_2d[0, :]))
        edge1 = float(cp.mean(theta_bias_2d[-1, :]))

        if not (abs(edge0 - theta_bc) < 1e-6 and abs(edge1 - theta_bc) < 1e-6):
            raise RuntimeError(
                f"theta_bias mismatch: edges ({edge0}, {edge1}) vs theta_bc={theta_bc}"
            )

    n_bg = compute_n_bg_from_bias(theta_bias_2d, ne, no, cp, reducer="median")
    refin = float(n_bg)
    prdata["n_bg"] = refin
    prdata["refin"] = refin

    windowx = tukey(Nx, alpha=windowedge, sym=False).astype(np.float32, copy=False)
    windowy = tukey(Ny, alpha=windowedge, sym=False).astype(np.float32, copy=False)
    windowxy = cp.asarray(np.sqrt(np.outer(windowx, windowy)), dtype=cp.float32)

    image_file = None
    if prdata.get("image_on_beam", "No Image") != "No Image":
        external_image = str(prdata.get("external_image", "")).strip()
        std_image = str(prdata.get("std_image", "MNIST 0"))
        std_dir = Path(str(prdata.get("std_image_dir", ""))).expanduser()
        std_map = {
            "MNIST 0": "mnist0.png", "MNIST 1": "mnist1.png", "MNIST 2": "mnist2.png",
            "MNIST 3": "mnist3.png", "MNIST 4": "mnist4.png", "MNIST 5": "mnist5.png",
            "MNIST 6": "mnist6.png", "MNIST 7": "mnist7.png", "MNIST 8": "mnist8.png",
            "MNIST 9": "mnist9.png", "AF Res Chart": "AF Res Chart.png",
        }
        image_file = Path(external_image).expanduser() if external_image else std_dir / std_map.get(std_image, "")
        if (image_file is None) or (not image_file.exists()):
            raise FileNotFoundError(f"Image file not found: {image_file}")

    arrin = cp.ones((28, 28), dtype=cp.float32)
    if image_file is not None:
        img = Image.open(str(image_file)).convert("L")
        if bool(prdata.get("image_invert", False)):
            img = ImageChops.invert(img)
        img_cpu = np.array(img, dtype=np.float32)

        imsizex, imsizey = img_cpu.shape
        maxsize = int(max(imsizex, imsizey))
        if maxsize % 2:
            maxsize += 1

        image_sq = np.ones((maxsize, maxsize), dtype=np.float32) * float(img_cpu.max() if img_cpu.size else 1.0)
        ox = (maxsize - imsizex) // 2
        oy = (maxsize - imsizey) // 2
        image_sq[ox:ox+imsizex, oy:oy+imsizey] = img_cpu

        zoomsc = float(prdata.get("image_size_factor", 1.0)) * (xaper / 2.0) / maxsize
        zx = zoomsc * Nx / xaper
        zy = zoomsc * Ny / yaper

        arrin_cpu = sp_zoom(np.rot90(image_sq, k=1), (zx, zy), order=0).astype(np.float32, copy=False)
        m = float(arrin_cpu.max()) if arrin_cpu.size else 1.0
        if m > 0:
            arrin_cpu /= m
        arrin = cp.asarray(arrin_cpu, dtype=cp.float32)

    ang_rad = math.radians(soliton_pair_angle)
    xoffset1 = xoffset + soliton_pair_sep * math.cos(ang_rad)
    xoffset2 = xoffset - soliton_pair_sep * math.cos(ang_rad)
    yoffset1 = yoffset + soliton_pair_sep * math.sin(ang_rad)
    yoffset2 = yoffset - soliton_pair_sep * math.sin(ang_rad)

    z_focus = 0.0 if bool(prdata.get("soliton", False)) else (rlen / 2.0)
    coord1 = genrot(rlen, thout1, phi1, x + xoffset1, y + yoffset1, refin=refin, z_focus=z_focus)
    coord2 = genrot(rlen, thout2, phi2, x + xoffset2, y + yoffset2, refin=refin, z_focus=z_focus)

    amp0m, amp0p = build_amp_pair(arrin, coord1, coord2, prdata)
    amp0 = cp.asarray((amp0m, amp0p), dtype=cp.complex64)

    support = (lm**2 * fxy2 < 1.0)
    A0 = spfft.fft2(amp0, axes=(-2, -1))
    A0 = cp.where(support[None, :, :], A0, 0)
    amp0 = spfft.ifft2(A0, axes=(-2, -1)).astype(cp.complex64, copy=False)

    I0 = intens(amp0, coh).astype(cp.float32, copy=False)
    norm0 = cp.sum(I0) * cp.float32(dx * dy)
    norm0 = cp.where(norm0 == 0.0, 1.0, norm0)
    amp0 = (amp0 / cp.sqrt(norm0)).astype(cp.complex64, copy=False)

    theta_full = cp.empty((Nz, Nx, Ny), dtype=cp.float32)
    if restart_theta:
        theta_full[...] = theta_bias_2d[None, :, :]

    theta_prev_time = cp.empty_like(theta_full)

    dt = float(tend) / max(int(tsteps), 1)
    if use_cons_tsteps and timedep:
        dt = dt / 2.0
        tsteps = int(np.ceil(tend / dt))
        dt = float(tend) / tsteps
        prdata["tsteps"] = int(tsteps)

    time_steps = cp.full((tsteps,), cp.float32(dt), dtype=cp.float32) if timedep else cp.full((1,), cp.float32(dt), dtype=cp.float32)

    nframes = (len(time_steps) + t_stride - 1) // t_stride
    if bool(prdata.get("store_slice_movies", True)):
        Ixz = cp.zeros((nframes, Nz, Nx), dtype=cp.float32)
        Iyz = cp.zeros((nframes, Nz, Ny), dtype=cp.float32)
        thetaxz = cp.zeros((nframes, Nz, Nx), dtype=cp.float32)
        thetayz = cp.zeros((nframes, Nz, Ny), dtype=cp.float32)

        def save_slices(slot, k, Ixy, theta_xy):
            Ixz[slot, k, :] = Ixy[:, Ny // 2]
            Iyz[slot, k, :] = Ixy[Nx // 2, :]
            dtheta = (theta_xy - theta_bias_2d).astype(cp.float32, copy=False)
            thetaxz[slot, k, :] = dtheta[:, Ny // 2]
            thetayz[slot, k, :] = dtheta[Nx // 2, :]
    else:
        Ixz = Iyz = thetaxz = thetayz = None
        save_slices = None

    ctx = LCContext(
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz, lm=lm,
        fxy2=fxy2, windowxy=windowxy,
        kout=kout, ne=ne, no=no, refin=refin, coh=coh,
        b=b, bi=bi, du=du, dv=dv,
        amp0=amp0, theta_bias_2d=theta_bias_2d, theta_full=theta_full,
        mobility=float(prdata.get("mobility", 1.0)),
        theta_prev_time=theta_prev_time,
        theta_bc=float(prdata.get("theta_bc", 0.0)),
        theta_z_gamma=float(prdata.get("theta_z_gamma", 0.0)),
        theta_z_stride_um=float(prdata.get("theta_z_stride_um", 0.0)),

        dtau_static=float(prdata.get("dtau_static", 0.01)),
        static_tol_resid=float(prdata.get("static_tol_resid", 5e-3)),
        static_resid_every=int(prdata.get("static_resid_every", 25)),
        static_max_steps=int(prdata.get("static_max_steps", 2000)),
        static_relax_omega=float(prdata.get("static_relax_omega", 0.3)),

        picard_iters=int(prdata.get("picard_iters", 4)),
        picard_tol_up=float(prdata.get("picard_tol_up", 1e-6)),

        td_theta_relax_steps=int(prdata.get("td_theta_relax_steps", 1)),
        td_theta_relax_omega=float(prdata.get("td_theta_relax_omega", 1.0)),

        save_full_I_mid_store=bool(prdata.get("save_full_I_mid_store", False)),
    )

    out = {
        "ctx": ctx,
        "time_steps": time_steps,
        "t_stride": t_stride,
        "timedep": timedep,
        "Ixz": Ixz,
        "Iyz": Iyz,
        "thetaxz": thetaxz,
        "thetayz": thetayz,
        "save_slices": save_slices,
        "x": x,
        "y": y,
        "z": z,
    }

    print("Initialized:")
    print("  Nx,Ny,Nz =", Nx, Ny, Nz)
    print("  dx,dy    =", float(dx), float(dy))
    print("  refin    =", float(refin))
    print("  b, bi    =", float(b), float(bi))
    print("  timedep  =", timedep, " tsteps =", int(len(time_steps)), " dt =", float(time_steps[0]))
    print("  store_slice_movies =", bool(prdata.get("store_slice_movies", True)))
    print("  save_full_I_mid_store =", bool(prdata.get("save_full_I_mid_store", False)))
    return out



# ================================================================================

# gaus

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



# ================================================================================

# build_amp_pair

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



# ================================================================================

# build_theta_bias_IC

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



# ================================================================================

# build_theta_bias_IC_dirichlet_value

def build_theta_bias_IC_dirichlet_value(
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



# ================================================================================

# compute_n_bg_from_bias

def compute_n_bg_from_bias(theta_bias, ne, no, xp_mod, reducer="median"):
    xp = xp_mod
    ct = xp.cos(theta_bias)
    st = xp.sin(theta_bias)
    inv_neff2 = (ct * ct) / (no * no) + (st * st) / (ne * ne)
    neff_bias = 1.0 / xp.sqrt(inv_neff2)
    return float(xp.mean(neff_bias)) if reducer == "mean" else float(xp.median(neff_bias))



# ================================================================================

# genrot

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



# ================================================================================

# intens

def intens(amp, coh, out=None):
    xp = cp.get_array_module(amp)
    a, b = amp[0], amp[1]
    if out is None:
        out = xp.empty(a.shape, a.real.dtype)
    if coh:
        s = a + b
        out[...] = (s.real * s.real + s.imag * s.imag)
    else:
        out[...] = (a.real * a.real + a.imag * a.imag + b.real * b.real + b.imag * b.imag)
    return out



# ================================================================================

# rebuild_ctx_from_repro

def rebuild_ctx_from_repro(repro):
    prdata = dict(repro["prdata"])
    c = repro["ctx"]

    prdata["Nx"] = int(c["Nx"])
    prdata["Ny"] = int(c["Ny"])
    prdata["Nz"] = int(c["Nz"])

    prdata["dx"] = float(c["dx"])
    prdata["dy"] = float(c["dy"])
    prdata["dz"] = float(c["dz"])

    prdata["xsamp"] = int(c["Nx"])
    prdata["ysamp"] = int(c["Ny"])
    prdata["zsteps"] = int(c["Nz"])

    out = initialize_lc_from_prdata(prdata)
    ctx = out["ctx"] if isinstance(out, dict) else out

    # Do NOT restore b/bi from repro.
    # They can be stale/wrong, especially after continuation/polish.
    skip = {
        "ctx_theta",
        "dual_grid",
        "b",
        "bi",
        "use_dual_grid",
    }

    for k, v in c.items():
        if k in skip:
            continue
        try:
            setattr(ctx, k, v)
        except Exception:
            pass

    ctx.use_dual_grid = False
    return ctx, prdata



# ================================================================================

# rebuild_ctx_from_run

def rebuild_ctx_from_run(run_dir, checkpoint_prefix="lc_eigensoliton"):
    run_dir = Path(run_dir)
    repro_path = run_dir / f"{checkpoint_prefix}_repro.json"

    if not repro_path.exists():
        raise FileNotFoundError(f"Missing repro file: {repro_path}")

    with open(repro_path, "r") as f:
        repro = json.load(f)

    ctx, prdata = rebuild_ctx_from_repro(repro)

    # FULL_polish stability should be full-grid
    ctx.use_dual_grid = False
    for attr in ["ctx_theta", "dual_grid"]:
        if hasattr(ctx, attr):
            delattr(ctx, attr)

    return ctx, prdata



# ================================================================================

# _npz_scalar

def _npz_scalar(z, key, default=None):
    if key not in _npz_keys(z):
        return default
    return float(cp.asnumpy(z[key]).ravel()[0])



# ================================================================================

# _npz_keys

def _npz_keys(z):
    return set(z.npz_file.files)



# ================================================================================

# _npz_string

def _npz_string(z, key, default="LEGACY"):
    keys = _npz_keys(z)
    if key not in keys:
        return default

    # String arrays should be read from the underlying NumPy npz file,
    # not through CuPy, because CuPy does not support unicode dtype <U.
    v = z.npz_file[key]

    try:
        return str(v.item())
    except Exception:
        return str(v)



# ================================================================================

# _npz_string_np

def _npz_string_np(profile_path, key, default="UNKNOWN"):
    # Use NumPy directly for strings; CuPy cannot load unicode dtype arrays.
    with np.load(profile_path, allow_pickle=True) as zn:
        if key not in zn.files:
            return default
        v = zn[key]
        try:
            return str(v.item())
        except Exception:
            return str(v)



# ================================================================================

# _lam_y_periodic_second_diff

def _lam_y_periodic_second_diff(Ny, dv, xp=cp):
    k = xp.arange(Ny, dtype=xp.float32)
    return (-4.0 * xp.sin(xp.pi * k / Ny)**2) / (dv * dv)



# ================================================================================

# _npz_float

def _npz_float(z, key, default=np.nan):
    if key not in _npz_keys(z):
        return default
    return float(cp.asnumpy(z[key]).ravel()[0])



# ================================================================================

# read_profile_summary

def read_profile_summary(profile_path):
    z = cp.load(profile_path, allow_pickle=True)
    try:
        out = {
            "path": Path(profile_path),
            "P_mW": _npz_scalar(z, "P_mW", np.nan),
            "bi": _npz_scalar(z, "bi", np.nan),
            "beta": _npz_scalar(z, "beta", np.nan),
            "grid_mode": _npz_string(z, "grid_mode", "LEGACY"),
            "branch": _npz_string(z, "branch", "LEGACY"),
        }
    finally:
        z.close()
    return out



# ================================================================================

# list_run_profiles

def list_run_profiles(run_dir, checkpoint_prefix="lc_eigensoliton"):
    run_dir = Path(run_dir)

    # normal existence-curve layout
    prof_dir = run_dir / f"{checkpoint_prefix}_profiles"
    files = sorted(prof_dir.glob(f"{checkpoint_prefix}_P_*mW.npz"))

    # polish legacy layout, if profiles were saved directly in folder
    if not files:
        files = sorted(run_dir.glob(f"{checkpoint_prefix}_P_*mW.npz"))

    return files



# ================================================================================

# choose_power_profile

def choose_power_profile(run_dir, checkpoint_prefix="lc_eigensoliton"):
    profiles = list_run_profiles(run_dir, checkpoint_prefix)
    if not profiles:
        raise FileNotFoundError(f"No profiles found in {run_dir}")

    rows = [read_profile_summary(p) for p in profiles]

    print("\nAvailable saved powers:")
    for i, r in enumerate(rows):
        print(
            f"  [{i}] P={r['P_mW']:.6g} mW, "
            f"beta={r['beta']:.8g}, grid={r['grid_mode']}"
        )

    i = int(input("Choose power index: "))
    return rows[i]["path"]



# ================================================================================

# load_saved_mode_profile

def load_saved_mode_profile(profile_path, ctx=None):
    z = cp.load(profile_path, allow_pickle=True)
    keys = _npz_keys(z)

    try:
        mode = {
            "path": Path(profile_path),
            "A": z["A"].astype(cp.complex64),
            "theta": z["theta"].astype(cp.float32),
            "I": z["I"].astype(cp.float32),
            "P_mW": _npz_scalar(z, "P_mW"),
            "bi": _npz_scalar(z, "bi"),
            "beta": _npz_scalar(z, "beta"),
        }

        # New format has theta_cont; legacy does not.
        mode["theta_cont"] = (
            z["theta_cont"].astype(cp.float32)
            if "theta_cont" in keys
            else mode["theta"].copy()
        )

        # Bias is essential; legacy fallback uses current ctx.
        if "theta_bias" in keys:
            mode["theta_bias"] = z["theta_bias"].astype(cp.float32)
        elif ctx is not None and hasattr(ctx, "theta_bias_2d"):
            mode["theta_bias"] = ctx.theta_bias_2d.copy()
        else:
            raise KeyError("Missing theta_bias and no ctx.theta_bias_2d fallback was provided.")

        mode["b"] = _npz_scalar(z, "b", float(getattr(ctx, "b", np.nan)))
        mode["theta_bc"] = _npz_scalar(
            z,
            "theta_bc",
            float(getattr(ctx, "theta_bc", 0.0)),
        )

        if "x_um" in keys and "y_um" in keys:
            mode["x_um"] = cp.asarray(z["x_um"])
            mode["y_um"] = cp.asarray(z["y_um"])
        elif ctx is not None:
            x_um, y_um = xy_um_from_ctx(ctx)
            mode["x_um"] = cp.asarray(x_um)
            mode["y_um"] = cp.asarray(y_um)
        else:
            mode["x_um"] = None
            mode["y_um"] = None

        mode["grid_mode"] = _npz_string(z, "grid_mode", "LEGACY")
        mode["branch"] = _npz_string(z, "branch", "LEGACY")

        if "theta_c" in keys:
            mode["theta_c"] = z["theta_c"].astype(cp.float32)
        if "theta_bias_c" in keys:
            mode["theta_bias_c"] = z["theta_bias_c"].astype(cp.float32)

    finally:
        z.close()

    return mode



# ================================================================================

# load_verified_profile

def load_verified_profile(profile_path):
    profile_path = Path(profile_path)

    z = cp.load(profile_path, allow_pickle=True)
    try:
        mode = dict(
            path=profile_path,
            A=z["A"].astype(cp.complex64),
            theta=z["theta"].astype(cp.float32),
            I=z["I"].astype(cp.float32),
            beta=_npz_float(z, "beta"),
            P_mW=_npz_float(z, "P_mW"),
            b=_npz_float(z, "b"),
            bi=_npz_float(z, "bi"),
            theta_bc=_npz_float(z, "theta_bc"),
            beta_verify_diff=_npz_float(z, "beta_verify_diff"),
            beta_verify_rayleigh=_npz_float(z, "beta_verify_rayleigh"),
            beta_verify_tol=_npz_float(z, "beta_verify_tol"),
        )
    finally:
        z.close()

    mode["grid_mode"] = _npz_string_np(profile_path, "grid_mode", "UNKNOWN")
    mode["branch"] = _npz_string_np(profile_path, "branch", "UNKNOWN")

    return mode



# ================================================================================

# assert_ctx_matches_profile

def assert_ctx_matches_profile(ctx, profile):
    shape = tuple(profile["A"].shape)

    if (ctx.Nx, ctx.Ny) != shape:
        raise ValueError(
            f"ctx grid {(ctx.Nx, ctx.Ny)} "
            f"does not match saved profile {shape}"
        )

    if tuple(ctx.fxy2.shape) != shape:
        raise ValueError(
            f"ctx.fxy2.shape={ctx.fxy2.shape} "
            f"does not match profile {shape}"
        )

    if tuple(ctx.theta_bias_2d.shape) != shape:
        raise ValueError(
            f"ctx.theta_bias_2d.shape={ctx.theta_bias_2d.shape} "
            f"does not match profile {shape}"
        )



# ================================================================================

# launch_stability_vs_z

def launch_stability_vs_z(
    ctx,
    profile_path,
    *,
    perturbations=None,
    runner_kwargs=None,
    save_dir=None,
    reuse_reference=False,
    reference_I_mid_path=None,
):
    mode = load_verified_profile(profile_path)
    A_ref, theta_ref = install_launch_profile_into_ctx(ctx, mode)

    if perturbations is None:
        perturbations = [
            dict(label="unperturbed", eps=0.0, seed=1),
            dict(label="noise_1e-6", eps=1e-6, seed=1),
            dict(label="noise_3e-6", eps=3e-6, seed=1),
            dict(label="noise_1e-5", eps=1e-5, seed=1),
        ]

    save_dir = Path(save_dir or (Path(profile_path).parent / "launch_stability"))
    save_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    rows = []

    if reuse_reference:
        if reference_I_mid_path is None:
            raise ValueError("reuse_reference=True requires reference_I_mid_path")

        zref = cp.load(reference_I_mid_path)
        runs["unperturbed"] = zref["I_mid"]

        print("[launch] loaded unperturbed reference:")
        print("   ", reference_I_mid_path)
        print("   I_mid shape:", runs["unperturbed"].shape)

    for p in perturbations:
        label = p["label"]

        if reuse_reference and label == "unperturbed":
            print("[launch] skipping unperturbed; using loaded reference")
            continue

        eps = float(p.get("eps", 0.0))

        print(f"\n[launch] {label}, eps={eps:g}")

        # Reset theta stack identically for each launch
        A_ref, theta_ref = install_launch_profile_into_ctx(ctx, mode)
        A_launch = launch_perturb_A(A_ref, ctx, **p)

        I_mid = run_launch_zmarch(
            ctx,
            A_launch,
            runner_kwargs=runner_kwargs,
        )

        runs[label] = I_mid

        # per-z scalar diagnostics
        for k in range(ctx.Nz):
            I = cp.asarray(I_mid[k])
            x0, y0, sx, sy, Imax = transverse_metrics_from_I(I, ctx)

            rows.append(dict(
                label=label,
                eps=eps,
                k=k,
                z_um=float(k * ctx.dz),
                x0_um=x0,
                y0_um=y0,
                sx_um=sx,
                sy_um=sy,
                Imax=Imax,
            ))

        cp.savez(
            save_dir / f"{Path(profile_path).stem}__launch__{label}.npz",
            I_mid=I_mid,
            P_mW=cp.array(mode["P_mW"]),
            beta=cp.array(mode["beta"]),
            eps=cp.array(eps),
        )

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    # paired separation vs unperturbed
    I0 = runs["unperturbed"]
    pair_rows = []

    for p in perturbations:
        label = p["label"]
        if label == "unperturbed":
            continue

        eps = float(p.get("eps", 0.0))
        Ip = runs[label]

        for k in range(ctx.Nz):
            dI = rel_l2_cp(I0[k], Ip[k])

            extra = launch_shape_metrics_from_I_pair(Ip[k], I0[k], ctx)

            row = dict(
                label=label,
                eps=eps,
                k=k,
                z_um=float(k * ctx.dz),
                paired_rel_I=dI,
                paired_gain_I=dI / max(eps, 1e-30),
            )
            row.update(extra)
            pair_rows.append(row)

    df = pd.DataFrame(rows)
    df_pair = pd.DataFrame(pair_rows)

    df.to_csv(save_dir / f"{Path(profile_path).stem}__launch_metrics.csv", index=False)
    df_pair.to_csv(save_dir / f"{Path(profile_path).stem}__launch_paired.csv", index=False)

    return df, df_pair



# ================================================================================

# install_launch_profile_into_ctx

def install_launch_profile_into_ctx(ctx, mode):
    """
    Prepare ctx for launching this saved eigensoliton.
    Full-grid launch only.
    """
    ctx.use_dual_grid = False

    for attr in ["ctx_theta", "dual_grid"]:
        if hasattr(ctx, attr):
            delattr(ctx, attr)

    ctx.b = float(mode["b"])
    ctx.bi = float(mode["bi"])
    ctx.theta_bc = float(mode["theta_bc"])

    theta = mode["theta"].astype(cp.float32, copy=True)
    theta[0, :] = cp.float32(ctx.theta_bc)
    theta[-1, :] = cp.float32(ctx.theta_bc)
    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    # z-stack initialized to the eigensoliton director everywhere
    ctx.theta_full = cp.repeat(theta[None, :, :], ctx.Nz, axis=0)

    if hasattr(ctx, "theta_prev_time"):
        ctx.theta_prev_time = ctx.theta_full.copy()

    return mode["A"].astype(cp.complex64, copy=True), theta



# ================================================================================

# enforce_theta_constraints

def enforce_theta_constraints(theta, theta_clamp):
    """
    Enforce optional clamp and Dirichlet-x boundary conditions.

    Parameters
    ----------
    theta : cp.ndarray
        Array of shape (Nx, Ny).
    theta_clamp : tuple[float, float] or None
        Optional (theta_min, theta_max).

    Returns
    -------
    cp.ndarray
        Constrained theta field.
    """
    def enforce_theta_constraints(theta, theta_clamp):
        if theta_clamp is not None:
            theta = cp.clip(
                theta,
                cp.float32(theta_clamp[0]),
                cp.float32(theta_clamp[1]),
            )

        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta[0, :] = theta_bc
        theta[-1, :] = theta_bc
    return theta



# ================================================================================

# launch_perturb_A

def launch_perturb_A(A, ctx, *, label="noise", eps=1e-5, seed=1):
    """
    Small perturbation to launch field, normalized back to same total power shape.
    """
    if eps == 0:
        return A.copy()

    rng = cp.random.RandomState(seed)

    if label.startswith("noise"):
        eta = rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)
        eta = eta.astype(cp.complex64)
        eta /= cp.sqrt(cp.mean(cp.abs(eta)**2))
        Ap = A + eps * cp.sqrt(cp.mean(cp.abs(A)**2)) * eta

    elif label.startswith("phase"):
        phi = rng.standard_normal(A.shape).astype(cp.float32)
        phi /= cp.std(phi)
        Ap = A * cp.exp(1j * eps * phi)

    else:
        raise ValueError(f"Unknown launch perturbation label={label}")

    # preserve ∫|A|² dxdy
    p0 = cp.sum(cp.abs(A)**2) * ctx.dx * ctx.dy
    p1 = cp.sum(cp.abs(Ap)**2) * ctx.dx * ctx.dy
    Ap *= cp.sqrt(p0 / p1)

    return Ap.astype(cp.complex64, copy=False)



# ================================================================================

# rel_l2_cp

def rel_l2_cp(a, b):
    return float(cp.sqrt(cp.sum(cp.abs(a - b)**2) / (cp.sum(cp.abs(a)**2) + 1e-30)))



# ================================================================================

# run_launch_zmarch

def run_launch_zmarch(ctx, A_launch, *, runner_kwargs=None):
    runner_kwargs = {} if runner_kwargs is None else dict(runner_kwargs)

    old = {}
    for name in ["amp0", "amp", "amp_time_state"]:
        if hasattr(ctx, name):
            old[name] = getattr(ctx, name)

    old_save_full = getattr(ctx, "save_full_I_mid_store", None)

    try:
        Aset2 = A_launch.astype(cp.complex64, copy=True)

        print("Aset2 shape, dim ", Aset2.shape, Aset2.ndim)

        if Aset2.ndim == 2:
            Aset = cp.zeros((2, int(ctx.Nx), int(ctx.Ny)), dtype=cp.complex64)
            Aset[0, :, :] = Aset2
        elif Aset2.ndim == 3 and Aset2.shape[0] == 1:
            Aset = cp.zeros((2, int(ctx.Nx), int(ctx.Ny)), dtype=cp.complex64)
            Aset[0, :, :] = Aset2[0]
        else:
            Aset = Aset2

        print("Aset shape, dim ", Aset.shape, Aset.ndim)

        ctx.amp0 = Aset.copy()
        ctx.amp = Aset.copy()
        ctx.amp_time_state = Aset.copy()
        ctx.save_full_I_mid_store = True

        print("launch power amp0:", float(cp.sum(cp.abs(ctx.amp0)**2) * ctx.dx * ctx.dy))
        print("launch power amp :", float(cp.sum(cp.abs(ctx.amp)**2) * ctx.dx * ctx.dy))
        print("launch power ats :", float(cp.sum(cp.abs(ctx.amp_time_state)**2) * ctx.dx * ctx.dy))

        I_mid_store = run_unified_td_static(
            ctx,
            timedep=False,
            true_td=False,
            restart_theta=False,
            restart_static_from_bias=False,
            static_mode=runner_kwargs.pop("static_mode", "strict_relax"),
            store_I_dtype=runner_kwargs.pop("store_I_dtype", cp.float32),
            tqdm_static_z=runner_kwargs.pop("tqdm_static_z", True),
            report_runtime=runner_kwargs.pop("report_runtime", True),
            strict_residual_tol_max=runner_kwargs.pop("strict_residual_tol_max", 5e-2),
            strict_residual_tol_rms=runner_kwargs.pop("strict_residual_tol_rms", 1e-2),
            strict_max_outer_passes=runner_kwargs.pop("strict_max_outer_passes", 8),
            strict_verbose_every=runner_kwargs.pop("strict_verbose_every", 100),
            **runner_kwargs,
        )

        if isinstance(I_mid_store, dict):
            I_mid_store = I_mid_store["I_mid_store"]
        elif isinstance(I_mid_store, tuple):
            I_mid_store = I_mid_store[0]

        I_mid_store = cp.asarray(I_mid_store)

        print(
            "[launch returned]",
            I_mid_store.shape,
            I_mid_store.dtype,
            "Imax=", float(cp.max(I_mid_store)),
            "Isum0=", float(cp.sum(I_mid_store[0]) * ctx.dx * ctx.dy),
        )

        return I_mid_store

    finally:
        for name, val in old.items():
            setattr(ctx, name, val)
        if old_save_full is not None:
            ctx.save_full_I_mid_store = old_save_full



# ================================================================================

# transverse_metrics_from_I



def launch_shape_metrics_from_I_pair(I, I0, ctx):
    """
    Extra launch-stability diagnostics comparing perturbed intensity I
    to reference intensity I0 at the same z slice.
    """
    import cupy as cp

    x_um, y_um = xy_um_from_ctx(ctx)
    x = cp.asarray(x_um, dtype=cp.float32)[:, None]
    y = cp.asarray(y_um, dtype=cp.float32)[None, :]

    I  = cp.asarray(I, dtype=cp.float32)
    I0 = cp.asarray(I0, dtype=cp.float32)

    P  = cp.sum(I)  + cp.float32(1e-30)
    P0 = cp.sum(I0) + cp.float32(1e-30)

    xc  = cp.sum(x * I)  / P
    yc  = cp.sum(y * I)  / P
    xc0 = cp.sum(x * I0) / P0
    yc0 = cp.sum(y * I0) / P0

    sx  = cp.sqrt(cp.sum((x - xc)**2  * I)  / P)
    sy  = cp.sqrt(cp.sum((y - yc)**2  * I)  / P)
    sx0 = cp.sqrt(cp.sum((x - xc0)**2 * I0) / P0)
    sy0 = cp.sqrt(cp.sum((y - yc0)**2 * I0) / P0)

    dI = I - I0
    rel_I = cp.linalg.norm(dI.ravel()) / (cp.linalg.norm(I0.ravel()) + cp.float32(1e-30))

    Imax  = cp.max(I)
    Imax0 = cp.max(I0)

    # Odd parts of the *difference* dI = I - I0.
    # These measure symmetry-breaking perturbation content, not baseline asymmetry.
    dI_flip_x = dI[::-1, :]
    dI_flip_y = dI[:, ::-1]

    odd_x_dI = cp.float32(0.5) * (dI - dI_flip_x)
    odd_y_dI = cp.float32(0.5) * (dI - dI_flip_y)

    odd_x_frac = cp.linalg.norm(odd_x_dI.ravel()) / (cp.linalg.norm(I0.ravel()) + cp.float32(1e-30))
    odd_y_frac = cp.linalg.norm(odd_y_dI.ravel()) / (cp.linalg.norm(I0.ravel()) + cp.float32(1e-30))

    # Breathing-like scalar: change in second moment radius.
    r2  = sx**2  + sy**2
    r20 = sx0**2 + sy0**2

    return dict(
        rel_I_extra=float(rel_I.get()),
        dx_centroid_um=float((xc - xc0).get()),
        dy_centroid_um=float((yc - yc0).get()),
        dsx_um=float((sx - sx0).get()),
        dsy_um=float((sy - sy0).get()),
        sx_um=float(sx.get()),
        sy_um=float(sy.get()),
        Imax_ratio=float((Imax / (Imax0 + cp.float32(1e-30))).get()),
        odd_x_frac=float(odd_x_frac.get()),
        odd_y_frac=float(odd_y_frac.get()),
        dr2_um2=float((r2 - r20).get()),
    )



def transverse_metrics_from_I(I, ctx):
    x_um, y_um = xy_um_from_ctx(ctx)
    X = cp.asarray(x_um)[:, None]
    Y = cp.asarray(y_um)[None, :]

    W = cp.sum(I) + 1e-30
    x0 = cp.sum(I * X) / W
    y0 = cp.sum(I * Y) / W
    sx = cp.sqrt(cp.sum(I * (X - x0)**2) / W)
    sy = cp.sqrt(cp.sum(I * (Y - y0)**2) / W)

    return float(x0), float(y0), float(sx), float(sy), float(cp.max(I))



# ================================================================================

# xy_um_from_ctx

def xy_um_from_ctx(ctx):
    """
    Physical transverse coordinates in microns.
    Assumes ctx.dx, ctx.dy are in microns.
    """
    x_um = (cp.arange(ctx.Nx, dtype=cp.float32) - cp.float32((ctx.Nx - 1) / 2)) * cp.float32(ctx.dx)
    y_um = (cp.arange(ctx.Ny, dtype=cp.float32) - cp.float32((ctx.Ny - 1) / 2)) * cp.float32(ctx.dy)
    return x_um, y_um



# ================================================================================

# _slice_residual_rms_local3d_z

def _slice_residual_rms_local3d_z(theta_k, I_k, theta_prev, theta_next, ctx, gamma_z):
    th = theta_k.astype(cp.float64, copy=False)
    I64 = I_k.astype(cp.float64, copy=False)
    tp = theta_prev.astype(cp.float64, copy=False)
    tn = theta_next.astype(cp.float64, copy=False)

    theta_bc = cp.float64(getattr(ctx, "theta_bc", 0.0))

    du = cp.float64(ctx.du)
    dv = cp.float64(ctx.dv)
    dz = cp.float64(ctx.dz)

    du2 = du * du
    dv2 = dv * dv
    dz2 = dz * dz

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    th_xplus = cp.empty_like(th)
    th_xminus = cp.empty_like(th)
    th_xplus[:-1, :] = th[1:, :]
    th_xplus[-1, :] = theta_bc
    th_xminus[1:, :] = th[:-1, :]
    th_xminus[0, :] = theta_bc

    lap_xy = (th_xplus - 2.0 * th + th_xminus) / du2 + (th_yplus - 2.0 * th + th_yminus) / dv2
    lap_xy[0, :] = 0.0
    lap_xy[-1, :] = 0.0

    lap_z = cp.float64(gamma_z) * (tn - 2.0 * th + tp) / dz2
    lap_z[0, :] = 0.0
    lap_z[-1, :] = 0.0

    drive = (cp.float64(ctx.b) + cp.float64(ctx.bi) * I64) * cp.sin(2.0 * th)
    R = (lap_xy + lap_z + drive) / cp.float64(ctx.mobility)
    R[0, :] = 0.0
    R[-1, :] = 0.0
    return float(cp.sqrt(cp.mean(R * R)))



# ================================================================================

# apply_nonlinear_phase_inplace

def apply_nonlinear_phase_inplace(amp, theta_xy, *, dz_step_um=None, dz_step=None, kout, ne, no, refin):
    if dz_step_um is None:
        dz_step_um = dz_step
    dn = lc_dn_from_theta(theta_xy, ne, no, refin=refin)
    phase = cp.exp((1j * cp.float32(kout * dz_step_um)) * dn).astype(cp.complex64, copy=False)
    amp *= phase
    return amp



# ================================================================================

# choose_optics_substeps

def choose_optics_substeps(dz_um, *, kout, dn_max_est, dz_opt_max_phi, max_substeps):
    phi_est = abs(float(kout)) * abs(float(dz_um)) * abs(float(dn_max_est))
    nsub = int(np.ceil(phi_est / float(dz_opt_max_phi))) if phi_est > 0 else 1
    nsub = max(1, min(int(max_substeps), nsub))
    dz_sub = float(dz_um) / nsub
    return nsub, dz_sub, phi_est



# ================================================================================

# get_h_for_dz

def get_h_for_dz(ctx, dz_step_um):
    key = float(np.round(float(dz_step_um), 12))
    if key in ctx._h_cache:
        return ctx._h_cache[key]

    lm = ctx.lm
    refin = ctx.refin
    fxy2 = ctx.fxy2

    h = cp.where(
        (lm / refin)**2 * fxy2 < 1.0,
        cp.exp(
            2.0j * cp.pi * refin * cp.float32(dz_step_um) / lm
            * cp.sqrt(1.0 - (lm / refin)**2 * fxy2)
        ),
        0.0
    ).astype(cp.complex64, copy=False)

    ctx._h_cache[key] = h
    return h



# ================================================================================

# hop_linear_inplace

def hop_linear_inplace(amp, hker, windowxy, Ahat, *, plan_f, plan_i):
    with plan_f:
        Ahat[...] = spfft.fft2(amp, axes=(-2, -1))
    Ahat *= hker
    with plan_i:
        amp[...] = spfft.ifft2(Ahat, axes=(-2, -1))
    if windowxy is not None:
        amp *= windowxy
    return amp



# ================================================================================

# intens_into

def intens_into(out_I32, amp, *, coh: bool):
    if coh:
        s = amp[0] + amp[1]
        out_I32[...] = (s.real * s.real + s.imag * s.imag).astype(cp.float32, copy=False)
    else:
        a0 = amp[0]
        a1 = amp[1]
        out_I32[...] = (
            a0.real * a0.real + a0.imag * a0.imag +
            a1.real * a1.real + a1.imag * a1.imag
        ).astype(cp.float32, copy=False)
    return out_I32



# ================================================================================

# prepare_cn_ky_operator

def prepare_cn_ky_operator(*, dt, mobility, du, dv, Ny):
    mob = float(mobility)
    s = float(dt) / (2.0 * mob)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-s) * (1.0 / (du * du)))
    diag = (1.0 + (2.0 * s) * (1.0 / (du * du)) - s * lam_y).astype(cp.float32, copy=False)
    return cp.float32(s), off, diag, lam_y



# ================================================================================

# prepare_ie_ky_operator

def prepare_ie_ky_operator(*, dt, mobility, du, dv, Ny):
    a_ie = float(dt) / float(mobility)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-a_ie) * (1.0 / (du * du)))
    diag = (1.0 + (2.0 * a_ie) * (1.0 / (du * du)) - a_ie * lam_y).astype(cp.float32, copy=False)
    return cp.float32(a_ie), off, diag, lam_y



# ================================================================================

# run_quasiglobal_td_branch

def run_quasiglobal_td_branch(
    ctx,
    *,
    dt_j,
    store_I_dtype,
    I_mid_store,
    save_slices_fn,
    t_stride,
    jt,
    jt_iter,
    report_runtime,
    tmp_dir="/tmp",
    use_memmap_pred=False,
    z_chunk=16,
):
    """
    Run one quasi-global TD step and update ctx.theta_full in place.

    Returns
    -------
    td_resid_last : float
    td_resid_max  : float
    """
    theta_new, I_mid_store_new, amp_out = td_step_quasiglobal_fast(
        ctx,
        ctx.amp_time_state,
        dt_j,
        I_store_dtype=store_I_dtype,
        tmp_dir=tmp_dir,
        use_memmap_pred=use_memmap_pred,
        z_chunk=z_chunk,
    )

    ctx.theta_full[...] = theta_new
    ctx.amp_time_state[...] = amp_out

    if I_mid_store is not None:
        I_mid_store[...] = I_mid_store_new.astype(store_I_dtype, copy=False)

    if save_slices_fn is not None and ((jt % t_stride) == 0):
        slot = jt // t_stride
        for k in range(ctx.Nz):
            save_slices_fn(slot, k, I_mid_store_new[k], ctx.theta_full[k])

    td_resid_last = 0.0
    td_resid_max = 0.0

    if report_runtime:
        gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

        for k in range(ctx.Nz):
            tp = ctx.theta_full[k - 1] if k > 0 else ctx.theta_full[k]
            tn = ctx.theta_full[k + 1] if (k + 1) < ctx.Nz else ctx.theta_full[k]

            rloc = _slice_residual_rms_local3d_z(
                ctx.theta_full[k],
                I_mid_store_new[k].astype(cp.float32, copy=False),
                tp,
                tn,
                ctx,
                gamma_z,
            )
            td_resid_last = rloc
            td_resid_max = max(td_resid_max, rloc)

        try:
            jt_iter.set_postfix(
                rlast=f"{td_resid_last:.2e}",
                rmax=f"{td_resid_max:.2e}",
            )
        except Exception:
            pass

    del theta_new, I_mid_store_new, amp_out


    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return td_resid_last, td_resid_max



# ================================================================================

# solve_theta_perturbation_dirichletx_periody

def solve_theta_perturbation_dirichletx_periody(
    theta_bias, Ixy, *, b, bi, du, dv,
    xp=None, return_theta=True, check_residual=True, verbose=False
):
    if xp is None:
        xp = cp.get_array_module(Ixy)

    Ixy = xp.asarray(Ixy, dtype=xp.float64)
    theta_b_1d = _extract_theta_b_1d(theta_bias, xp).astype(xp.float64, copy=False)

    Nx, Ny = Ixy.shape
    qx = 2.0 * b * xp.cos(2.0 * theta_b_1d)
    rhs = -bi * Ixy * xp.sin(2.0 * theta_b_1d)[:, None]

    qx_i = qx[1:-1]
    rhs_i = rhs[1:-1, :]

    rhs_hat_i = xp.fft.fft(rhs_i, axis=1)
    k = xp.arange(Ny, dtype=xp.float64)
    lam_y = (2.0 * xp.cos(2.0 * xp.pi * k / Ny) - 2.0) / (dv * dv)

    off = xp.full((Nx - 3,), 1.0 / (du * du), dtype=xp.float64)
    diag = (-2.0 / (du * du)) + qx_i[:, None] + lam_y[None, :]

    delta_hat_i = thomas_batched_modevarying(off, diag, off, rhs_hat_i, xp)
    delta_i = xp.fft.ifft(delta_hat_i, axis=1).real

    delta = xp.zeros((Nx, Ny), dtype=xp.float64)
    delta[1:-1, :] = delta_i

    out = {"delta": delta, "rhs": rhs, "lam_y": lam_y, "qx": qx}

    if return_theta:
        theta2d = theta_b_1d[:, None] + delta
        theta2d[0, :] = theta_b_1d[0]
        theta2d[-1, :] = theta_b_1d[-1]
        out["theta"] = theta2d

    if check_residual:
        lin_res = apply_Lpert_delta(delta, theta_b_1d, b=b, du=du, dv=dv, xp=xp) - rhs
        lin_res[0, :] = 0.0
        lin_res[-1, :] = 0.0
        out["linear_residual"] = lin_res

    return out



# ================================================================================

# strict_static_relax_slice_selfconsistent

def strict_static_relax_slice_selfconsistent(
    amp_in,
    theta_seed,
    tp,
    tn,
    ctx,
    *,
    dz_sub,
    Nsub,
    h_sub,
    Ahat,
    plan_f,
    plan_i,
    sS,
    offS,
    diagS,
    lamS,
    use_linear_seed=True,
    early_accept_linear_seed=True,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    max_selfcons_passes=3,
    selfcons_tol_theta=1e-4,
    selfcons_tol_I=1e-4,
    verbose=False,
):
    """
    Self-consistent strict static slice solve for gamma_z = 0 case.

    Returns
    -------
    theta_out : cp.ndarray
    I_mid_out : cp.ndarray
    amp_out   : cp.ndarray
    info      : dict
    """
    # incoming intensity
    I_b = cp.empty((ctx.Nx, ctx.Ny), dtype=cp.float32)
    intens_into(I_b, amp_in, coh=ctx.coh)

    # work buffers
    amp_work = cp.empty_like(amp_in)
    I_a = cp.empty_like(I_b)
    I_mid = cp.empty_like(I_b)

    # initial theta guess
    theta = theta_seed.astype(cp.float32, copy=False)
    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    last_I_mid = None
    last_info = None
    converged = False

    for sc_pass in range(max_selfcons_passes):
        # optics with current theta
        amp_work[...] = amp_in
        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp_work, theta,
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(
                amp_work, h_sub, ctx.windowxy, Ahat,
                plan_f=plan_f, plan_i=plan_i
            )

        intens_into(I_a, amp_work, coh=ctx.coh)
        fill_mid_intensity(I_mid, I_b, I_a)

        # strict theta solve at this midpoint forcing
        theta_new, info = strict_static_relax_slice(
            theta,
            I_mid,
            tp,
            tn,
            ctx,
            sS=sS,
            offS=offS,
            diagS=diagS,
            lamS=lamS,
            use_linear_seed=(use_linear_seed and sc_pass == 0),
            early_accept_linear_seed=(early_accept_linear_seed and sc_pass == 0),
            residual_tol_max=residual_tol_max,
            residual_tol_rms=residual_tol_rms,
            max_outer_passes=max_outer_passes,
            verbose=verbose,
        )

        theta = enforce_theta_constraints(theta, ctx.theta_clamp)

        dtheta = float(cp.max(cp.abs(theta_new - theta)))

        if last_I_mid is None:
            dI = cp.inf
        else:
            denom = max(float(cp.max(cp.abs(last_I_mid))), 1e-12)
            dI = float(cp.max(cp.abs(I_mid - last_I_mid)) / denom)

        if verbose:
            print(
                f"[strict selfcons] pass={sc_pass+1:2d} "
                f"dtheta={dtheta:.3e} dI={dI:.3e} "
                f"max={info['max_interior']:.3e} rms={info['rms_interior']:.3e}"
            )

        theta = theta_new
        last_info = info

        if last_I_mid is None:
            last_I_mid = I_mid.copy()
        else:
            last_I_mid[...] = I_mid

        if (
            dtheta <= selfcons_tol_theta
            and dI <= selfcons_tol_I
            and info["max_interior"] <= residual_tol_max
            and info["rms_interior"] <= residual_tol_rms
        ):
            converged = True
            break

    return theta, I_mid, amp_work, {
        "converged": converged,
        "n_selfcons_passes": sc_pass + 1,
        "max_interior": last_info["max_interior"],
        "rms_interior": last_info["rms_interior"],
    }



# ================================================================================

# td_get_slice_mid_intensity

def td_get_slice_mid_intensity(
    ctx,
    amp,
    theta_use,
    I_b,
    I_a,
    I_mid,
    *,
    Nsub,
    dz_sub,
    h_sub,
    Ahat,
    plan_f,
    plan_i,
    frozen_I_mid_k=None,
):
    """
    Fill I_mid for one TD slice.

    If frozen_I_mid_k is provided, use it directly and leave amp unchanged.
    Otherwise march optics through the slice using theta_use.
    Returns
    -------
    amp_after : cp.ndarray
        The optical field after this slice (same object as amp if marched in place).
    """
    if frozen_I_mid_k is not None:
        I_mid[...] = frozen_I_mid_k
        return amp

    for _ in range(Nsub):
        apply_nonlinear_phase_inplace(
            amp, theta_use,
            dz_step_um=dz_sub,
            kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
        )
        hop_linear_inplace(
            amp, h_sub, ctx.windowxy, Ahat,
            plan_f=plan_f, plan_i=plan_i
        )

    intens_into(I_a, amp, coh=ctx.coh)
    fill_mid_intensity(I_mid, I_b, I_a)
    return amp



# ================================================================================

# td_update_theta_from_midintensity

def td_update_theta_from_midintensity(
    ctx,
    theta_ref_k,
    tp,
    tn,
    I_mid,
    *,
    dt_j,
    true_td,
    a_ie,
    s,
    off,
    diag,
    lam_y,
    Nsub,
    dz_sub,
    h_sub,
    amp_for_pred=None,
    I_b=None,
    I_a=None,
    Ahat=None,
    plan_f=None,
    plan_i=None,
):
    """
    Update theta for one TD slice given I_mid.

    Returns
    -------
    theta_new : cp.ndarray
    """
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    predictor_only = bool(getattr(ctx, "true_td_predictor_only", False))
    do_full_pred = bool(getattr(ctx, "true_td_full_pred_optics", False))

    if not true_td:
        return relax_theta_to_current_I_td_zcoupled(
            theta_ref_k,
            I_mid,
            tp,
            tn,
            ctx,
            dt=dt_j,
            s=s,
            off=off,
            diag=diag,
            lam_y=lam_y,
        )

    # true physical-time branch
    if gamma_z != 0.0:
        theta_new = advance_theta_physical_timestep_semiimplicit_zcoupled(
            theta_ref_k,
            dt=dt_j,
            b=ctx.b,
            bi=ctx.bi,
            Ixy=I_mid,
            theta_prev_n=tp,
            theta_next_n=tn,
            gamma_z=gamma_z,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            a_ie=a_ie,
            off=off,
            diag=diag,
            lam_y=lam_y,
            inv_dz2=_get_inv_dz2(ctx),
            clamp=ctx.theta_clamp,
        )
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta_new[0, :] = theta_bc
        theta_new[-1, :] = theta_bc
        return theta_new

    # gamma_z == 0: predictor/corrector route
    theta_pred_k = advance_theta_physical_timestep_semiimplicit_zcoupled(
        theta_ref_k,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        Ixy=I_mid,
        theta_prev_n=tp,
        theta_next_n=tn,
        gamma_z=0.0,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        a_ie=a_ie,
        off=off,
        diag=diag,
        lam_y=lam_y,
        inv_dz2=_get_inv_dz2(ctx),
        clamp=ctx.theta_clamp,
    )

    if predictor_only:
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta_pred_k[0, :] = theta_bc
        theta_pred_k[-1, :] = theta_bc
        return theta_pred_k

    if do_full_pred:
        amp_pred_buf = ctx._amp_pred_buf
        amp_pred_buf[...] = amp_for_pred

        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp_pred_buf, theta_pred_k,
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(
                amp_pred_buf, h_sub, ctx.windowxy, Ahat,
                plan_f=plan_f, plan_i=plan_i
            )

        I_mid_pred = ctx._I_mid_pred_buf
        intens_into(I_a, amp_pred_buf, coh=ctx.coh)
        fill_mid_intensity(I_mid_pred, I_b, I_a)
        I_pic_use = I_mid_pred
    else:
        I_pic_use = I_mid

    s_cn, off_cn, diag_cn, lam_y_cn = _get_cn_ky_operator_cached(ctx, dt_j)


    theta_new = advance_theta_timestep_cn_trap_picard_prepared(
        theta_ref_k,
        dt=dt_j,
        b=ctx.b,
        bi=ctx.bi,
        I_n=I_mid,
        I_pic=I_pic_use,
        mobility=ctx.mobility,
        du=ctx.du,
        dv=ctx.dv,
        s=s_cn,
        off=off_cn,
        diag=diag_cn,
        lam_y=lam_y_cn,
        max_iter=getattr(ctx, "picard_iters", 4),
        tol_update=getattr(ctx, "picard_tol_up", 1e-6),
        clamp=ctx.theta_clamp,
    )

    theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
    theta_new[0, :] = theta_bc
    theta_new[-1, :] = theta_bc

    return theta_new



# ================================================================================

# run_unified_td_static

def run_unified_td_static(
    ctx, *,
    timedep,
    time_steps=None,
    tsteps=None,
    t_stride=1,
    restart_theta=True,
    restart_static_from_bias=True,
    store_I_dtype=cp.float16,
    save_slices_fn=None,
    tqdm_timedep=True,
    tqdm_static_z=True,
    report_runtime=True,
    true_td=True,
    static_mode="strict_relax",      # "linear", "strict_relax"
    use_linear_seed_for_static_relax=True,
    early_accept_linear_seed=True,
    strict_residual_tol_max=5e-2,
    strict_residual_tol_rms=1e-2,
    strict_max_outer_passes=8,
    strict_verbose_every=100,
    frozen_I_mid_store=None,
    frozen_I_from_theta=False,
    use_quasiglobal_td=False,
    freeze_theta=False,
    checkpoint_dir=None,
    checkpoint_every=0,
    checkpoint_keep_history=False,
    checkpoint_prefix="lc",
    checkpoint_Ixz=None,
    checkpoint_Iyz=None,
    checkpoint_thetaxz=None,
    checkpoint_thetayz=None,
    checkpoint_theta_history_dir=None,
    checkpoint_theta_history_prefix="theta_hist",
    checkpoint_theta_history_dtype=np.float32,
):
    if static_mode not in ("linear", "strict_relax"):
        raise ValueError("static_mode must be 'linear', or 'strict_relax'")

    if timedep:
        if time_steps is None:
            raise ValueError("timedep=True requires time_steps")
        if tsteps is None:
            tsteps = len(time_steps)
    else:
        tsteps = 1
    t_model = 0.0
    if timedep and restart_theta:
        if use_dual_grid:
            ctx_t.theta_full[...] = ctx_t.theta_bias_2d[None, :, :]
    else:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]
    if (not timedep) and restart_static_from_bias:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    if (ctx.theta_z_stride_um is None) or (float(ctx.theta_z_stride_um) <= 0.0):
        theta_k_stride = 1
    else:
        theta_k_stride = max(1, int(round(float(ctx.theta_z_stride_um) / float(ctx.dz))))

    use_dual_grid = bool(getattr(ctx, "use_dual_grid", False))

    if use_dual_grid:
        ctx_t = ctx.ctx_theta
        dg = ctx.dual_grid
        print(
            f"[dual grid] optics=({ctx.Nx},{ctx.Ny}) "
            f"theta=({ctx_t.Nx},{ctx_t.Ny}) "
            f"rx,ry=({dg.rx},{dg.ry})"
        )
    else:
        ctx_t = ctx
        dg = None


    Nsub, dz_sub, phi_est = choose_optics_substeps(
        ctx.dz,
        kout=ctx.kout,
        dn_max_est=ctx.dn_max_est,
        dz_opt_max_phi=ctx.dz_opt_max_phi,
        max_substeps=ctx.max_substeps
    )
    h_sub = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)
    ctx.Nsub = Nsub
    ctx.dz_sub = dz_sub
    quasi_report = True
    print(f"[theta stride] theta_z_stride_um={ctx.theta_z_stride_um} -> theta_k_stride={theta_k_stride} slices")
    print(f"[optics substep] dz={ctx.dz} um phi_est={phi_est:.3f} -> Nsub={Nsub} dz_sub={dz_sub} um")
    print(f"[theta z coupling] theta_z_gamma={float(getattr(ctx, 'theta_z_gamma', 0.0))}")
    if timedep:
        print(f"[TD mode] {'true physical-time integrator' if true_td else 'pseudo stepper'}")
    else:
        print(f"[static mode] {static_mode}")

    amp = cp.empty_like(ctx.amp0)
    Ahat = cp.empty_like(ctx.amp0)
    I_b = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_a = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_mid = cp.empty((ctx.Nx, ctx.Ny), cp.float32)

    need_full_I_store = (not timedep) or bool(getattr(ctx, "save_full_I_mid_store", False))
    I_mid_store = cp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=store_I_dtype) if need_full_I_store else None

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=ctx.Ny
    )

    jt_iter = tqdm(range(tsteps), desc=("LC timedep" if timedep else "LC static"), dynamic_ncols=True) \
        if (tqdm_timedep and timedep) else range(tsteps)

    quasi_report = True

    if timedep and true_td and use_quasiglobal_td:
        if (not hasattr(ctx, "amp_time_state")) or (ctx.amp_time_state is None) or (ctx.amp_time_state.shape != ctx.amp0.shape):
            ctx.amp_time_state = ctx.amp0.copy()

    if not hasattr(ctx, "_amp_in_buf") or ctx._amp_in_buf.shape != amp.shape:
        ctx._amp_in_buf = cp.empty_like(amp)

    if not hasattr(ctx, "_amp_pred_buf") or ctx._amp_pred_buf.shape != amp.shape:
        ctx._amp_pred_buf = cp.empty_like(amp)

    if not hasattr(ctx, "_I_mid_pred_buf") or ctx._I_mid_pred_buf.shape != I_mid.shape:
        ctx._I_mid_pred_buf = cp.empty_like(I_mid)

    for jt in jt_iter:
        if timedep:
            dt_j = float(time_steps[jt])
            t_model += dt_j

            if true_td:
                a_ie, off, diag, lam_y = prepare_ie_ky_operator(
                    dt=dt_j,
                    mobility=float(ctx.mobility),
                    du=float(ctx.du),
                    dv=float(ctx.dv),
                    Ny=ctx.Ny
                )
                s = None
            else:
                s, off, diag, lam_y = prepare_cn_ky_operator(
                    dt=dt_j,
                    mobility=float(ctx.mobility),
                    du=float(ctx.du),
                    dv=float(ctx.dv),
                    Ny=ctx.Ny
                )
                a_ie = None

            if ctx.theta_prev_time is None or ctx.theta_prev_time.shape != ctx.theta_full.shape:
                ctx.theta_prev_time = cp.empty_like(ctx.theta_full)

            ctx.theta_prev_time[...] = ctx.theta_full
            theta_ref = ctx.theta_prev_time

            td_resid_last = 0.0
            td_resid_max = 0.0
        else:
            if ctx.theta_prev_time is None or ctx.theta_prev_time.shape != ctx.theta_full.shape:
                ctx.theta_prev_time = cp.empty_like(ctx.theta_full)

            ctx.theta_prev_time[...] = ctx.theta_full
            theta_ref = ctx.theta_prev_time
            dt_j = 0.0

        # =========================================================
        # QUASI-GLOBAL TD BRANCH
        # =========================================================
        if timedep and true_td and use_quasiglobal_td:
            if quasi_report:
                print("Quasi Global Branch")
                quasi_report = False

            td_resid_last, td_resid_max = run_quasiglobal_td_branch(
                ctx,
                dt_j=dt_j,
                store_I_dtype=store_I_dtype,
                I_mid_store=I_mid_store,
                save_slices_fn=save_slices_fn,
                t_stride=t_stride,
                jt=jt,
                jt_iter=jt_iter,
                report_runtime=report_runtime,
                tmp_dir="/tmp",
                use_memmap_pred=False,
                z_chunk=16,
            )

            continue
        # =====================================================
        linearized_td = bool(getattr(ctx, "linearized_td", False))
        amp[...] = ctx.amp0
        hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

        k_iter = tqdm(range(ctx.Nz), desc="LC z", dynamic_ncols=True) \
            if (tqdm_static_z and (not timedep)) else range(ctx.Nz)

        if not timedep:
            theta_running = ctx.theta_bias_2d.copy() if restart_static_from_bias else ctx.theta_full[0].copy()
        else:
            theta_running = None

        for k in k_iter:

            do_theta_update = (theta_k_stride <= 1) or ((k % theta_k_stride) == 0)
            if freeze_theta:
                do_theta_update = False

            intens_into(I_b, amp, coh=ctx.coh)

            if timedep:
                theta_use = theta_ref[k]
                tp = theta_ref[k - 1] if k > 0 else theta_ref[k]
                tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else theta_ref[k]
            else:
                if k > 0:
                    tp = 0.5 * (ctx.theta_full[k - 1] + theta_ref[k - 1])
                else:
                    tp = theta_running if restart_static_from_bias else ctx.theta_full[k]

                tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else (
                    theta_running if restart_static_from_bias else ctx.theta_full[k]
                )

            # ---------------- TD path ----------------
            if timedep:
                gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))

                if frozen_I_mid_store is not None:
                    amp_for_pred = amp
                    td_get_slice_mid_intensity(
                        ctx, amp, theta_use, I_b, I_a, I_mid,
                        Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
                        Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
                        frozen_I_mid_k=frozen_I_mid_store[k],
                    )
                else:
                    do_full_pred = bool(getattr(ctx, "true_td_full_pred_optics", False))
                    amp_for_pred = None
                    if do_full_pred:
                        amp_for_pred = ctx._amp_in_buf
                        amp_for_pred[...] = amp

                    td_get_slice_mid_intensity(
                        ctx, amp, theta_use, I_b, I_a, I_mid,
                        Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
                        Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
                        frozen_I_mid_k=None,
                    )

                if I_mid_store is not None:
                    I_mid_store[k] = I_mid.astype(store_I_dtype, copy=False)

                if do_theta_update:

                    if linearized_td:
                        ctx.theta_full[k] = advance_theta_linearized_td_step(
                            ctx,
                            theta_ref[k],
                            I_mid,
                            tp,
                            tn,
                            dt_j=dt_j,
                            off=off,
                            diag=diag,
                        )
                    else:

                        ctx.theta_full[k] = td_update_theta_from_midintensity(
                            ctx,
                            theta_ref[k],
                            tp,
                            tn,
                            I_mid,
                            dt_j=dt_j,
                            true_td=true_td,
                            a_ie=a_ie,
                            s=s,
                            off=off,
                            diag=diag,
                            lam_y=lam_y,
                            Nsub=Nsub,
                            dz_sub=dz_sub,
                            h_sub=h_sub,
                            amp_for_pred=amp_for_pred,
                            I_b=I_b,
                            I_a=I_a,
                            Ahat=Ahat,
                            plan_f=plan_f,
                            plan_i=plan_i,
                        )
                else:
                    ctx.theta_full[k] = theta_ref[k]

                theta_save = ctx.theta_full[k]

                runtime_every_k = int(getattr(ctx, "runtime_every_k", 0))
                if report_runtime:
                    do_report_here = (
                        runtime_every_k <= 0
                        or (k % runtime_every_k) == 0
                        or (k == ctx.Nz - 1)
                    )
                    if do_report_here:
                        rloc = _slice_residual_rms_local3d_z(
                            ctx.theta_full[k], I_mid, tp, tn, ctx,
                            float(getattr(ctx, "theta_z_gamma", 0.0))
                        )
                        td_resid_last = rloc
                        td_resid_max = max(td_resid_max, rloc)

            # ---------------- static path ----------------
            else:
                I_mid[...] = I_b

                did_selfconsistent_strict = False
                if do_theta_update:


                    if static_mode == "linear":
                        out_lin = solve_theta_perturbation_dirichletx_periody(
                            theta_bias=ctx.theta_bias_2d,
                            Ixy=I_mid,
                            b=float(ctx.b),
                            bi=float(ctx.bi),
                            du=float(ctx.du),
                            dv=float(ctx.dv),
                            xp=cp,
                            return_theta=True,
                            check_residual=False,
                            verbose=False,
                        )
                        theta_running = out_lin["theta"].astype(cp.float32, copy=False)

                    elif static_mode == "strict_relax":
                        theta_seed = ctx.theta_bias_2d if restart_static_from_bias else (
                            theta_running if (k > 0) else ctx.theta_full[k]
                        )

                        theta_running, I_mid_sc, amp_sc, info = strict_static_relax_slice_selfconsistent(
                            amp_in=amp,
                            theta_seed=theta_seed,
                            tp=tp,
                            tn=tn,
                            ctx=ctx,
                            dz_sub=dz_sub,
                            Nsub=Nsub,
                            h_sub=h_sub,
                            Ahat=Ahat,
                            plan_f=plan_f,
                            plan_i=plan_i,
                            sS=sS,
                            offS=offS,
                            diagS=diagS,
                            lamS=lamS,
                            use_linear_seed=use_linear_seed_for_static_relax,
                            early_accept_linear_seed=early_accept_linear_seed,
                            residual_tol_max=strict_residual_tol_max,
                            residual_tol_rms=strict_residual_tol_rms,
                            max_outer_passes=strict_max_outer_passes,
                            max_selfcons_passes=int(getattr(ctx, "strict_selfcons_passes", 3)),
                            selfcons_tol_theta=float(getattr(ctx, "strict_selfcons_tol_theta", 1e-4)),
                            selfcons_tol_I=float(getattr(ctx, "strict_selfcons_tol_I", 1e-4)),

                            verbose=False,
                        )

                        if True:
                          I_mid[...] = I_mid_sc
                          amp[...] = amp_sc
                          did_selfconsistent_strict = True

                        if False:
                            I_mid_candidate = I_mid_sc
                            amp[...] = amp_sc



                            print(
                                "[amp diag]",
                                "shape=", amp.shape,
                                "dtype=", amp.dtype,
                                "iscomplex=", cp.iscomplexobj(amp),
                            )

                            print(
                                "[amp diag]",
                                "abs max=", float(cp.max(cp.abs(amp))),
                                "abs min=", float(cp.min(cp.abs(amp))),
                            )

                            print(
                                "[amp diag]",
                                "power=", float(cp.sum(cp.abs(amp)**2) * ctx.dx * ctx.dy),
                            )

                            tmpI = cp.abs(amp)**2

                            print(
                                "[amp diag]",
                                "|amp|^2 max=", float(cp.max(tmpI)),
                                "|amp|^2 sum=", float(cp.sum(tmpI) * ctx.dx * ctx.dy),
                            )






                            # For launch diagnostics: strict_static_relax_slice_selfconsistent
                            # currently returns a bad I_mid_sc even when amp_sc is valid.
                            if (not bool(cp.all(cp.isfinite(I_mid_sc)))) or float(cp.max(I_mid_sc)) < 1e-12:
                                intens_into(I_mid, amp, coh=ctx.coh)
                                print(
                                    "[I_mid patch] replaced bad I_mid_sc:",
                                    "I_mid_sc max=", float(cp.max(I_mid_sc)),
                                    "amp power=", float(cp.sum(cp.abs(amp)**2) * ctx.dx * ctx.dy),
                                    "new Imax=", float(cp.max(I_mid)),
                                    "new Isum=", float(cp.sum(I_mid) * ctx.dx * ctx.dy),
                                )
                            else:
                                I_mid[...] = I_mid_sc

                            did_selfconsistent_strict = True

                        if (strict_verbose_every is not None) and ((k % strict_verbose_every) == 0 or k == ctx.Nz - 1):
                            print(
                                f"[k={k:4d}] max={info['max_interior']:.3e}  "
                                f"rms={info['rms_interior']:.3e}  "
                                f"sc_passes={info['n_selfcons_passes']}  "
                                f"ok={info['converged']}"
                            )

                    theta_running = enforce_theta_constraints(theta_running, ctx.theta_clamp)

                    ctx.theta_full[k] = theta_running
                    theta_save = theta_running
                    theta_use = theta_running
                else:
                    theta_current = theta_running if theta_running is not None else ctx.theta_full[k]

                    ctx.theta_full[k] = theta_current
                    theta_save = theta_current
                    theta_use = theta_current

                if not did_selfconsistent_strict:
                    for _ in range(Nsub):
                        apply_nonlinear_phase_inplace(
                            amp, theta_use,
                            dz_step_um=dz_sub,
                            kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                        )
                        hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

                    intens_into(I_a, amp, coh=ctx.coh)
                else:
                    intens_into(I_a, amp, coh=ctx.coh)

                if I_mid_store is not None:
                    I_mid_store[k] = I_mid.astype(store_I_dtype, copy=False)

            if (save_slices_fn is not None) and ((jt % t_stride) == 0):
                slot = jt // t_stride
                save_slices_fn(slot, k, I_a, theta_save)

        hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

        if timedep and report_runtime and hasattr(jt_iter, "set_postfix"):
            try:
                jt_iter.set_postfix(
                    rlast=f"{td_resid_last:.2e}",
                    rmax=f"{td_resid_max:.2e}",
                )
            except Exception:
                pass

        if timedep and checkpoint_dir and checkpoint_every:
            if (((jt + 1) % int(checkpoint_every)) == 0) or (jt == (tsteps - 1)):
                last_movie_slot = None
                if jt >= 0:
                    last_movie_slot = jt // t_stride

                save_lc_checkpoint(
                    checkpoint_dir,
                    prefix=checkpoint_prefix,
                    step_kind="jt",
                    step_index=jt,
                    t_model=t_model,
                    theta_full=ctx.theta_full,
                    Ixz=checkpoint_Ixz,
                    Iyz=checkpoint_Iyz,
                    thetaxz=checkpoint_thetaxz,
                    thetayz=checkpoint_thetayz,
                    keep_history=checkpoint_keep_history,
                    last_movie_slot=last_movie_slot,
                )
    archive_dir = None
    if checkpoint_dir is not None:
        archive_dir = finalize_checkpoint_archive(
            checkpoint_dir,
            prefix=checkpoint_prefix
        )

        # save prdata if available
        if hasattr(ctx, "prdata"):
            save_prdata_json(
                ctx.prdata,
                os.path.join(archive_dir, "prdata.json")
            )

    return I_mid_store



# ================================================================================

# lc_dn_from_theta

def lc_dn_from_theta(theta, ne, no, refin):
    th64 = theta.astype(cp.float64, copy=False)
    ct, st = cp.cos(th64), cp.sin(th64)
    n_eff = (ne * no) / cp.sqrt((ne * ct)**2 + (no * st)**2)
    return (n_eff - refin).astype(cp.float32, copy=False)



# ================================================================================

# fill_mid_intensity

def fill_mid_intensity(I_mid, I_b, I_a):
    I_mid[...] = 0.5 * (I_b + I_a)
    return I_mid



# ================================================================================

# strict_static_relax_slice

def strict_static_relax_slice(
    theta_seed, I_mid, tp, tn, ctx, *,
    sS, offS, diagS, lamS,
    use_linear_seed=True,
    early_accept_linear_seed=True,
    residual_tol_max=5e-2,
    residual_tol_rms=1e-2,
    max_outer_passes=8,
    verbose=False,
):
    if use_linear_seed:
        out_lin = solve_theta_perturbation_dirichletx_periody(
            theta_bias=ctx.theta_bias_2d,
            Ixy=I_mid,
            b=float(ctx.b),
            bi=float(ctx.bi),
            du=float(ctx.du),
            dv=float(ctx.dv),
            xp=cp,
            return_theta=True,
            check_residual=False,
            verbose=False,
        )
        theta = out_lin["theta"].astype(cp.float32, copy=False)
    else:
        theta = theta_seed.astype(cp.float32, copy=False)

    theta = enforce_theta_constraints(theta, ctx.theta_clamp)

    if early_accept_linear_seed:
        R2 = lc_residual2d_dirichletx_periody(
            theta, I_mid,
            b=float(ctx.b), bi=float(ctx.bi),
            du=float(ctx.du), dv=float(ctx.dv),
            mobility=float(ctx.mobility),
        )
        stats0 = residual_stats_2d(R2)
        if (stats0["max_interior"] <= residual_tol_max) and (stats0["rms_interior"] <= residual_tol_rms):
            return theta, {
                "converged": True,
                "n_outer_passes": 0,
                "max_interior": stats0["max_interior"],
                "rms_interior": stats0["rms_interior"],
            }

    history_max = []
    history_rms = []
    converged = False

    for outer in range(max_outer_passes):
        theta = _static_relax_to_resid_zcoupled(
            theta, I_mid, tp, tn, ctx,
            sS=sS, offS=offS, diagS=diagS, lamS=lamS
        ).astype(cp.float32, copy=False)

        if ctx.theta_clamp is not None:
            theta = cp.clip(theta, cp.float32(ctx.theta_clamp[0]), cp.float32(ctx.theta_clamp[1]))
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta[0, :] = theta_bc
        theta[-1, :] = theta_bc

        R2 = lc_residual2d_dirichletx_periody(
            theta, I_mid,
            b=float(ctx.b), bi=float(ctx.bi),
            du=float(ctx.du), dv=float(ctx.dv),
            mobility=float(ctx.mobility),
        )
        stats = residual_stats_2d(R2)
        history_max.append(stats["max_interior"])
        history_rms.append(stats["rms_interior"])

        if verbose:
            print(f"[strict static] pass={outer+1:2d} max={stats['max_interior']:.3e} rms={stats['rms_interior']:.3e}")

        if (stats["max_interior"] <= residual_tol_max) and (stats["rms_interior"] <= residual_tol_rms):
            converged = True
            break

    return theta, {
        "converged": converged,
        "n_outer_passes": outer + 1,
        "max_interior": history_max[-1],
        "rms_interior": history_rms[-1],
    }



# ================================================================================

# _extract_theta_b_1d

def _extract_theta_b_1d(theta_bias, xp):
    th = xp.asarray(theta_bias)
    if th.ndim == 1:
        return th
    if th.ndim == 2:
        return xp.mean(th, axis=1)
    raise ValueError("theta_bias must have shape (Nx,) or (Nx,Ny).")



# ================================================================================

# apply_Lpert_delta

def apply_Lpert_delta(delta, theta_b, b, du, dv, xp):
    theta_b = xp.asarray(theta_b)
    delta = xp.asarray(delta)

    qx = 2.0 * b * xp.cos(2.0 * theta_b)

    left = xp.zeros_like(delta)
    right = xp.zeros_like(delta)
    left[1:, :] = delta[:-1, :]
    right[:-1, :] = delta[1:, :]
    d2x = (left - 2.0 * delta + right) / (du * du)
    d2y = (xp.roll(delta, 1, axis=1) - 2.0 * delta + xp.roll(delta, -1, axis=1)) / (dv * dv)

    return d2x + d2y + qx[:, None] * delta



# ================================================================================

# thomas_batched_modevarying

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



# ================================================================================

# lc_residual2d_dirichletx_periody

def lc_residual2d_dirichletx_periody(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    return lc_residual64(theta, Ixy, b=b, bi=bi, du=du, dv=dv, mobility=mobility)



# ================================================================================

# lc_residual64

def lc_residual64(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    th = theta.astype(cp.float64, copy=False)
    I64 = Ixy.astype(cp.float64, copy=False)

    du2 = cp.float64(du) * cp.float64(du)
    dv2 = cp.float64(dv) * cp.float64(dv)

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    lap = cp.zeros_like(th)

    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        +
        (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    drive = (cp.float64(b) + cp.float64(bi) * I64) * cp.sin(2.0 * th)

    R = (lap + drive) / cp.float64(mobility)

    # prescribed Dirichlet rows are not residual-tested
    R[0, :] = 0.0
    R[-1, :] = 0.0

    return R



# ================================================================================

# _static_relax_to_resid_zcoupled

def _static_relax_to_resid_zcoupled(theta_seed, Ixy, theta_prev, theta_next, ctx, *, sS, offS, diagS, lamS):
    th = theta_seed.astype(cp.float32, copy=True)
    I32 = Ixy.astype(cp.float32, copy=False)
    gamma_z = float(getattr(ctx, "theta_z_gamma", 0.0))
    inv_dz2 = _get_inv_dz2(ctx)

    max_steps = int(getattr(ctx, "static_max_steps", 200))
    resid_every = int(getattr(ctx, "static_resid_every", 10))
    tol = float(getattr(ctx, "static_tol_resid", 0.08))
    omega = float(getattr(ctx, "static_relax_omega", 0.3))

    for it in range(max_steps):
        th_new = advance_theta_timestep_cn_trap_picard_prepared_zcoupled(ctx, 
            th,
            dt=float(ctx.dtau_static),
            b=ctx.b,
            bi=ctx.bi,
            Ixy=I32,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            s=sS,
            off=offS,
            diag=diagS,
            lam_y=lamS,
            max_iter=ctx.picard_iters,
            tol_update=ctx.picard_tol_up,
            clamp=ctx.theta_clamp,
            inv_dz2=inv_dz2,
        )

        th = (1.0 - omega) * th + omega * th_new
        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        th[0, :] = theta_bc
        th[-1, :] = theta_bc

        if ctx.theta_clamp is not None:
            th = cp.clip(th, cp.float32(ctx.theta_clamp[0]), cp.float32(ctx.theta_clamp[1]))

        if (it % resid_every) == 0 or it == (max_steps - 1):
            rloc = _slice_residual_rms_local3d_z(th, I32, theta_prev, theta_next, ctx, gamma_z)
            if rloc < tol:
                break

    return th



# ================================================================================

# residual_stats_2d

def residual_stats_2d(R):
    R = R.astype(cp.float64, copy=False)
    Rint = R[1:-1, :]
    return {
        "max_full": float(cp.max(cp.abs(R))),
        "rms_full": float(cp.sqrt(cp.mean(R * R))),
        "max_interior": float(cp.max(cp.abs(Rint))),
        "rms_interior": float(cp.sqrt(cp.mean(Rint * Rint))),
    }



# ================================================================================

# _get_inv_dz2

def _get_inv_dz2(ctx):
    dz = float(ctx.dz)
    return cp.float32(1.0 / (dz * dz))



# ================================================================================

# advance_theta_timestep_cn_trap_picard_prepared_zcoupled

def advance_theta_timestep_cn_trap_picard_prepared_zcoupled(ctx, 
    theta_n, *, dt, b, bi, Ixy,
    theta_prev, theta_next, gamma_z,
    mobility, du, dv,
    s, off, diag, lam_y,
    max_iter, tol_update, clamp,
    inv_dz2
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    tp = theta_prev.astype(cp.float32, copy=False)
    tn = theta_next.astype(cp.float32, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv)).astype(cp.float32, copy=False)
    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    N_n = alpha * cp.sin(2.0 * theta_n)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs_base = (
        theta_n
        + s * lap_n
        + cp.float32(0.5) * dt_over_m * N_n
        + dt_over_m * gam * (tp + tn)
    )
    rhs_base[0, :] = 0.0
    rhs_base[-1, :] = 0.0

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam

    theta_g = advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(ctx, 
        theta_n,
        dt=dt, b=b, bi=bi, Ixy=Ixy,
        theta_prev=tp, theta_next=tn, gamma_z=gamma_z,
        mobility=mobility, du=du, dv=dv,
        s=s, off=off, diag=diag, lam_y=lam_y,
        inv_dz2=inv_dz2
    )

    for _ in range(int(max_iter)):
        N_g = alpha * cp.sin(2.0 * theta_g)
        rhs = rhs_base + cp.float32(0.5) * dt_over_m * N_g
        rhs[0, :] = 0.0
        rhs[-1, :] = 0.0

        theta_new = _cn_solve_dirichletx_periody(ctx, rhs, off=off, diag=diag_eff)
        theta_new = enforce_theta_constraints(theta_new, clamp)

        dth = theta_new - theta_g
        rms_up = float(cp.sqrt(cp.mean(dth * dth)))
        theta_g = theta_new

        if rms_up < float(tol_update):
            break

    return theta_g



# ================================================================================

# _laplacian_dirichletx_periody

def _laplacian_dirichletx_periody(theta, du, dv):
    th = theta.astype(cp.float32, copy=False)

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    du2 = cp.float32(du * du)
    dv2 = cp.float32(dv * dv)

    lap = cp.zeros_like(th)

    # interior ONLY
    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        + (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    # boundary rows remain zero
    return lap



# ================================================================================

# _cn_solve_dirichletx_periody

def _cn_solve_dirichletx_periody(ctx, rhs, *, off, diag):
    theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))

    rhs2 = rhs.astype(cp.float32, copy=True)
    rhs2[0, :] = 0.0
    rhs2[-1, :] = 0.0

    rhs_hat = cp.fft.fft(rhs2, axis=1).astype(cp.complex64, copy=False)
    d_hatB = rhs_hat[1:-1, :].T.copy(order="C")  # (Ny, Nx-2)

    Ny_local = rhs.shape[1]
    bc_hat = cp.fft.fft(
        cp.full((Ny_local,), theta_bc, dtype=cp.float32)
    ).astype(cp.complex64)

    d_hatB[:, 0]  -= off * bc_hat
    d_hatB[:, -1] -= off * bc_hat

    x_hatB = thomas_batched_const_tridiag(off, diag, off, d_hatB)

    theta_hat = cp.zeros_like(rhs_hat)
    theta_hat[1:-1, :] = x_hatB.T

    th = cp.fft.ifft(theta_hat, axis=1).real.astype(cp.float32, copy=False)
    th[0, :] = theta_bc
    th[-1, :] = theta_bc
    return th



# ================================================================================

# advance_theta_timestep_cn_fft_thomas_prepared_zcoupled

def advance_theta_timestep_cn_fft_thomas_prepared_zcoupled(ctx, 
    theta_n, *, dt, b, bi, Ixy,
    theta_prev, theta_next, gamma_z,
    mobility, du, dv,
    s, off, diag, lam_y,
    inv_dz2
):
    theta_n = theta_n.astype(cp.float32, copy=False)
    Ixy = Ixy.astype(cp.float32, copy=False)
    tp = theta_prev.astype(cp.float32, copy=False)
    tn = theta_next.astype(cp.float32, copy=False)

    lap_n = _laplacian_dirichletx_periody(theta_n, float(du), float(dv)).astype(cp.float32, copy=False)
    alpha = cp.float32(b) + cp.float32(bi) * Ixy
    drive_n = alpha * cp.sin(2.0 * theta_n)

    gam = cp.float32(gamma_z) * inv_dz2
    dt_over_m = cp.float32(float(dt) / float(mobility))

    rhs = theta_n + s * lap_n + dt_over_m * (drive_n + gam * (tp + tn))
    rhs[0, :] = 0.0
    rhs[-1, :] = 0.0

    diag_eff = diag + cp.float32(2.0) * dt_over_m * gam
    return _cn_solve_dirichletx_periody(ctx, rhs, off=off, diag=diag_eff)



# ================================================================================

# thomas_batched_const_tridiag

def thomas_batched_const_tridiag(a, bvec, c, d_hatB):
    B, n = d_hatB.shape
    d_hatB = cp.ascontiguousarray(d_hatB.astype(cp.complex64, copy=False))
    x = cp.empty_like(d_hatB)
    cprime = cp.empty_like(d_hatB)
    dprime = cp.empty_like(d_hatB)
    threads = 128
    blocks = (B + threads - 1) // threads
    _thomas_kernel(
        (blocks,), (threads,),
        (
            cp.float32(a),
            bvec.astype(cp.float32),
            cp.float32(c),
            d_hatB.view(cp.float32).reshape(B, n, 2),
            x.view(cp.float32).reshape(B, n, 2),
            cprime.view(cp.float32).reshape(B, n, 2),
            dprime.view(cp.float32).reshape(B, n, 2),
            cp.int32(B),
            cp.int32(n)
        )
    )
    return x



# ================================================================================

# _thomas_kernel

_thomas_kernel = cp.RawKernel(r'''


extern "C" __global__
void thomas_kernel(
    const float a,
    const float* __restrict__ bvec,
    const float c,
    const float2* __restrict__ d,
    float2* __restrict__ x,
    float2* __restrict__ cprime,
    float2* __restrict__ dprime,
    const int B,
    const int n
){
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= B) return;

    int base = k * n;
    float b0 = bvec[k];
    float denom = b0;
    float2 d0 = d[base + 0];

    cprime[base + 0] = make_float2(c / denom, 0.0f);
    dprime[base + 0] = make_float2(d0.x / denom, d0.y / denom);

    for (int i = 1; i < n; ++i){
        float2 cp_im1 = cprime[base + (i-1)];
        denom = b0 - a * cp_im1.x;
        cprime[base + i] = make_float2(c / denom, 0.0f);
        float2 di = d[base + i];
        float2 dp_im1 = dprime[base + (i-1)];
        float2 num;
        num.x = di.x - a * dp_im1.x;
        num.y = di.y - a * dp_im1.y;
        dprime[base + i] = make_float2(num.x / denom, num.y / denom);
    }

    x[base + (n-1)] = dprime[base + (n-1)];
    for (int i = n-2; i >= 0; --i){
        float2 dpi = dprime[base + i];
        float2 cpi = cprime[base + i];
        float2 xi1 = x[base + (i+1)];
        float2 val;
        val.x = dpi.x - cpi.x * xi1.x;
        val.y = dpi.y - cpi.x * xi1.y;
        x[base + i] = val;
    }
}


''', 'thomas_kernel')
