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

import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .runner_core import *

def save_movie(data, fname, fps=5):
    fig, ax = plt.subplots()
    im = ax.imshow(data[0].T, origin="lower", aspect="auto")

    def update(i):
        im.set_data(data[i].T)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(data), blit=True)
    ani.save(fname, fps=fps)
    plt.close(fig)

# yz movie

def _cp_to_np_downsample2d(a, max_nx=900, max_ny=700):
    if a is None:
        return None, 1, 1
    A = cp.asnumpy(a)
    ny, nx = A.shape
    sy = max(1, int(np.ceil(ny / max_ny)))
    sx = max(1, int(np.ceil(nx / max_nx)))
    return A[::sy, ::sx], sy, sx

def last_filled_frame(arr):
    if arr is None:
        return None
    mags = cp.max(cp.abs(arr), axis=(1, 2))
    idx = cp.where(mags > 0)[0]
    if idx.size == 0:
        return None
    return int(idx[-1].item())

def _resolve_frame(arr, frame):
    if arr is None:
        return None
    if frame is None:
        frame = last_filled_frame(arr)
    if frame is None:
        return None
    return int(frame)

def show_map_physical(arr2d, *, x0, dx, y0, dy, title="", xlabel="", ylabel="", max_nx=900, max_ny=700):
    if arr2d is None:
        print(f"{title or 'array'} is None.")
        return

    A, sy, sx = _cp_to_np_downsample2d(arr2d, max_nx=max_nx, max_ny=max_ny)
    ny_ds, nx_ds = A.shape

    x1 = x0 + dx * sx * (nx_ds - 1)
    y1 = y0 + dy * sy * (ny_ds - 1)

    plt.figure(figsize=(7, 4))
    plt.imshow(
        A,
        origin="lower",
        aspect="auto",
        extent=[x0, x1, y0, y1],
    )
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def show_theta_yz_physical(thetayz, ctx, frame=None, *, delta_theta=True, max_nx=900, max_ny=700):
    if thetayz is None:
        print("thetayz is None (store_slice_movies was off).")
        return

    frame = _resolve_frame(thetayz, frame)
    if frame is None:
        print("No filled frames found in thetayz.")
        return

    arr = thetayz[frame].T
    y0 = -0.5 * ctx.Ny * ctx.dy
    z0 = 0.0

    title = f"{'Δθ' if delta_theta else 'θ'} yz, frame {frame}"
    show_map_physical(
        arr,
        x0=z0, dx=ctx.dz,
        y0=y0, dy=ctx.dy,
        title=title,
        xlabel="z (µm)",
        ylabel="y (µm)",
        max_nx=max_nx,
        max_ny=max_ny,
    )

def show_theta_xz_physical(thetaxz, ctx, frame=None, *, delta_theta=True, max_nx=900, max_ny=700):
    if thetaxz is None:
        print("thetaxz is None (store_slice_movies was off).")
        return

    frame = _resolve_frame(thetaxz, frame)
    if frame is None:
        print("No filled frames found in thetaxz.")
        return

    arr = thetaxz[frame].T
    x0 = -0.5 * ctx.Nx * ctx.dx
    z0 = 0.0

    title = f"{'Δθ' if delta_theta else 'θ'} xz, frame {frame}"
    show_map_physical(
        arr,
        x0=z0, dx=ctx.dz,
        y0=x0, dy=ctx.dx,
        title=title,
        xlabel="z (µm)",
        ylabel="x (µm)",
        max_nx=max_nx,
        max_ny=max_ny,
    )

def show_Iyz_physical(Iyz, ctx, frame=None, *, max_nx=900, max_ny=700):
    if Iyz is None:
        print("Iyz is None (store_slice_movies was off).")
        return

    frame = _resolve_frame(Iyz, frame)
    if frame is None:
        print("No filled frames found in Iyz.")
        return

    arr = Iyz[frame].T
    y0 = -0.5 * ctx.Ny * ctx.dy
    z0 = 0.0

    show_map_physical(
        arr,
        x0=z0, dx=ctx.dz,
        y0=y0, dy=ctx.dy,
        title=f"Iyz, frame {frame}",
        xlabel="z (µm)",
        ylabel="y (µm)",
        max_nx=max_nx,
        max_ny=max_ny,
    )

def show_Ixz_physical(Ixz, ctx, frame=None, *, max_nx=900, max_ny=700):
    if Ixz is None:
        print("Ixz is None (store_slice_movies was off).")
        return

    frame = _resolve_frame(Ixz, frame)
    if frame is None:
        print("No filled frames found in Ixz.")
        return

    arr = Ixz[frame].T
    x0 = -0.5 * ctx.Nx * ctx.dx
    z0 = 0.0

    show_map_physical(
        arr,
        x0=z0, dx=ctx.dz,
        y0=x0, dy=ctx.dx,
        title=f"Ixz, frame {frame}",
        xlabel="z (µm)",
        ylabel="x (µm)",
        max_nx=max_nx,
        max_ny=max_ny,
    )

def show_theta_xy_movie_along_z(
    ctx,
    *,
    z_stride=4,                 # show every 4th z slice
    fps=8,
    interval_ms=100,
    delta_theta=True,           # plot theta - theta_bias_2d
    robust_pct=99.5,            # color scaling percentile
    repeat=False,
):
    theta_full = ctx.theta_full
    theta_bias = ctx.theta_bias_2d

    Nz, Nx, Ny = theta_full.shape
    frames = np.arange(0, Nz, z_stride, dtype=int)

    # small probe for color scale, not full-volume copy
    probe_idx = frames[::max(1, len(frames)//32)]
    vals = []
    for k in probe_idx:
        A = theta_full[k]
        if delta_theta:
            A = A - theta_bias
        vals.append(cp.asnumpy(A[::max(1, Nx//64), ::max(1, Ny//64)]).ravel())
    vals = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)

    vmax = np.percentile(np.abs(vals), robust_pct) if delta_theta else np.percentile(vals, robust_pct)
    vmax = max(float(vmax), 1e-8)
    vmin = -vmax if delta_theta else float(np.min(vals))

    x0 = -0.5 * ctx.Nx * ctx.dx
    x1 = x0 + ctx.dx * (ctx.Nx - 1)
    y0 = -0.5 * ctx.Ny * ctx.dy
    y1 = y0 + ctx.dy * (ctx.Ny - 1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    k0 = int(frames[0])
    A0 = theta_full[k0]
    if delta_theta:
        A0 = A0 - theta_bias
    A0 = cp.asnumpy(A0)

    im = ax.imshow(
        A0,
        origin="lower",
        aspect="auto",
        extent=[y0, y1, x0, x1],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("x (µm)")
    ttl = ax.set_title(f"{'Δθ' if delta_theta else 'θ'} at z = {k0*ctx.dz:.1f} µm")
    plt.colorbar(im, ax=ax)

    def update(i):
        k = int(frames[i])
        A = theta_full[k]
        if delta_theta:
            A = A - theta_bias
        im.set_data(cp.asnumpy(A))
        ttl.set_text(f"{'Δθ' if delta_theta else 'θ'} at z = {k*ctx.dz:.1f} µm")
        return (im, ttl)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(fps=fps)))
    return ani

# @title I(x,y) movie along z

def show_intensity_xy_movie_along_z(
    ctx,
    *,
    z_stride=4,
    fps=8,
    interval_ms=100,
    mode="log",          # "linear" | "log" | "normalized" | "gamma"
    robust_pct=99.5,
    gamma=0.3,
    repeat=False,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    Nx, Ny, Nz = ctx.Nx, ctx.Ny, ctx.Nz
    frames = np.arange(0, Nz, z_stride, dtype=int)

    # --- optics setup (local, consistent with runner) ---
    Nsub, dz_sub, _ = choose_optics_substeps(
        ctx.dz,
        kout=ctx.kout,
        dn_max_est=ctx.dn_max_est,
        dz_opt_max_phi=ctx.dz_opt_max_phi,
        max_substeps=ctx.max_substeps
    )
    h_sub  = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)

    amp  = cp.empty_like(ctx.amp0)
    Ahat = cp.empty_like(ctx.amp0)

    I_b   = cp.empty((Nx, Ny), cp.float32)
    I_a   = cp.empty((Nx, Ny), cp.float32)
    I_mid = cp.empty((Nx, Ny), cp.float32)

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    # --- display transform ---
    def transform(I):
        if mode == "log":
            return cp.log10(I + 1e-6)
        elif mode == "normalized":
            return I / (cp.max(I) + 1e-8)
        elif mode == "gamma":
            return cp.power(I / (cp.max(I) + 1e-8), gamma)
        else:  # linear
            return I

    # --- color scaling ---
    if mode == "linear":
        vals = []
        probe = frames[::max(1, len(frames)//32)]
        tmp_amp = ctx.amp0.copy()
        tmp_Ahat = cp.empty_like(tmp_amp)

        tmp_plan_f = spfft.get_fft_plan(tmp_Ahat, axes=(-2, -1))
        tmp_plan_i = spfft.get_fft_plan(tmp_Ahat, axes=(-2, -1), value_type="C2C")

        hop_linear_inplace(tmp_amp, h_half, ctx.windowxy, tmp_Ahat,
                           plan_f=tmp_plan_f, plan_i=tmp_plan_i)

        kcur = 0
        for k in probe:
            while kcur <= k:
                intens_into(I_b, tmp_amp, coh=ctx.coh)

                for _ in range(Nsub):
                    apply_nonlinear_phase_inplace(
                        tmp_amp, ctx.theta_full[kcur],
                        dz_step_um=dz_sub,
                        kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                    )
                    hop_linear_inplace(tmp_amp, h_sub, ctx.windowxy, tmp_Ahat,
                                       plan_f=tmp_plan_f, plan_i=tmp_plan_i)

                intens_into(I_a, tmp_amp, coh=ctx.coh)
                fill_mid_intensity(I_mid, I_b, I_a)

                kcur += 1

            vals.append(cp.asnumpy(I_mid[::max(1,Nx//64), ::max(1,Ny//64)]).ravel())

        vals = np.concatenate(vals) if vals else np.array([0.0])
        vmin = 0.0
        vmax = max(np.percentile(vals, robust_pct), 1e-8)

    else:
        vmin, vmax = None, None  # autoscale per frame

    # --- axes ---
    x0 = -0.5 * Nx * ctx.dx
    x1 = x0 + ctx.dx * (Nx - 1)
    y0 = -0.5 * Ny * ctx.dy
    y1 = y0 + ctx.dy * (Ny - 1)

    # --- initialize ---
    amp[...] = ctx.amp0
    hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat,
                       plan_f=plan_f, plan_i=plan_i)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    current_k = 0
    target_k0 = int(frames[0])

    while current_k <= target_k0:
        intens_into(I_b, amp, coh=ctx.coh)

        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp, ctx.theta_full[current_k],
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat,
                               plan_f=plan_f, plan_i=plan_i)

        intens_into(I_a, amp, coh=ctx.coh)
        fill_mid_intensity(I_mid, I_b, I_a)


        current_k += 1

    I_disp = transform(I_mid)

    im = ax.imshow(
        cp.asnumpy(I_disp),
        origin="lower",
        aspect="auto",
        extent=[y0, y1, x0, x1],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("y (µm)")
    ax.set_ylabel("x (µm)")
    ttl = ax.set_title(f"I(x,y) at z = {target_k0*ctx.dz:.1f} µm  [{mode}]")
    plt.colorbar(im, ax=ax)

    # --- animation ---
    def update(i):
        nonlocal current_k
        target_k = int(frames[i])

        while current_k <= target_k:
            intens_into(I_b, amp, coh=ctx.coh)

            for _ in range(Nsub):
                apply_nonlinear_phase_inplace(
                    amp, ctx.theta_full[current_k],
                    dz_step_um=dz_sub,
                    kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                )
                hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat,
                                   plan_f=plan_f, plan_i=plan_i)

            intens_into(I_a, amp, coh=ctx.coh)
            fill_mid_intensity(I_mid, I_b, I_a)


            current_k += 1

        I_disp = transform(I_mid)
        im.set_data(cp.asnumpy(I_disp))

        if mode != "linear":
            im.set_clim(float(cp.min(I_disp)), float(cp.max(I_disp)))

        ttl.set_text(f"I(x,y) at z = {target_k*ctx.dz:.1f} µm  [{mode}]")
        return (im, ttl)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(fps=fps)))
    return ani

# @title I(x,y) movie along z, cropped around beam core

def show_intensity_xy_movie_along_z_cropped(
    ctx,
    *,
    z_stride=4,
    fps=8,
    interval_ms=100,
    mode="log",          # "linear" | "log" | "normalized" | "gamma"
    robust_pct=99.5,
    gamma=0.3,
    repeat=False,
    crop_y_um=120.0,     # displayed y width in microns
    crop_x_um=None,      # displayed x width in microns; None -> full x
    center_mode="peak",  # "peak" | "centroid"
    lock_window=True,    # keep same crop window for whole movie
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    Nx, Ny, Nz = ctx.Nx, ctx.Ny, ctx.Nz
    frames = np.arange(0, Nz, z_stride, dtype=int)

    # --- optics setup (local, consistent with runner) ---
    Nsub, dz_sub, _ = choose_optics_substeps(
        ctx.dz,
        kout=ctx.kout,
        dn_max_est=ctx.dn_max_est,
        dz_opt_max_phi=ctx.dz_opt_max_phi,
        max_substeps=ctx.max_substeps
    )
    h_sub  = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)

    amp  = cp.empty_like(ctx.amp0)
    Ahat = cp.empty_like(ctx.amp0)

    I_b   = cp.empty((Nx, Ny), cp.float32)
    I_a   = cp.empty((Nx, Ny), cp.float32)
    I_mid = cp.empty((Nx, Ny), cp.float32)

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    # physical coordinates
    x_coords = (np.arange(Nx) - Nx / 2) * ctx.dx
    y_coords = (np.arange(Ny) - Ny / 2) * ctx.dy

    # --- display transform ---
    def transform(I):
        if mode == "log":
            return cp.log10(I + 1e-6)
        elif mode == "normalized":
            return I / (cp.max(I) + 1e-8)
        elif mode == "gamma":
            return cp.power(I / (cp.max(I) + 1e-8), gamma)
        else:  # linear
            return I

    def find_center_indices(Ixy):
        if center_mode == "centroid":
            I = Ixy.astype(cp.float32, copy=False)
            s = cp.sum(I)
            s = cp.maximum(s, cp.float32(1e-20))

            x_idx = cp.arange(Nx, dtype=cp.float32)[:, None]
            y_idx = cp.arange(Ny, dtype=cp.float32)[None, :]

            xc = int(cp.round(cp.sum(I * x_idx) / s).get())
            yc = int(cp.round(cp.sum(I * y_idx) / s).get())
            xc = max(0, min(Nx - 1, xc))
            yc = max(0, min(Ny - 1, yc))
            return xc, yc

        # default: peak
        flat_idx = int(cp.argmax(Ixy).get())
        xc, yc = np.unravel_index(flat_idx, (Nx, Ny))
        return int(xc), int(yc)

    def make_crop_slices(xc, yc):
        if crop_y_um is None:
            ylo, yhi = 0, Ny
        else:
            hy = max(1, int(round(0.5 * crop_y_um / ctx.dy)))
            ylo = max(0, yc - hy)
            yhi = min(Ny, yc + hy + 1)

        if crop_x_um is None:
            xlo, xhi = 0, Nx
        else:
            hx = max(1, int(round(0.5 * crop_x_um / ctx.dx)))
            xlo = max(0, xc - hx)
            xhi = min(Nx, xc + hx + 1)

        return slice(xlo, xhi), slice(ylo, yhi)

    def cropped_extent(xs, ys):
        xlo = x_coords[xs.start]
        xhi = x_coords[xs.stop - 1]
        ylo = y_coords[ys.start]
        yhi = y_coords[ys.stop - 1]
        return [ylo, yhi, xlo, xhi]

    # --- initialize propagation ---
    amp[...] = ctx.amp0
    hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat,
                       plan_f=plan_f, plan_i=plan_i)

    current_k = 0
    target_k0 = int(frames[0])

    while current_k <= target_k0:
        intens_into(I_b, amp, coh=ctx.coh)

        for _ in range(Nsub):
            apply_nonlinear_phase_inplace(
                amp, ctx.theta_full[current_k],
                dz_step_um=dz_sub,
                kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
            )
            hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat,
                               plan_f=plan_f, plan_i=plan_i)

        intens_into(I_a, amp, coh=ctx.coh)
        fill_mid_intensity(I_mid, I_b, I_a)


        current_k += 1

    # initial crop window
    xc0, yc0 = find_center_indices(I_mid)
    xs_fixed, ys_fixed = make_crop_slices(xc0, yc0)

    # --- color scaling ---
    if mode == "linear":
        vals = []
        probe = frames[::max(1, len(frames)//32)]

        tmp_amp = ctx.amp0.copy()
        tmp_Ahat = cp.empty_like(tmp_amp)
        tmp_plan_f = spfft.get_fft_plan(tmp_Ahat, axes=(-2, -1))
        tmp_plan_i = spfft.get_fft_plan(tmp_Ahat, axes=(-2, -1), value_type="C2C")

        hop_linear_inplace(tmp_amp, h_half, ctx.windowxy, tmp_Ahat,
                           plan_f=tmp_plan_f, plan_i=tmp_plan_i)

        kcur = 0
        for k in probe:
            while kcur <= k:
                intens_into(I_b, tmp_amp, coh=ctx.coh)

                for _ in range(Nsub):
                    apply_nonlinear_phase_inplace(
                        tmp_amp, ctx.theta_full[kcur],
                        dz_step_um=dz_sub,
                        kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                    )
                    hop_linear_inplace(tmp_amp, h_sub, ctx.windowxy, tmp_Ahat,
                                       plan_f=tmp_plan_f, plan_i=tmp_plan_i)

                intens_into(I_a, tmp_amp, coh=ctx.coh)
                fill_mid_intensity(I_mid, I_b, I_a)

                kcur += 1

            if lock_window:
                xs, ys = xs_fixed, ys_fixed
            else:
                xc, yc = find_center_indices(I_mid)
                xs, ys = make_crop_slices(xc, yc)

            vals.append(cp.asnumpy(I_mid[xs, ys]).ravel())

        vals = np.concatenate(vals) if vals else np.array([0.0])
        vmin = 0.0
        vmax = max(np.percentile(vals, robust_pct), 1e-8)
    else:
        vmin, vmax = None, None

    # --- initial image ---
    if lock_window:
        xs, ys = xs_fixed, ys_fixed
    else:
        xc, yc = find_center_indices(I_mid)
        xs, ys = make_crop_slices(xc, yc)

    I_disp = transform(I_mid[xs, ys])
    extent = cropped_extent(xs, ys)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im = ax.imshow(
        cp.asnumpy(I_disp),
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("y (µm)")
    ax.set_ylabel("x (µm)")
    ttl = ax.set_title(
        f"I(x,y) at z = {target_k0*ctx.dz:.1f} µm  [{mode}]"
    )
    plt.colorbar(im, ax=ax)

    # --- animation ---
    def update(i):
        nonlocal current_k

        target_k = int(frames[i])

        while current_k <= target_k:
            intens_into(I_b, amp, coh=ctx.coh)

            for _ in range(Nsub):
                apply_nonlinear_phase_inplace(
                    amp, ctx.theta_full[current_k],
                    dz_step_um=dz_sub,
                    kout=ctx.kout, ne=ctx.ne, no=ctx.no, refin=ctx.refin
                )
                hop_linear_inplace(amp, h_sub, ctx.windowxy, Ahat,
                                   plan_f=plan_f, plan_i=plan_i)

            intens_into(I_a, amp, coh=ctx.coh)
            fill_mid_intensity(I_mid, I_b, I_a)


            current_k += 1

        if lock_window:
            xs, ys = xs_fixed, ys_fixed
        else:
            xc, yc = find_center_indices(I_mid)
            xs, ys = make_crop_slices(xc, yc)

        I_disp = transform(I_mid[xs, ys])
        im.set_data(cp.asnumpy(I_disp))
        im.set_extent(cropped_extent(xs, ys))

        if mode != "linear":
            im.set_clim(float(cp.min(I_disp)), float(cp.max(I_disp)))

        ttl.set_text(f"I(x,y) at z = {target_k*ctx.dz:.1f} µm  [{mode}]")
        return (im, ttl)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(fps=fps)))
    return ani

def gpu_status(min_array_gb=0.2, top_n=10):
    import cupy as cp
    import subprocess, textwrap

    print("========== GPU STATUS ==========")

    # --- Driver-level memory ---
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"[Driver]   Free: {free/1e9:6.2f} GB   Total: {total/1e9:6.2f} GB")

    # --- CuPy memory pool ---
    mp = cp.get_default_memory_pool()
    print(f"[CuPy]     Used: {mp.used_bytes()/1e9:6.2f} GB   Cached: {mp.total_bytes()/1e9:6.2f} GB")

    # --- Live arrays in globals() ---
    rows = []
    total_live = 0
    for name, obj in globals().items():
        try:
            if isinstance(obj, cp.ndarray):
                gb = obj.nbytes / 1e9
                total_live += gb
                if gb >= min_array_gb:
                    rows.append((name, obj.shape, str(obj.dtype), gb))
        except Exception:
            pass

    rows.sort(key=lambda x: x[3], reverse=True)

    print(f"[Live CuPy arrays] total: {total_live:6.2f} GB")
    if rows:
        print(f"  Top {min(len(rows), top_n)} arrays ≥ {min_array_gb} GB:")
        for name, shape, dtype, gb in rows[:top_n]:
            print(f"   - {name:20s} {str(shape):18s} {dtype:10s} {gb:6.2f} GB")
    else:
        print(f"  (no arrays ≥ {min_array_gb} GB)")

    # --- nvidia-smi snapshot ---
    print("\n[nvidia-smi]")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]
        ).decode()
        print(textwrap.indent(out.strip(), "  "))
    except Exception as e:
        print("  (nvidia-smi unavailable)", e)

    print("================================")

def gpu_cleanup(names):
    import cupy as cp
    for n in names:
        if n in globals():
            del globals()[n]
            print(f"deleted {n}")
    cp.get_default_memory_pool().free_all_blocks()
    print("CuPy pool cleared.")

def plot_residual_summary_upstream(ctx, res_k, res_yz=None, *, title_prefix="Residual diagnostics"):
    z_um = ctx.dz * np.arange(len(res_k))
    res_k_cpu = cp.asnumpy(res_k)
    zfrac_denom=2
    plt.figure(figsize=(7, 4))
    plt.plot(z_um[:ctx.Nz//zfrac_denom], res_k_cpu[:ctx.Nz//zfrac_denom])
    plt.xlabel("z (µm)")
    plt.ylabel("slice RMS residual")
    plt.title(f"{title_prefix}: residual vs z")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if res_yz is not None:
        Ryz = cp.asnumpy(res_yz[:ctx.Nz//zfrac_denom,:].T)
        y0 = -0.5 * ctx.Ny * ctx.dy
        y1 = y0 + ctx.dy * (ctx.Ny - 1)
        z1 = ctx.dz * (ctx.Nz - 1)

        plt.figure(figsize=(7, 4))
        plt.imshow(
            Ryz,
            origin="lower",
            aspect="auto",
            extent=[0.0, z1/zfrac_denom, y0, y1],
        )
        plt.colorbar(label="RMS residual over x")
        plt.xlabel("z (µm)")
        plt.ylabel("y (µm)")
        plt.title(f"{title_prefix}: yz residual map")
        plt.tight_layout()
        plt.show()

def show_yz_movie(
    arr3d, ctx, *,
    title="yz movie",
    fps=8,
    interval_ms=100,
    robust_pct=99.5,
    symmetric=False,
    log_scale=False,
    repeat=False,
    stride=1,
    max_frames=None,
):
    """
    arr3d shape: (nframes, Nz, Ny)
    Displays y vertically, z horizontally.

    Parameters
    ----------
    stride : int
        Use every `stride`-th stored frame.
    max_frames : int or None
        Cap the displayed movie length. If needed, stride is increased automatically.
    """
    if arr3d is None:
        print(f"{title}: array is None")
        return None

    last = _last_filled_frame(arr3d)
    if last is None:
        print(f"{title}: no filled frames found")
        return None

    nsrc = last + 1
    idx = _frame_indices(nsrc, stride=stride, max_frames=max_frames)
    A = cp.asnumpy(arr3d[idx]).astype(np.float32, copy=False)   # (Nt, Nz, Ny)
    Aplot = np.transpose(A, (0, 2, 1))                          # (Nt, Ny, Nz)

    if log_scale:
        Aplot = np.log10(np.maximum(Aplot, 1e-8))

    if symmetric:
        vmax = np.percentile(np.abs(Aplot), robust_pct)
        vmax = max(float(vmax), 1e-8)
        vmin = 0 #-vmax
    else:
        vmin = float(np.percentile(Aplot, 100 - robust_pct))
        vmax = float(np.percentile(Aplot, robust_pct))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    z0 = 0.0
    z1 = ctx.dz * (ctx.Nz - 1)
    y0 = -0.5 * ctx.Ny * ctx.dy
    y1 = y0 + ctx.dy * (ctx.Ny - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        Aplot[0],
        origin="lower",
        aspect="auto",
        extent=[z0, z1, y0, y1],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("z (µm)")
    ax.set_ylabel("y (µm)")
    ttl = ax.set_title(f"{title}  frame {idx[0]}/{last}")
    plt.colorbar(im, ax=ax)

    def update(i):
        im.set_data(Aplot[i])
        ttl.set_text(f"{title}  frame {idx[i]}/{last}")
        return (im, ttl)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(idx),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(fps=fps)))
    return ani

def show_xz_movie(
    arr3d, ctx, *,
    title="xz movie",
    fps=8,
    interval_ms=100,
    robust_pct=99.5,
    symmetric=False,
    log_scale=False,
    repeat=False,
    stride=1,
    max_frames=None,
):
    """
    arr3d shape: (nframes, Nz, Nx)
    Displays x vertically, z horizontally.

    Parameters
    ----------
    stride : int
        Use every `stride`-th stored frame.
    max_frames : int or None
        Cap the displayed movie length. If needed, stride is increased automatically.
    """
    if arr3d is None:
        print(f"{title}: array is None")
        return None

    last = _last_filled_frame(arr3d)
    if last is None:
        print(f"{title}: no filled frames found")
        return None

    nsrc = last + 1
    idx = _frame_indices(nsrc, stride=stride, max_frames=max_frames)
    A = cp.asnumpy(arr3d[idx]).astype(np.float32, copy=False)   # (Nt, Nz, Nx)
    Aplot = np.transpose(A, (0, 2, 1))                          # (Nt, Nx, Nz)

    if log_scale:
        Aplot = np.log10(np.maximum(Aplot, 1e-8))

    if symmetric:
        vmax = np.percentile(np.abs(Aplot), robust_pct)
        vmax = max(float(vmax), 1e-8)
        vmin = 0 #-vmax
    else:
        vmin = float(np.percentile(Aplot, 100 - robust_pct))
        vmax = float(np.percentile(Aplot, robust_pct))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    z0 = 0.0
    z1 = ctx.dz * (ctx.Nz - 1)
    x0 = -0.5 * ctx.Nx * ctx.dx
    x1 = x0 + ctx.dx * (ctx.Nx - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        Aplot[0],
        origin="lower",
        aspect="auto",
        extent=[z0, z1, x0, x1],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("z (µm)")
    ax.set_ylabel("x (µm)")
    ttl = ax.set_title(f"{title}  frame {idx[0]}/{last}")
    plt.colorbar(im, ax=ax)

    def update(i):
        im.set_data(Aplot[i])
        ttl.set_text(f"{title}  frame {idx[i]}/{last}")
        return (im, ttl)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(idx),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(fps=fps)))
    return ani

# @title moviewriters

def save_movie_streaming(
    arr3d, ctx, out_path, *,
    plane="yz",                 # "yz" or "xz"
    title="movie",
    fps=8,
    dpi=100,
    robust_pct=99.5,
    symmetric=False,
    log_scale=False,
    stride=1,
    max_frames=None,
    cmap=None,
    delete_source=False,
    verbose=True,
):
    """
    Stream a movie directly to MP4 without materializing all frames on CPU.

    arr3d shape:
      plane="yz": (nframes, Nz, Ny)
      plane="xz": (nframes, Nz, Nx)
    """
    if arr3d is None:
        raise ValueError(f"{title}: arr3d is None")

    last = _last_filled_frame(arr3d)
    if last is None:
        raise ValueError(f"{title}: no filled frames found")

    idx = _frame_indices(last + 1, stride=stride, max_frames=max_frames)
    if len(idx) == 0:
        raise ValueError(f"{title}: no frames selected")

    # --- robust limits from a small sample of frames ---
    sample_count = min(12, len(idx))
    sample_idx = np.linspace(0, len(idx) - 1, sample_count, dtype=int)
    sample_frames = []

    for j in sample_idx:
        k = int(idx[j])
        fr = arr3d[k]
        fr = cp.asnumpy(fr) if isinstance(fr, cp.ndarray) else np.asarray(fr)
        fr = fr.astype(np.float32, copy=False)
        if log_scale:
            fr = np.log10(np.maximum(fr, 1e-8))
        sample_frames.append(fr)

    sample = np.stack(sample_frames, axis=0)

    if symmetric:
        vmax = np.percentile(np.abs(sample), robust_pct)
        vmax = max(float(vmax), 1e-8)
        vmin = -vmax
    else:
        vmin = float(np.percentile(sample, 100 - robust_pct))
        vmax = float(np.percentile(sample, robust_pct))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    del sample_frames, sample
    gc.collect()

    # --- geometry ---
    z0 = 0.0
    z1 = ctx.dz * (ctx.Nz - 1)

    if plane == "yz":
        y0 = -0.5 * ctx.Ny * ctx.dy
        y1 = y0 + ctx.dy * (ctx.Ny - 1)
        ylabel = "y (µm)"
        extent = [z0, z1, y0, y1]
    elif plane == "xz":
        x0 = -0.5 * ctx.Nx * ctx.dx
        x1 = x0 + ctx.dx * (ctx.Nx - 1)
        ylabel = "x (µm)"
        extent = [z0, z1, x0, x1]
    else:
        raise ValueError("plane must be 'yz' or 'xz'")

    # first displayed frame
    k0 = int(idx[0])
    frame0 = arr3d[k0]
    frame0 = cp.asnumpy(frame0) if isinstance(frame0, cp.ndarray) else np.asarray(frame0)
    frame0 = frame0.astype(np.float32, copy=False)
    if log_scale:
        frame0 = np.log10(np.maximum(frame0, 1e-8))
    frame0_plot = frame0.T

    # --- writer setup ---
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        frame0_plot,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("z (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  frame {k0}/{last}")
    plt.colorbar(im, ax=ax)

    writer = animation.FFMpegWriter(fps=fps)

    if verbose:
        print(f"[movie] saving {out_path}")
        print(f"[movie] selected {len(idx)} frames from {last + 1} stored frames")

    with writer.saving(fig, str(out_path), dpi=dpi):
        for n, k in enumerate(idx):
            fr = arr3d[int(k)]
            fr = cp.asnumpy(fr) if isinstance(fr, cp.ndarray) else np.asarray(fr)
            fr = fr.astype(np.float32, copy=False)

            if log_scale:
                fr = np.log10(np.maximum(fr, 1e-8))

            im.set_data(fr.T)
            ax.set_title(f"{title}  frame {int(k)}/{last}")
            writer.grab_frame()

            del fr

            if isinstance(arr3d, cp.ndarray) and (n % 10 == 0):
                cp.get_default_pinned_memory_pool().free_all_blocks()

    plt.close(fig)
    gc.collect()

    if isinstance(arr3d, cp.ndarray):
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if delete_source:
        name = None
        for nm, obj in list(globals().items()):
            if obj is arr3d:
                name = nm
                break
        try:
            del arr3d
        except Exception:
            pass
        gc.collect()
        if isinstance(arr3d, cp.ndarray):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        if verbose:
            print("[movie] source array deleted" + (f" ({name})" if name else ""))

    if verbose:
        print(f"[movie] done: {out_path}")

# @title streaming movie writer for I_mid_store_td as I(x,y) along z

def save_Ixy_movie_streaming_from_I_mid_store(
    I_mid_store_td, ctx, out_path, *,
    title="I(x,y) from I_mid_store_td",
    fps=8,
    dpi=100,
    robust_pct=99.5,
    log_scale=False,
    gamma=None,              # e.g. 0.4, or None
    stride=1,
    max_frames=None,
    cmap=None,
    crop_y_um=None,          # e.g. 120.0
    crop_x_um=None,          # e.g. 40.0
    center_mode="peak",      # "peak" or "centroid"
    lock_window=True,
    delete_source=False,
    verbose=True,
):
    """
    Stream I(x,y) movie directly to MP4 from I_mid_store_td of shape (Nz, Nx, Ny).

    Cropping is around the beam core, either locked from the first shown frame
    or updated frame-by-frame.
    """
    arr3d = I_mid_store_td
    if arr3d is None:
        raise ValueError("I_mid_store_td is None")
    if arr3d.ndim != 3:
        raise ValueError(f"Expected shape (Nz, Nx, Ny), got {arr3d.shape}")

    is_cupy = isinstance(arr3d, cp.ndarray)
    Nz, Nx, Ny = arr3d.shape

    def _frame_indices(nframes, stride=1, max_frames=None):
        if nframes <= 0:
            return np.array([], dtype=int)

        stride = max(1, int(stride))
        if max_frames is not None:
            max_frames = max(1, int(max_frames))
            needed_stride = int(np.ceil(nframes / max_frames))
            stride = max(stride, needed_stride)

        idx = np.arange(0, nframes, stride, dtype=int)
        if idx[-1] != nframes - 1:
            idx = np.append(idx, nframes - 1)

        if max_frames is not None and len(idx) > max_frames:
            idx = np.linspace(0, nframes - 1, max_frames, dtype=int)
            idx = np.unique(idx)
            if idx[-1] != nframes - 1:
                idx[-1] = nframes - 1
        return idx

    def _to_np(fr):
        if is_cupy:
            fr = cp.asnumpy(fr)
        else:
            fr = np.asarray(fr)
        return fr.astype(np.float32, copy=False)

    def _transform_np(fr):
        if log_scale:
            fr = np.log10(np.maximum(fr, 1e-8))
        if gamma is not None:
            fmax = max(float(fr.max()), 1e-12)
            fr = np.power(np.maximum(fr, 0.0) / fmax, gamma)
        return fr

    x_coords = (np.arange(Nx) - Nx / 2) * ctx.dx
    y_coords = (np.arange(Ny) - Ny / 2) * ctx.dy

    def _find_center_indices(fr):
        if center_mode == "centroid":
            I = fr.astype(np.float32, copy=False)
            s = float(I.sum())
            s = max(s, 1e-20)
            xi = np.arange(Nx, dtype=np.float32)[:, None]
            yi = np.arange(Ny, dtype=np.float32)[None, :]
            xc = int(np.round((I * xi).sum() / s))
            yc = int(np.round((I * yi).sum() / s))
            return max(0, min(Nx - 1, xc)), max(0, min(Ny - 1, yc))

        flat_idx = int(np.argmax(fr))
        xc, yc = np.unravel_index(flat_idx, (Nx, Ny))
        return int(xc), int(yc)

    def _make_crop_slices(xc, yc):
        if crop_y_um is None:
            ylo, yhi = 0, Ny
        else:
            hy = max(1, int(round(0.5 * crop_y_um / ctx.dy)))
            ylo = max(0, yc - hy)
            yhi = min(Ny, yc + hy + 1)

        if crop_x_um is None:
            xlo, xhi = 0, Nx
        else:
            hx = max(1, int(round(0.5 * crop_x_um / ctx.dx)))
            xlo = max(0, xc - hx)
            xhi = min(Nx, xc + hx + 1)

        return slice(xlo, xhi), slice(ylo, yhi)

    def _extent(xs, ys):
        xlo = x_coords[xs.start]
        xhi = x_coords[xs.stop - 1]
        ylo = y_coords[ys.start]
        yhi = y_coords[ys.stop - 1]
        return [ylo, yhi, xlo, xhi]

    idx = _frame_indices(Nz, stride=stride, max_frames=max_frames)
    if len(idx) == 0:
        raise ValueError("No frames selected")

    # fixed crop window from first displayed frame
    frame0_raw = _to_np(arr3d[int(idx[0])])
    xc0, yc0 = _find_center_indices(frame0_raw)
    xs_fixed, ys_fixed = _make_crop_slices(xc0, yc0)

    # robust limits from sampled frames
    sample_count = min(12, len(idx))
    sample_idx = np.linspace(0, len(idx) - 1, sample_count, dtype=int)
    sample_frames = []

    for j in sample_idx:
        k = int(idx[j])
        fr = _to_np(arr3d[k])

        if lock_window:
            xs, ys = xs_fixed, ys_fixed
        else:
            xc, yc = _find_center_indices(fr)
            xs, ys = _make_crop_slices(xc, yc)

        fr = _transform_np(fr[xs, ys])
        sample_frames.append(fr)

    sample = np.stack(sample_frames, axis=0)
    vmin = float(np.percentile(sample, 100 - robust_pct))
    vmax = float(np.percentile(sample, robust_pct))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    del sample_frames, sample
    gc.collect()

    # first displayed frame
    if lock_window:
        xs, ys = xs_fixed, ys_fixed
    else:
        xc, yc = _find_center_indices(frame0_raw)
        xs, ys = _make_crop_slices(xc, yc)

    frame0 = _transform_np(frame0_raw[xs, ys])

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    im = ax.imshow(
        frame0,
        origin="lower",
        aspect="auto",
        extent=_extent(xs, ys),

        cmap=cmap,
    )
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("x (µm)")
    ax.set_title(f"{title}  z = {int(idx[0]) * ctx.dz:.1f} µm")
    plt.colorbar(im, ax=ax)

    writer = animation.FFMpegWriter(fps=fps)

    if verbose:
        print(f"[movie] saving {out_path}")
        print(f"[movie] selected {len(idx)} frames from {Nz} stored slices")

    with writer.saving(fig, str(out_path), dpi=dpi):
        for n, k in enumerate(idx):
            fr_raw = _to_np(arr3d[int(k)])

            if lock_window:
                xs, ys = xs_fixed, ys_fixed
            else:
                xc, yc = _find_center_indices(fr_raw)
                xs, ys = _make_crop_slices(xc, yc)

            fr = _transform_np(fr_raw[xs, ys])

            im.set_data(fr)
            im.set_extent(_extent(xs, ys))
            ax.set_title(f"{title}  z = {int(k) * ctx.dz:.1f} µm")
            writer.grab_frame()

            del fr_raw, fr

            if is_cupy and (n % 10 == 0):
                cp.get_default_pinned_memory_pool().free_all_blocks()

    plt.close(fig)
    gc.collect()

    if is_cupy:
        cp.get_default_pinned_memory_pool().free_all_blocks()

    if delete_source:
        try:
            del arr3d
        except Exception:
            pass
        gc.collect()
        if is_cupy:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        if verbose:
            print("[movie] source array deleted")

    if verbose:
        print(f"[movie] done: {out_path}")

