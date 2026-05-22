# lc_core/gpu_context.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import math
import numpy as np


@dataclass
class LCContextGPU:
    """
    GPU context corresponding to the old LCContext, but constructed from RunConfig.

    This class deliberately stores xp=cupy and does not import cupy until
    build_gpu_context() is called.
    """
    xp: Any

    Nx: int
    Ny: int
    Nz: int

    dx: float
    dy: float
    dz: float
    lm: float

    x: Any
    y: Any
    z: Any

    fxy2: Any
    windowxy: Any

    kout: float
    ne: float
    no: float
    refin: float
    coh: bool

    b: float
    bi: float
    du: float
    dv: float

    amp0: Any
    theta_bias_2d: Any
    theta_full: Any
    theta_prev_time: Any

    mobility: float
    theta_bc: float
    theta_z_gamma: float
    theta_z_stride_um: float

    dtau_static: float
    static_tol_resid: float
    static_resid_every: int
    static_max_steps: int
    static_relax_omega: float

    picard_iters: int
    picard_tol_up: float

    td_theta_relax_steps: int
    td_theta_relax_omega: float

    save_full_I_mid_store: bool

    h_cache: dict = field(default_factory=dict)


@dataclass
class RuntimeStoresGPU:
    time_steps: Any
    t_stride: int
    timedep: bool

    Ixz: Any = None
    Iyz: Any = None
    thetaxz: Any = None
    thetayz: Any = None
    save_slices: Callable | None = None


def _make_tukey_window(N: int, alpha: float) -> np.ndarray:
    """
    Small local fallback equivalent to scipy.signal.windows.tukey for our use.
    Avoids requiring scipy just to build the sponge/window.
    """
    if alpha <= 0:
        return np.ones(N, dtype=np.float32)
    if alpha >= 1:
        return np.hanning(N).astype(np.float32, copy=False)

    x = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float64)
    w = np.ones(N, dtype=np.float64)

    first = x < alpha / 2.0
    last = x >= (1.0 - alpha / 2.0)

    w[first] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[first] - alpha / 2.0)))
    w[last] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[last] - 1.0 + alpha / 2.0)))

    return w.astype(np.float32, copy=False)


def build_theta_bias_gpu(cfg, cp, *, build_theta_bias_IC, build_theta_bias_IC_dirichlet_value):
    """
    Build theta_bias_2d using old validated routines, injected as callables.
    """
    g = cfg.grid
    m = cfg.material

    theta_bc = float(m.theta_bc)

    if theta_bc == 0.0:
        theta_bias_2d = build_theta_bias_IC(
            int(g.Nx), int(g.Ny), 1, float(m.b),
            kick_eps=0.01,
            eps_clip=1e-12,
            return_2d=True,
        ).astype(cp.float32, copy=False)

        theta_bias_2d[0, :] = 0.0
        theta_bias_2d[-1, :] = 0.0
        return theta_bias_2d

    theta_bias_2d = build_theta_bias_IC_dirichlet_value(
        int(g.Nx),
        int(g.Ny),
        1,
        float(m.b),
        theta_bc=theta_bc,
        du=float(g.du),
        dv=float(g.dv),
        mobility=float(m.mobility),
        max_iter=int(cfg.legacy_prdata.get("theta_bias_max_iter", 50000)),
        tol_rms=float(cfg.legacy_prdata.get("theta_bias_tol_rms", 1e-8)),
        tol_max=float(cfg.legacy_prdata.get("theta_bias_tol_max", 1e-6)),
        report_every=int(cfg.legacy_prdata.get("theta_bias_report_every", 1000)),
        verbose=bool(cfg.legacy_prdata.get("theta_bias_verbose", True)),
        return_2d=True,
        dtype=cp.float32,
    ).astype(cp.float32, copy=False)

    theta_bias_2d[0, :] = cp.float32(theta_bc)
    theta_bias_2d[-1, :] = cp.float32(theta_bc)

    edge0 = float(cp.mean(theta_bias_2d[0, :]).get())
    edge1 = float(cp.mean(theta_bias_2d[-1, :]).get())

    if not (abs(edge0 - theta_bc) < 1e-6 and abs(edge1 - theta_bc) < 1e-6):
        raise RuntimeError(
            f"theta_bias mismatch: edges ({edge0}, {edge1}) vs theta_bc={theta_bc}"
        )

    return theta_bias_2d


def build_gpu_context(
    cfg,
    *,
    restart_theta: bool = True,
    build_theta_bias_IC,
    build_theta_bias_IC_dirichlet_value,
    compute_n_bg_from_bias,
    build_launch_field_gpu,
    genrot,
    build_amp_pair,
    intens,
    fft_module=None,
):
    """
    Build GPU context from a CPU-safe RunConfig.

    Old validated functions are injected:
        build_theta_bias_IC
        build_theta_bias_IC_dirichlet_value
        compute_n_bg_from_bias
        build_launch_field_gpu
        genrot
        build_amp_pair
        intens

    This lets us reuse stable old code without mixing all responsibilities in one
    initializer.
    """
    import cupy as cp

    g = cfg.grid
    m = cfg.material

    Nx, Ny, Nz = int(g.Nx), int(g.Ny), int(g.Nz)
    dx, dy, dz = float(g.dx_um), float(g.dy_um), float(g.dz_um)

    x = (cp.arange(Nx, dtype=cp.float32) - Nx / 2) * dx + cp.float32(0.5 * dx)
    y = (cp.arange(Ny, dtype=cp.float32) - Ny / 2) * dy + cp.float32(0.5 * dy)
    z = cp.arange(Nz, dtype=cp.float32) * cp.float32(dz)

    fx = cp.fft.fftfreq(Nx, d=dx).astype(cp.float32, copy=False)
    fy = cp.fft.fftfreq(Ny, d=dy).astype(cp.float32, copy=False)
    fxy2 = (fx[:, None] ** 2 + fy[None, :] ** 2).astype(cp.float32, copy=False)

    kout = float(2.0 * math.pi / float(g.lm_um))

    theta_bias_2d = build_theta_bias_gpu(
        cfg,
        cp,
        build_theta_bias_IC=build_theta_bias_IC,
        build_theta_bias_IC_dirichlet_value=build_theta_bias_IC_dirichlet_value,
    )

    n_bg = compute_n_bg_from_bias(theta_bias_2d, float(m.ne), float(m.no), cp, reducer="median")
    refin = float(n_bg)

    windowx = _make_tukey_window(Nx, float(cfg.boundary.windowedge))
    windowy = _make_tukey_window(Ny, float(cfg.boundary.windowedge))

    if not cfg.boundary.use_sponge:
        windowxy = cp.ones((Nx, Ny), dtype=cp.float32)
    else:
        windowxy = cp.asarray(np.sqrt(np.outer(windowx, windowy)), dtype=cp.float32)

    theta_full = cp.empty((Nz, Nx, Ny), dtype=cp.float32)
    if restart_theta:
        theta_full[...] = theta_bias_2d[None, :, :]

    theta_prev_time = cp.empty_like(theta_full)

    # Create temporary context with amp0=None, then build launch using ctx.x/y/fxy2/refin.
    ctx = LCContextGPU(
        xp=cp,
        Nx=Nx, Ny=Ny, Nz=Nz,
        dx=dx, dy=dy, dz=dz, lm=float(g.lm_um),
        x=x, y=y, z=z,
        fxy2=fxy2,
        windowxy=windowxy,
        kout=kout,
        ne=float(m.ne),
        no=float(m.no),
        refin=refin,
        coh=bool(cfg.launch.coh),
        b=float(m.b),
        bi=float(m.bi),
        du=float(g.du),
        dv=float(g.dv),
        amp0=None,
        theta_bias_2d=theta_bias_2d,
        theta_full=theta_full,
        theta_prev_time=theta_prev_time,
        mobility=float(m.mobility),
        theta_bc=float(m.theta_bc),
        theta_z_gamma=float(m.theta_z_gamma),
        theta_z_stride_um=float(m.theta_z_stride_um),
        dtau_static=float(cfg.static.dtau_static),
        static_tol_resid=float(cfg.static.static_tol_resid),
        static_resid_every=int(cfg.static.static_resid_every),
        static_max_steps=int(cfg.static.static_max_steps),
        static_relax_omega=float(cfg.static.static_relax_omega),
        picard_iters=int(cfg.static.picard_iters),
        picard_tol_up=float(cfg.static.picard_tol_up),
        td_theta_relax_steps=int(cfg.td.td_theta_relax_steps),
        td_theta_relax_omega=float(cfg.td.td_theta_relax_omega),
        save_full_I_mid_store=bool(cfg.save.save_full_I_mid_store),
    )

    launch_arrays = build_launch_field_gpu(
        cfg,
        ctx,
        genrot=genrot,
        build_amp_pair=build_amp_pair,
        intens=intens,
        fft_module=fft_module,
    )
    ctx.amp0 = launch_arrays.amp0

    stores = build_runtime_stores_gpu(cfg, ctx, intens=intens)

    print_gpu_context_summary(ctx, stores)

    return ctx, stores, launch_arrays


def build_runtime_stores_gpu(cfg, ctx: LCContextGPU, *, intens):
    cp = ctx.xp

    tsteps = int(cfg.time.tsteps)
    dt = float(cfg.time.dt)

    time_steps = (
        cp.full((tsteps,), cp.float32(dt), dtype=cp.float32)
        if cfg.time.timedep
        else cp.full((1,), cp.float32(dt), dtype=cp.float32)
    )

    t_stride = int(cfg.time.t_stride)
    nframes = (len(time_steps) + t_stride - 1) // t_stride

    if bool(cfg.save.store_slice_movies):
        Ixz = cp.zeros((nframes, ctx.Nz, ctx.Nx), dtype=cp.float32)
        Iyz = cp.zeros((nframes, ctx.Nz, ctx.Ny), dtype=cp.float32)
        thetaxz = cp.zeros((nframes, ctx.Nz, ctx.Nx), dtype=cp.float32)
        thetayz = cp.zeros((nframes, ctx.Nz, ctx.Ny), dtype=cp.float32)

        def save_slices(slot, k, Ixy, theta_xy):
            Ixz[slot, k, :] = Ixy[:, ctx.Ny // 2]
            Iyz[slot, k, :] = Ixy[ctx.Nx // 2, :]
            dtheta = (theta_xy - ctx.theta_bias_2d).astype(cp.float32, copy=False)
            thetaxz[slot, k, :] = dtheta[:, ctx.Ny // 2]
            thetayz[slot, k, :] = dtheta[ctx.Nx // 2, :]

    else:
        Ixz = Iyz = thetaxz = thetayz = None
        save_slices = None

    return RuntimeStoresGPU(
        time_steps=time_steps,
        t_stride=t_stride,
        timedep=bool(cfg.time.timedep),
        Ixz=Ixz,
        Iyz=Iyz,
        thetaxz=thetaxz,
        thetayz=thetayz,
        save_slices=save_slices,
    )


def estimate_gpu_bytes(ctx: LCContextGPU, stores: RuntimeStoresGPU | None = None) -> int:
    total = 0
    for name in [
        "x", "y", "z", "fxy2", "windowxy",
        "amp0", "theta_bias_2d", "theta_full", "theta_prev_time",
    ]:
        arr = getattr(ctx, name, None)
        if arr is not None and hasattr(arr, "nbytes"):
            total += int(arr.nbytes)

    if stores is not None:
        for name in ["time_steps", "Ixz", "Iyz", "thetaxz", "thetayz"]:
            arr = getattr(stores, name, None)
            if arr is not None and hasattr(arr, "nbytes"):
                total += int(arr.nbytes)

    return total


def print_gpu_context_summary(ctx: LCContextGPU, stores: RuntimeStoresGPU | None = None) -> None:
    total_gib = estimate_gpu_bytes(ctx, stores) / 1024**3

    print("Initialized GPU context:")
    print(f"  Nx,Ny,Nz = {ctx.Nx}, {ctx.Ny}, {ctx.Nz}")
    print(f"  dx,dy,dz = {ctx.dx:.6g}, {ctx.dy:.6g}, {ctx.dz:.6g} um")
    print(f"  refin    = {ctx.refin:.9g}")
    print(f"  b, bi    = {ctx.b:.9g}, {ctx.bi:.9g}")
    print(f"  coh      = {ctx.coh}")
    print(f"  amp0     = shape {None if ctx.amp0 is None else ctx.amp0.shape}")
    print(f"  theta    = shape {ctx.theta_full.shape}, dtype {ctx.theta_full.dtype}")
    print(f"  approx allocated in ctx/stores = {total_gib:.3f} GiB")

    if stores is not None:
        print(f"  timedep  = {stores.timedep}, tsteps={len(stores.time_steps)}, dt={float(stores.time_steps[0].get()):.6g}")
        print(f"  slice movies = {stores.Ixz is not None}")
