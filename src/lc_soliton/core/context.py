from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math
from .backend import asnumpy, get_backend
from ..physics.lc.bias import compute_neff
from ..validated_core.runner_core import intens_into  # optional only if needed
from .dual_grid import restrict_block_mean, prolong_repeat
from .launch import intensity, make_input_field
from lc_soliton.physics.lc.bias import build_theta_bias_2d_from_params

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
    theta_bias_tol_rms: float = 5e-3
    theta_bias_tol_max: float = 2e-2
    theta_bias_max_iter: int = 2000
    theta_bias_report_every: int = 1000
    theta_bias_verbose: bool = False
    V_bias: float = 1.0
    P_mW: float = 1.0
    K: float = 7e-12
    De: float = 13.0

    # optical input
    power_norm: float = 1.0
    waist_x_um: float = 8.0
    waist_y_um: float = 8.0
    y_sep_um: float = 0.0
    coherent: bool = True
    pair_angle_deg: float = 0.0
    theta_out1_deg: float = 0.0
    theta_out2_deg: float = 0.0
    phi1_deg: float = 0.0
    phi2_deg: float = 0.0
    power_ratio: float = 0.0

    # stepping controls
    dz_opt_max_phi: float = 0.30
    dn_max_est: float = 0.02
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


@dataclass
class LegacyPlans:
    Ahat: Any = None
    plan_f: Any = None
    plan_i: Any = None
    sS: Any = None
    offS: Any = None
    diagS: Any = None
    lamS: Any = None

def make_context(params: LCParams) -> Tuple[LCContext, Optional[DualGrid]]:
    xp, _ = get_backend(params.backend)
    Nx, Ny, Nz = int(params.Nx), int(params.Ny), int(params.Nz)
    dx = float(params.xaper_um) / Nx
    dy = float(params.yaper_um) / Ny
    dz = float(params.dz_um)

    x = (xp.arange(Nx, dtype=xp.float32) - Nx / 2) * dx + 0.5 * dx
    y = (xp.arange(Ny, dtype=xp.float32) - Ny / 2) * dy + 0.5 * dy

    theta_bias_2d = build_theta_bias_2d_from_params(params, xp)
    refin = float(
        asnumpy(
            compute_neff(theta_bias_2d, ne=params.ne, no=params.no, xp=xp)
        ).mean()
    )
    
    kout = float(2.0 * math.pi / params.wavelength_um)

    fx = xp.fft.fftfreq(Nx, d=dx).astype(xp.float32)
    fy = xp.fft.fftfreq(Ny, d=dy).astype(xp.float32)
    fxy2 = (fx[:, None] ** 2 + fy[None, :] ** 2).astype(xp.float32)

    ctx = LCContext(
        p=params,
        xp=xp,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=dx,
        dy=dy,
        dz=dz,
        du_lc=2.0 / max(1, Nx - 1),
        dv_lc=(2.0 / max(1, Nx - 1)) * (dy / dx),
        x=x,
        y=y,
        fxy2=fxy2,
        amp0=make_input_field(params, x, y, xp),
        theta_bias_2d=theta_bias_2d,
        refin=refin,
        kout=kout,
        windowxy=xp.ones((Nx, Ny), dtype=xp.float32),
        theta_clamp=(params.theta_clamp_min, params.theta_clamp_max),
        _h_cache={},
    )

    dg = None
    if params.use_dual_grid:
        f = int(params.dual_grid_factor)
        if Nx % f or Ny % f:
            raise ValueError("Nx and Ny must be divisible by dual_grid_factor")
        dg = DualGrid(factor=f, Nx_c=Nx // f, Ny_c=Ny // f, theta_bias_c=restrict_block_mean(theta_bias_2d, f, xp=xp))

    print("theta_bias_2d min/max:", float(xp.min(theta_bias_2d)), float(xp.max(theta_bias_2d)))
    print("theta_bias_amp:", params.theta_bias_amp)
    print("theta_bc:", params.theta_bc)
    return ctx, dg


def apply_legacy_context_aliases(ctx: LCContext, params: LCParams) -> None:
    # Material / optical names expected by validated_core.runner_core.
    ctx.lm = params.wavelength_um
    ctx.ne = params.ne
    ctx.no = params.no
    ctx.b = params.b
    ctx.bi = params.bi
    ctx.mobility = params.mobility
    ctx.theta_bc = params.theta_bc
    ctx.theta_z_gamma = params.theta_z_gamma
    ctx.theta_z_stride_um = 0.0
    ctx.coh = params.coherent

    # Grid aliases.
    ctx.du = ctx.du_lc
    ctx.dv = ctx.dv_lc

    # Solver controls.
    ctx.dtau_static = params.dtau_static
    ctx.strict_selfcons_passes = params.static_selfcons_passes
    ctx.strict_selfcons_tol_theta = 1e-4
    ctx.strict_selfcons_tol_I = 1e-4
    ctx.picard_iters = 4
    ctx.picard_tol_up = 1e-6
    ctx.true_td_predictor_only = False
    ctx.true_td_full_pred_optics = False
    ctx.linearized_td = False
    ctx.runtime_every_k = 0
    ctx.freeze_theta = False
    ctx.save_full_I_mid_store = False

    # Optics controls.
    ctx.dn_max_est = params.dn_max_est
    ctx.dz_opt_max_phi = params.dz_opt_max_phi
    ctx.max_substeps = params.max_substeps

    # Runtime storage expected by some kernels.
    ctx.use_dual_grid = params.use_dual_grid
    ctx.theta_prev_time = None
    ctx.theta_full = None
    ctx._cn_cache = {}
    ctx._amp_in_buf = None
    ctx._amp_pred_buf = None
    ctx._I_mid_pred_buf = None



__all__ = [
    "LCParams",
    "LCContext",
    "DualGrid",
    "LegacyPlans",
    "intensity",
    "make_input_field",
    "restrict_block_mean",
    "prolong_repeat",
    "make_context",
    "apply_legacy_context_aliases",
]