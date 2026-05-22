# lc_core/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional, Any
import math


# =========================================================
# CPU/JSON-safe configuration objects
# No CuPy imports in this file.
# =========================================================

@dataclass
class GridConfig:
    lm_um: float = 0.633

    Nx: int = 512
    Ny: int = 512
    Nz: int = 200

    xaper_um: float = 75.0
    yaper_um: float = 1000.0
    rlen_um: float = 4000.0

    dx_um: float = 75.0 / 512
    dy_um: float = 1000.0 / 512
    dz_um: float = 20.0

    du: float = 2.0 / (512 - 1)
    dv: float = 2.0 * (1000.0 / 75.0) / 512


@dataclass
class MaterialConfig:
    d_um: float = 75.0
    K_SI: float = 7e-12
    De_rel: float = 13.0
    bias_voltage: float = 4.0

    ne: float = 1.7
    no: float = 1.5

    P_mW: float = 1.0
    P_W: float = 1e-3

    b: float = 0.0
    bi: float = 0.0

    mobility: float = 1.0
    theta_bc: float = 0.0
    theta_z_gamma: float = 0.0
    theta_z_stride_um: float = 0.0


@dataclass
class LaunchConfig:
    # Old names preserved deliberately.
    thout1: float = 0.1
    thout2: float = -0.1
    phi1: float = 0.0
    phi2: float = 0.0

    w0x1_um: float = 3.0
    w0y1_um: float = 5.0
    
    w0x2_um: float = 3.0
    w0y2_um: float = 5.0

    xoffset_um: float = 0.0
    yoffset_um: float = 0.0
    soliton_pair_sep_um: float = 0.0
    soliton_pair_angle_deg: float = 0.0

    coh: bool = True
    soliton: bool = False

    # Image-mask support retained but disabled by default.
    image_on_beam: str = "No Image"
    external_image: str = ""
    std_image: str = "MNIST 0"
    std_image_dir: str = ""
    image_invert: bool = False
    image_size_factor: float = 1.0


@dataclass
class BoundaryConfig:
    # This is the proven sponge/window knob from the old runner.
    use_sponge: bool = True
    windowedge: float = 0.1


@dataclass
class TimeConfig:
    timedep: bool = False
    tend: float = 10.0
    tsteps: int = 1
    use_cons_tsteps: bool = False
    t_stride: int = 2
    dt: float = 10.0


@dataclass
class StaticSolverConfig:
    dtau_static: float = 0.01
    static_tol_resid: float = 5e-3
    static_resid_every: int = 25
    static_max_steps: int = 2000
    static_relax_omega: float = 0.3

    picard_iters: int = 4
    picard_tol_up: float = 1e-6


@dataclass
class TDSolverConfig:
    td_theta_relax_steps: int = 1
    td_theta_relax_omega: float = 1.0


@dataclass
class SaveConfig:
    store_slice_movies: bool = True
    save_full_I_mid_store: bool = False
    base_dir: str = "runs"
    run_name: str = "lc_run"


@dataclass
class RunConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    static: StaticSolverConfig = field(default_factory=StaticSolverConfig)
    td: TDSolverConfig = field(default_factory=TDSolverConfig)
    save: SaveConfig = field(default_factory=SaveConfig)

    # Keep a copy of original legacy knobs for reproducibility/debugging.
    legacy_prdata: dict[str, Any] = field(default_factory=dict)


def config_to_dict(cfg: RunConfig) -> dict:
    return asdict(cfg)


def derive_lc_constants(cfg: RunConfig) -> RunConfig:
    """
    Fill derived geometry and LC coefficients using the validated old formulas.

    This mutates and returns cfg intentionally, so callers can do:
        cfg = derive_lc_constants(cfg)
    """
    eps0 = 8.854e-12
    c0 = 3e8

    g = cfg.grid
    m = cfg.material

    g.xaper_um = float(m.d_um)
    g.Nz = max(1, int(round(float(g.rlen_um) / float(g.dz_um))))

    g.dx_um = float(g.xaper_um) / int(g.Nx)
    g.dy_um = float(g.yaper_um) / int(g.Ny)

    g.du = 2.0 / (int(g.Nx) - 1)
    g.dv = (2.0 * (float(g.yaper_um) / float(m.d_um))) / int(g.Ny)

    m.P_W = float(m.P_mW) * 1e-3

    d_m = float(m.d_um) * 1e-6
    De_SI = float(m.De_rel) * eps0
    E_SI = float(m.bias_voltage) / d_m

    m.b = float((De_SI * (E_SI**2) * (d_m**2)) / (8.0 * float(m.K_SI)))

    na2 = float(m.ne)**2 - float(m.no)**2
    m.bi = float((na2 * (d_m**2) * 1e12 * float(m.P_W)) / (8.0 * c0 * float(m.K_SI)))

    # Time-step logic preserved from old initializer.
    t = cfg.time
    t.tsteps = int(t.tsteps) if t.timedep else 1
    t.dt = float(t.tend) / max(int(t.tsteps), 1)

    if t.use_cons_tsteps and t.timedep:
        t.dt = t.dt / 2.0
        t.tsteps = int(math.ceil(float(t.tend) / float(t.dt)))
        t.dt = float(t.tend) / int(t.tsteps)

    return cfg


def config_from_prdata(prdata: dict[str, Any]) -> RunConfig:
    """
    Convert legacy prdata-style dictionary into a structured, CPU-safe RunConfig.

    This function intentionally preserves old parameter names and defaults, so the
    production runner can be checked directly against old notebook behavior.
    """
    cfg = RunConfig()
    cfg.legacy_prdata = dict(prdata)

    g = cfg.grid
    g.lm_um = float(prdata.get("lm", 0.633))
    g.Nx = int(prdata.get("xsamp", 512))
    g.Ny = int(prdata.get("ysamp", 512))
    g.yaper_um = float(prdata.get("yaper", 1000.0))
    g.rlen_um = float(prdata.get("rlen", 4000.0))
    g.dz_um = float(prdata.get("dz", 20.0))

    mat = cfg.material
    mat.d_um = float(prdata.get("d", 75.0))
    mat.K_SI = float(prdata.get("K", 7e-12))
    mat.De_rel = float(prdata.get("De", 13.0))
    mat.bias_voltage = float(prdata.get("bias_voltage", 4.0))
    mat.ne = float(prdata.get("ne", 1.7))
    mat.no = float(prdata.get("no", 1.5))
    mat.P_mW = float(prdata.get("P", 1.0))
    mat.mobility = float(prdata.get("mobility", 1.0))
    mat.theta_bc = float(prdata.get("theta_bc", 0.0))
    mat.theta_z_gamma = float(prdata.get("theta_z_gamma", 0.0))
    mat.theta_z_stride_um = float(prdata.get("theta_z_stride_um", 0.0))

    launch = cfg.launch
    launch.thout1 = float(prdata.get("thout1", 0.1))
    launch.thout2 = float(prdata.get("thout2", -0.1))
    launch.phi1 = float(prdata.get("phi1", 0.0))
    launch.phi2 = float(prdata.get("phi2", 0.0))
    launch.w0x1_um = float(prdata.get("w0x1", prdata.get("w01x", 2.0)))
    launch.w0y1_um = float(prdata.get("w0y1", prdata.get("w01y", 8.0)))
    
    launch.w0x2_um = float(prdata.get("w0x2", prdata.get("w02x", launch.w0x1_um)))
    launch.w0y2_um = float(prdata.get("w0y2", prdata.get("w02y", launch.w0y1_um)))

    
    launch.xoffset_um = float(prdata.get("xoffset", 0.0))
    launch.yoffset_um = float(prdata.get("yoffset", 0.0))
    launch.soliton_pair_sep_um = float(prdata.get("soliton_pair_sep", 0.0))
    launch.soliton_pair_angle_deg = float(prdata.get("soliton_pair_angle", 0.0))
    launch.coh = bool(prdata.get("coh", True))
    launch.soliton = bool(prdata.get("soliton", False))

    launch.image_on_beam = str(prdata.get("image_on_beam", "No Image"))
    launch.external_image = str(prdata.get("external_image", "")).strip()
    launch.std_image = str(prdata.get("std_image", "MNIST 0"))
    launch.std_image_dir = str(prdata.get("std_image_dir", ""))
    launch.image_invert = bool(prdata.get("image_invert", False))
    launch.image_size_factor = float(prdata.get("image_size_factor", 1.0))

    cfg.boundary.windowedge = float(prdata.get("windowedge", 0.1))
    cfg.boundary.use_sponge = bool(prdata.get("use_sponge", True))

    t = cfg.time
    t.timedep = (prdata.get("time_behavior", "Static") == "Time Dependent")
    t.tend = float(prdata.get("tend", 10.0))
    t.tsteps = int(prdata.get("tsteps", 120)) if t.timedep else 1
    t.use_cons_tsteps = bool(prdata.get("use_cons_tsteps", False))
    t.t_stride = int(prdata.get("t_stride", 2))

    s = cfg.static
    s.dtau_static = float(prdata.get("dtau_static", 0.01))
    s.static_tol_resid = float(prdata.get("static_tol_resid", 5e-3))
    s.static_resid_every = int(prdata.get("static_resid_every", 25))
    s.static_max_steps = int(prdata.get("static_max_steps", 2000))
    s.static_relax_omega = float(prdata.get("static_relax_omega", 0.3))
    s.picard_iters = int(prdata.get("picard_iters", 4))
    s.picard_tol_up = float(prdata.get("picard_tol_up", 1e-6))

    td = cfg.td
    td.td_theta_relax_steps = int(prdata.get("td_theta_relax_steps", 1))
    td.td_theta_relax_omega = float(prdata.get("td_theta_relax_omega", 1.0))

    cfg.save.store_slice_movies = bool(prdata.get("store_slice_movies", True))
    cfg.save.save_full_I_mid_store = bool(prdata.get("save_full_I_mid_store", False))

    return derive_lc_constants(cfg)


def validate_config(cfg: RunConfig) -> None:
    g = cfg.grid
    m = cfg.material

    assert g.Nx > 2 and g.Ny > 2 and g.Nz > 0
    assert g.dx_um > 0 and g.dy_um > 0 and g.dz_um > 0
    assert g.du > 0 and g.dv > 0
    assert g.lm_um > 0

    assert m.d_um > 0
    assert m.K_SI > 0
    assert m.ne > 0 and m.no > 0
    assert m.mobility > 0
    assert m.P_mW >= 0

    assert cfg.boundary.windowedge >= 0.0
    assert cfg.time.tsteps >= 1
    assert cfg.time.dt > 0.0
    assert cfg.time.t_stride >= 1


def print_config_summary(cfg: RunConfig) -> None:
    g = cfg.grid
    m = cfg.material
    l = cfg.launch
    t = cfg.time

    print("LC RunConfig summary")
    print("--------------------")
    print(f"Grid:    Nx={g.Nx}, Ny={g.Ny}, Nz={g.Nz}")
    print(f"Spacing: dx={g.dx_um:.6g} um, dy={g.dy_um:.6g} um, dz={g.dz_um:.6g} um")
    print(f"Aperture: x={g.xaper_um:.6g} um, y={g.yaper_um:.6g} um, z={g.rlen_um:.6g} um")
    print(f"LC:      b={m.b:.6g}, bi={m.bi:.6g}, ne={m.ne}, no={m.no}")
    print(f"Bias:    V={m.bias_voltage:.6g}, theta_bc={m.theta_bc:.6g}")
    print(f"Launch:  coh={l.coh}, thout=({l.thout1}, {l.thout2}), sep={l.soliton_pair_sep_um} um")
    print(f"Time:    timedep={t.timedep}, tsteps={t.tsteps}, dt={t.dt:.6g}")
    print(f"Sponge:  enabled={cfg.boundary.use_sponge}, windowedge={cfg.boundary.windowedge}")

import json
from pathlib import Path


def load_prdata_json(path):
    path = Path(path)
    return json.loads(path.read_text())


