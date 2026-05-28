"""
Translate stored legacy reference-case configs into canonical SimulationRequest objects.
"""

from __future__ import annotations

from pathlib import Path

from .request import (
    SimulationRequest,
    GridRequest,
    GeometryRequest,
    MaterialRequest,
    LaunchRequest,
    BoundaryRequest,
    SolverRequest,
    OutputRequest,
    RuntimeRequest,
)


def reference_config_to_request(
    reference_data: dict,
    run_dir: str | Path,
) -> SimulationRequest:

    cfg = reference_data["config"]

    return SimulationRequest(
        mode="static",

        grid=GridRequest(
            Nx=int(cfg["grid"]["Nx"]),
            Ny=int(cfg["grid"]["Ny"]),
            Nz=int(cfg["grid"]["Nz"]),
        ),

        geometry=GeometryRequest(
            xaper_um=float(cfg["grid"]["xaper_um"]),
            yaper_um=float(cfg["grid"]["yaper_um"]),
            dz_um=float(cfg["grid"]["dz_um"]),
            wavelength_um=float(cfg["grid"].get("lm_um", 0.633)),
        ),

        material=MaterialRequest(
            ne=float(cfg["material"]["ne"]),
            no=float(cfg["material"]["no"]),
            b=float(cfg["material"]["b"]),
            bi=float(cfg["material"]["bi"]),
            mobility=float(cfg["material"].get("mobility", 1.0)),
            theta_bc=float(cfg["material"].get("theta_bc", 0.0)),
            theta_bias_amp=float(cfg["material"].get("theta_bias_amp", 0.1)),
            theta_clamp_min=float(cfg["material"].get("theta_clamp_min", -1.2)),
            theta_clamp_max=float(cfg["material"].get("theta_clamp_max", 1.2)),
            theta_z_gamma=float(cfg["material"].get("theta_z_gamma", 0.0)),
        ),

        launch=LaunchRequest(
            power_mW=float(cfg["material"].get("P_mW", 1.0)),
            waist_x_um=float(cfg["launch"]["w0x1_um"]),
            waist_y_um=float(cfg["launch"]["w0y1_um"]),
            separation_um=float(cfg["launch"].get("soliton_pair_sep_um", 0.0)),
            coherent=bool(cfg["launch"].get("coh", False)),
        ),

        boundary=BoundaryRequest(
            use_sponge=bool(cfg.get("boundary", {}).get("use_sponge", True)),
            windowedge=float(cfg.get("boundary", {}).get("windowedge", 0.1)),
        ),

        solver=SolverRequest(
            static_max_steps=int(cfg["static"].get("static_max_steps", 2000)),
            dtau_static=float(cfg["static"].get("dtau_static", 0.01)),
            static_tol_rms=float(cfg["static"].get("static_tol_resid", 0.005)),
            static_tol_max=float(cfg["static"].get("static_tol_max", 0.01)),
            static_selfcons_passes=int(cfg["static"].get("static_selfcons_passes", 3)),
            static_mix=float(cfg["static"].get("static_relax_omega", 0.3)),
            Nt=int(cfg.get("time", {}).get("tsteps", 1)),
            dt=float(cfg.get("time", {}).get("dt", 0.02)),
            t_stride=int(cfg.get("time", {}).get("t_stride", 1)),
            dz_opt_max_phi=float(cfg.get("grid", {}).get("dz_opt_max_phi", 0.3)),
            dn_max_est=float(cfg.get("grid", {}).get("dn_max_est", 0.02)),
            max_substeps=int(cfg.get("grid", {}).get("max_substeps", 16)),
        ),

        output=OutputRequest(
            run_dir=str(run_dir),
            save_slices=True,
            save_full=False,
        ),

        runtime=RuntimeRequest(
            backend="auto",
            progress=True,
        ),
    )


__all__ = ["reference_config_to_request"]
