from pathlib import Path

from lc_soliton import (
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
from lc_soliton.eigensoliton_runner import run_eigensoliton_existence_curve


run_dir = Path("small_runs/cpu_eigensoliton_sweep")
run_dir.mkdir(parents=True, exist_ok=True)

req = SimulationRequest(
    mode="eigensoliton_sweep",
    grid=GridRequest(Nx=128, Ny=128, Nz=16),
    geometry=GeometryRequest(
        xaper_um=75.0,
        yaper_um=100.0,
        dz_um=5.0,
        wavelength_um=0.633,
    ),
    material=MaterialRequest(
        ne=1.7,
        no=1.5,
        b=1.7185702034057138,
        bi=214.28571428571414,
        mobility=1.0,
        theta_bc=0.0,
        theta_bias_amp=0.1,
        theta_clamp_min=-1.2,
        theta_clamp_max=1.2,
        theta_z_gamma=0.0,
        K=7e-12,
        De=13.0,
    ),
    launch=LaunchRequest(
        power_mW=0.1,
        waist_x_um=4.0,
        waist_y_um=4.0,
        separation_um=0.0,
        pair_angle_deg=0.0,
        theta_out1_deg=0.0,
        theta_out2_deg=0.0,
        phi1_deg=0.0,
        phi2_deg=0.0,
        power_ratio=0.0,
        coherent=False,
    ),
    boundary=BoundaryRequest(use_sponge=True, windowedge=0.1),
    solver=SolverRequest(
        static_max_steps=200,
        Nt=1,
        dt=5e-4,
        t_stride=1,
        dtau_static=0.01,
        static_tol_rms=0.005,
        static_tol_max=0.01,
        static_selfcons_passes=3,
        static_mix=0.3,
        dz_opt_max_phi=0.3,
        dn_max_est=0.02,
        max_substeps=8,
        strict_max_outer_passes=24,
    ),
    output=OutputRequest(
        run_dir=str(run_dir),
        save_slices=False,
        save_full=False,
    ),
    runtime=RuntimeRequest(backend="cpu", progress=True),
)

def progress(*args, **kw):
    if kw:
        print("[eig]", kw)
    else:
        print("[eig]", args)

result = run_eigensoliton_existence_curve(
    req,
    powers_mW=[0.05, 0.10, 0.20, 0.50],
    run_dir=run_dir,
    w0_um=4.0,
    live_plot=False,
    save_profiles=True,
    progress_callback=progress,
solve_kwargs=dict(
    max_outer=160,
    theta_residual_tol_rms=5e-3,
    theta_residual_tol_max=5e-2,
),
)

print("DONE")
print("run_dir:", result["run_dir"])
print("csv:", result["csv"])
print("n_completed:", result["n_completed"])
print(result["dataframe"])
