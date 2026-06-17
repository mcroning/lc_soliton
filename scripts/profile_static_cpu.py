from pathlib import Path
import shutil
import time

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
    run_engine,
)

run_dir = Path("runs/profile_static_cpu")
if run_dir.exists():
    shutil.rmtree(run_dir)

req = SimulationRequest(
    mode="strict_static",
    grid=GridRequest(Nx=256, Ny=256, Nz=300),
    geometry=GeometryRequest(
        xaper_um=75.0,
        yaper_um=100.0,
        dz_um=5.0,
        wavelength_um=0.633,
    ),
    material=MaterialRequest(
        ne=1.7,
        no=1.5,
        b=1.0071425,
        bi=214.285714285714,
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
        power_mW=1.0,
        waist_x_um=3.0,
        waist_y_um=3.0,
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
        static_max_steps=100,
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
        max_substeps=16,
    ),
    output=OutputRequest(
        run_dir=str(run_dir),
        save_slices=False,
        save_full=False,
    ),
    runtime=RuntimeRequest(
        backend="cpu",
        progress=False,
    ),
)

t0 = time.perf_counter()
result = run_engine(req)
dt = time.perf_counter() - t0

print("result:", result)
print(f"elapsed_s: {dt:.3f}")
print(f"run_dir: {run_dir}")