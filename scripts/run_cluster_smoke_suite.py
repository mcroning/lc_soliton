from pathlib import Path
import time

from lc_soliton.request import (
    SimulationRequest, GridRequest, GeometryRequest, MaterialRequest,
    LaunchRequest, BoundaryRequest, SolverRequest, OutputRequest, RuntimeRequest,
)
from lc_soliton.engine import run_engine


ROOT = Path("small_runs/cluster_smoke")
ROOT.mkdir(parents=True, exist_ok=True)


def base_request(name, *, mode="time_dependent", use_dual_grid=False, factor=2):
    return SimulationRequest(
        mode=mode,
        grid=GridRequest(
            Nx=128 if not use_dual_grid else 256,
            Ny=128 if not use_dual_grid else 256,
            Nz=20,
            use_dual_grid=bool(use_dual_grid),
            dual_grid_factor=int(factor),
        ),
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
            K=7e-12,
            De=13.0,
        ),
        launch=LaunchRequest(
            power_mW=1.0,
            waist_x_um=3.0,
            waist_y_um=3.0,
            coherent=False,
        ),
        boundary=BoundaryRequest(
            use_sponge=True,
            windowedge=0.1,
        ),
        solver=SolverRequest(
            Nt=5,
            dt=5e-4,
            t_stride=1,
            static_max_steps=20,
            strict_max_outer_passes=8,
            dz_opt_max_phi=0.3,
            dn_max_est=0.02,
            max_substeps=16,
        ),
        output=OutputRequest(
            run_dir=str(ROOT / name),
            save_slices=True,
            save_full=False,
        ),
        runtime=RuntimeRequest(
            backend="auto",
            progress=True,
        ),
    )


def run_case(label, req):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    t0 = time.time()
    result = run_engine(req)
    print("elapsed_s =", time.time() - t0)
    print("result =", result)
    return result


if __name__ == "__main__":
    run_case(
        "TD full grid smoke",
        base_request("td_full", mode="time_dependent", use_dual_grid=False),
    )

    run_case(
        "TD dual-grid factor 2 smoke",
        base_request("td_dg2", mode="time_dependent", use_dual_grid=True, factor=2),
    )

    run_case(
        "TD dual-grid factor 4 smoke",
        base_request("td_dg4", mode="time_dependent", use_dual_grid=True, factor=4),
    )

    run_case(
        "Static smoke",
        base_request("static", mode="static", use_dual_grid=False),
    )

    print("\nSMOKE SUITE DONE")
