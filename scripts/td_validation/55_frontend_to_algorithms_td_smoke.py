#!/usr/bin/env python
"""55: wire clean front-end preparation into clean algorithms for TD.

This is the first end-to-end smoke test using the new architecture:

    physics/liquid_crystal.py
    physics/beam.py
    physics/bias.py
    physics/launch.py
    numerics/backend.py
    numerics/grid.py
        ↓
    algorithms/splitstep.py
    algorithms/theta_picard.py
    algorithms/td_zmarch.py

It intentionally stays small and fast.  The goal is to verify that a human
experiment specification can be prepared and consumed by the numerical core.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def moment_metrics(I, grid, *, asnumpy):
    xp = grid.xp
    raw = xp.sum(I) + xp.asarray(1e-300, dtype=I.dtype)
    P = raw * float(grid.dx_um) * float(grid.dy_um)

    x = grid.x_um
    y = grid.y_um

    xc = xp.sum(I * x[:, None]) / raw
    yc = xp.sum(I * y[None, :]) / raw
    sx = xp.sqrt(xp.sum(I * (x[:, None] - xc) ** 2) / raw)
    sy = xp.sqrt(xp.sum(I * (y[None, :] - yc) ** 2) / raw)

    return {
        "power": float(asnumpy(P)),
        "Imax": float(asnumpy(xp.max(I))),
        "sx_um": float(asnumpy(sx)),
        "sy_um": float(asnumpy(sy)),
        "xc_um": float(asnumpy(xc)),
        "yc_um": float(asnumpy(yc)),
    }


def main() -> None:
    _ensure_src_on_path()

    from lc.numerics.backend import BackendSpec, get_backend, asnumpy, synchronize
    from lc.numerics.grid import GridSpec, TimeSpec, make_grid
    from lc.physics.liquid_crystal import LCBias, LCCell, LCMaterial, LCSpec, resolved_b, neff_from_theta
    from lc.physics.beam import single_gaussian
    from lc.physics.bias import build_bias
    from lc.physics.launch import build_launch
    from lc.algorithms.splitstep import (
        advance_slice_with_midintensity,
        hop_linear_inplace,
        linear_kernel,
        total_intensity,
    )
    from lc.algorithms.theta_cn import prepare_cn_operator
    from lc.algorithms.theta_picard import cn_trapezoid_picard_step
    from lc.algorithms.thomas import solve_const_offdiag_batched
    from lc.algorithms.td_zmarch import TDZMarchControls, run_td_zmarch

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--z-um", type=float, default=500.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--Nt", type=int, default=3)
    ap.add_argument("--dt", type=float, default=7.5e-4)
    ap.add_argument("--sample-every", type=int, default=25)
    args = ap.parse_args()

    if args.tridiag == "fast":
        from lc.algorithms.thomas_fast import solve_const_offdiag_batched_fast as tridiag_solver
    else:
        tridiag_solver = solve_const_offdiag_batched

    backend = get_backend(BackendSpec(args.backend, args.precision, verbose=True))

    lc = LCSpec(
        material=LCMaterial(name="generic", ne=1.7, no=1.5, K_N=7.0e-12, delta_eps=13.0),
        cell=LCCell(
            thickness_um=75.0,
            y_aperture_um=100.0,
            interaction_length_um=float(args.z_um),
            theta_bc=0.0,
        ),
        bias=LCBias(Vapp=0.9144),
        mobility=1.0,
    )
    lc.validate()

    time_spec = TimeSpec(dt=float(args.dt), Nt=int(args.Nt))
    time_spec.validate()

    grid = make_grid(
        GridSpec(
            Nx=int(args.Nx),
            Ny=int(args.Ny),
            dz_um=float(args.dz_um),
            x_aperture_um=lc.cell.thickness_um,
            y_aperture_um=lc.cell.y_aperture_um,
            z_length_um=lc.cell.interaction_length_um,
        ),
        xp=backend.xp,
        real_dtype=backend.real_dtype,
    )

    bias = build_bias(lc, grid)

    beams = single_gaussian(wavelength_um=0.633, power=1.0, waist_um=3.0)
    launch = build_launch(beams, grid, complex_dtype=backend.complex_dtype)

    xp = backend.xp
    b = resolved_b(lc)

    n_ref = float(asnumpy(neff_from_theta(
        bias.theta_2d,
        ne=lc.material.ne,
        no=lc.material.no,
        xp=xp,
    )).mean())

    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))
    k0 = 2.0 * math.pi / wavelength_um
    dn_max_est = 0.02
    dz_opt_max_phi = 0.30
    max_substeps = 16
    phi_est = k0 * float(grid.dz_um) * dn_max_est
    Nsub = max(1, min(max_substeps, int(math.ceil(phi_est / dz_opt_max_phi))))
    dz_sub = float(grid.dz_um) / Nsub

    h_sub = linear_kernel(grid.fxy2_um, dz=dz_sub, wavelength=wavelength_um, n_ref=n_ref, xp=xp)
    h_half = linear_kernel(grid.fxy2_um, dz=0.5 * float(grid.dz_um), wavelength=wavelength_um, n_ref=n_ref, xp=xp)

    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=time_spec.dt,
        mobility=lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    samples = []
    sample_every = max(1, int(args.sample_every))
    sample_set = set(range(0, grid.Nz, sample_every))
    sample_set.add(grid.Nz - 1)

    def optics_half_step(A):
        return hop_linear_inplace(A, h_half, xp=xp)

    def optics_step(A, theta_k, k):
        A, I_before, I_after, I_mid = advance_slice_with_midintensity(
            A,
            theta_k,
            kernel=h_sub,
            dz=float(grid.dz_um),
            wavelength=wavelength_um,
            n_ref=n_ref,
            ne=lc.material.ne,
            no=lc.material.no,
            Nsub=Nsub,
            coherent=(launch.coherence == "coherent"),
            theta_weights=launch.theta_weights,
            xp=xp,
        )
        return A, I_mid

    def theta_step(theta_k, I_mid, theta_prev, theta_next, k):
        out = cn_trapezoid_picard_step(
            theta_k,
            I_mid,
            I_mid,
            b=b,
            bi=214.28571428571428,
            dt=time_spec.dt,
            mobility=lc.mobility,
            dx=grid.du,
            dy=grid.dv,
            s=s_cn,
            off=off_cn,
            diag=diag_cn,
            max_iter=4,
            tol_update=1e-6,
            clamp=bias.theta_clamp,
            tridiag_solver=tridiag_solver,
            xp=xp,
        )
        out[0, :] = lc.cell.theta_bc
        out[-1, :] = lc.cell.theta_bc
        return out

    def observer(payload):
        jt = payload["jt"]
        k = payload["k"]
        if jt == time_spec.Nt and k in sample_set:
            mm = moment_metrics(payload["I_mid"], grid, asnumpy=asnumpy)
            samples.append([
                jt, k, (k + 1) * grid.dz_um,
                mm["power"], mm["Imax"], mm["sx_um"], mm["sy_um"],
                mm["xc_um"], mm["yc_um"],
                float(asnumpy(xp.max(payload["theta_k"]))),
            ])

    synchronize(xp)
    t0 = time.perf_counter()
    result = run_td_zmarch(
        bias.theta_stack,
        launch.A0,
        optics_step=optics_step,
        theta_step=theta_step,
        controls=TDZMarchControls(Nt=time_spec.Nt, observer_stride_z=sample_every, observer_stride_t=1),
        optics_half_step=optics_half_step,
        observer=observer,
    )
    synchronize(xp)
    elapsed = time.perf_counter() - t0

    final_I = total_intensity(result.A_last, coherent=(launch.coherence == "coherent"), xp=xp)
    mm_final = moment_metrics(final_I, grid, asnumpy=asnumpy)
    theta_final = result.theta

    print("55_frontend_to_algorithms_td_smoke")
    print(f"  backend                 : {backend.name}")
    print(f"  precision               : {args.precision}")
    print(f"  tridiag                 : {args.tridiag}")
    print(f"  grid                    : {grid.Nx} x {grid.Ny} x {grid.Nz}")
    print(f"  Nt                      : {time_spec.Nt}")
    print(f"  Nsub                    : {Nsub}")
    print(f"  dz_sub_um               : {dz_sub:.12g}")
    print(f"  elapsed_s               : {elapsed:.12g}")
    print(f"  sec_per_zslice          : {elapsed / max(1, grid.Nz * time_spec.Nt):.12g}")
    print(f"  b                       : {b:.12g}")
    print(f"  n_ref                   : {n_ref:.12g}")
    print(f"  final_power             : {mm_final['power']:.12g}")
    print(f"  final_Imax              : {mm_final['Imax']:.12g}")
    print(f"  final_sx_um             : {mm_final['sx_um']:.12g}")
    print(f"  final_sy_um             : {mm_final['sy_um']:.12g}")
    print(f"  final_xc_um             : {mm_final['xc_um']:.12g}")
    print(f"  final_yc_um             : {mm_final['yc_um']:.12g}")
    print(f"  final_theta_max         : {float(asnumpy(xp.max(theta_final))):.12g}")

    if samples:
        print("  sampled_rows:")
        print("    jt    k      z_um      power        Imax        sx_um      sy_um      xc_um      yc_um    theta_max")
        for row in samples:
            print(
                f"    {int(row[0]):3d} {int(row[1]):4d} {row[2]:9.1f} "
                f"{row[3]:.6e} {row[4]:.6e} {row[5]:9.4f} {row[6]:9.4f} "
                f"{row[7]:9.4f} {row[8]:9.4f} {row[9]:.6f}"
            )

    assert abs(mm_final["power"] - 1.0) < (5e-3 if args.precision == "float32" else 1e-8)
    assert mm_final["sx_um"] < 10.0
    assert mm_final["sy_um"] < 10.0
    assert float(asnumpy(xp.max(theta_final))) > float(asnumpy(xp.max(bias.theta_stack)))

    print("55_frontend_to_algorithms_td_smoke: PASS")


if __name__ == "__main__":
    main()
