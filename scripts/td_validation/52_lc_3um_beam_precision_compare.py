#!/usr/bin/env python
"""52: 3 um waist, 1 mW precision comparison using the clean lc.algorithms core."""

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


def get_backend(name: str):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy as cp
        return cp
    if name == "auto":
        try:
            import cupy as cp
            _ = cp.arange(1)
            return cp
        except Exception:
            return np
    raise ValueError(f"unknown backend {name!r}")


def asnumpy(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def dtype_pair(name: str):
    if name == "float32":
        return np.float32, np.complex64
    if name == "float64":
        return np.float64, np.complex128
    raise ValueError(name)


def compute_b_from_voltage(Vapp: float, delta_eps: float, K_N: float) -> float:
    eps0 = 8.8541878128e-12
    return float(delta_eps) * eps0 * float(Vapp) ** 2 / (8.0 * float(K_N))


def neff_from_theta(theta, *, ne: float, no: float, xp):
    c = xp.cos(theta)
    s = xp.sin(theta)
    return (float(ne) * float(no)) / xp.sqrt((float(ne) * c) ** 2 + (float(no) * s) ** 2)


def build_grid(args, *, xp, real_dtype):
    Nx = int(args.Nx)
    Ny = int(args.Ny)
    Nz = int(round(float(args.z_um) / float(args.dz_um)))
    if min(Nx, Ny, Nz) <= 0:
        raise ValueError("Nx, Ny, Nz must be positive")

    dx_um = float(args.xaper_um) / Nx
    dy_um = float(args.yaper_um) / Ny
    dz_um = float(args.dz_um)

    x_um = ((xp.arange(Nx, dtype=real_dtype) - Nx / 2) * dx_um + 0.5 * dx_um).astype(real_dtype, copy=False)
    y_um = ((xp.arange(Ny, dtype=real_dtype) - Ny / 2) * dy_um + 0.5 * dy_um).astype(real_dtype, copy=False)

    fx = xp.fft.fftfreq(Nx, d=dx_um).astype(real_dtype, copy=False)
    fy = xp.fft.fftfreq(Ny, d=dy_um).astype(real_dtype, copy=False)
    fxy2 = (fx[:, None] ** 2 + fy[None, :] ** 2).astype(real_dtype, copy=False)

    du = 2.0 / max(1, Nx - 1)
    dv = du * (dy_um / dx_um)

    return dict(Nx=Nx, Ny=Ny, Nz=Nz, dx_um=dx_um, dy_um=dy_um, dz_um=dz_um, du=du, dv=dv, x_um=x_um, y_um=y_um, fxy2=fxy2)


def build_theta_bias(args, grid, *, xp, real_dtype):
    x = grid["x_um"]
    center = math.pi / 4 if args.theta_center is None else float(args.theta_center)
    xnorm = x / max(1e-300, float(args.xaper_um) / 2.0)
    profile = float(args.theta_bc) + (center - float(args.theta_bc)) * xp.cos(0.5 * xp.pi * xnorm)
    profile = xp.maximum(profile, float(args.theta_bc))
    theta = xp.repeat(profile[:, None], int(grid["Ny"]), axis=1)
    theta = xp.clip(theta, float(args.theta_min), float(args.theta_max)).astype(real_dtype, copy=False)
    theta[0, :] = float(args.theta_bc)
    theta[-1, :] = float(args.theta_bc)
    return theta


def build_launch_stack(args, grid, *, xp, complex_dtype):
    x = grid["x_um"]
    y = grid["y_um"]
    X = x[:, None]
    Y = y[None, :]

    wx = float(args.waist_um)
    wy = float(args.waist_um)
    A = xp.exp(-((X / wx) ** 2 + (Y / wy) ** 2)).astype(complex_dtype, copy=False)
    if args.tilt_x_rad_per_um or args.tilt_y_rad_per_um:
        A = A * xp.exp(1j * (float(args.tilt_x_rad_per_um) * X + float(args.tilt_y_rad_per_um) * Y))

    dxdy = float(grid["dx_um"]) * float(grid["dy_um"])
    p0 = xp.sum(xp.abs(A) ** 2) * dxdy
    A = A * xp.sqrt(float(args.power) / p0)
    return A.astype(complex_dtype, copy=False)[None, :, :]


def moment_metrics(I, grid, *, xp):
    P = xp.sum(I) * float(grid["dx_um"]) * float(grid["dy_um"])
    eps = xp.asarray(1e-300, dtype=I.dtype)

    x = grid["x_um"]
    y = grid["y_um"]

    raw = xp.sum(I) + eps
    xc = xp.sum(I * x[:, None]) / raw
    yc = xp.sum(I * y[None, :]) / raw
    sx = xp.sqrt(xp.sum(I * (x[:, None] - xc) ** 2) / raw)
    sy = xp.sqrt(xp.sum(I * (y[None, :] - yc) ** 2) / raw)

    return dict(
        power=float(asnumpy(P)),
        Imax=float(asnumpy(xp.max(I))),
        sx_um=float(asnumpy(sx)),
        sy_um=float(asnumpy(sy)),
        xc_um=float(asnumpy(xc)),
        yc_um=float(asnumpy(yc)),
    )


def run_case(args, dtype_name: str):
    _ensure_src_on_path()

    from lc.algorithms.splitstep import advance_slice_with_midintensity, linear_kernel, total_intensity, hop_linear_inplace
    from lc.algorithms.theta_cn import prepare_cn_operator
    from lc.algorithms.theta_picard import cn_trapezoid_picard_step
    from lc.algorithms.thomas import solve_const_offdiag_batched
    from lc.algorithms.td_zmarch import TDZMarchControls, run_td_zmarch

    if args.tridiag == "fast":
        from lc.algorithms.thomas_fast import solve_const_offdiag_batched_fast as tridiag_solver
    else:
        tridiag_solver = solve_const_offdiag_batched

    real_dtype, complex_dtype = dtype_pair(dtype_name)
    xp = get_backend(args.backend)

    grid = build_grid(args, xp=xp, real_dtype=real_dtype)

    theta0_2d = build_theta_bias(args, grid, xp=xp, real_dtype=real_dtype)
    theta0 = xp.repeat(theta0_2d[None, :, :], grid["Nz"], axis=0).astype(real_dtype, copy=False)

    A0 = build_launch_stack(args, grid, xp=xp, complex_dtype=complex_dtype)

    b = compute_b_from_voltage(args.Vapp, args.delta_eps, args.K_N) if args.b is None else float(args.b)
    n_ref = float(asnumpy(neff_from_theta(theta0_2d, ne=args.ne, no=args.no, xp=xp)).mean())

    k0 = 2.0 * math.pi / float(args.wavelength_um)
    phi_est = k0 * float(args.dz_um) * float(args.dn_max_est)
    Nsub = max(1, min(int(args.max_substeps), int(math.ceil(phi_est / float(args.dz_opt_max_phi)))))
    dz_sub = float(args.dz_um) / Nsub
    h_sub = linear_kernel(grid["fxy2"], dz=dz_sub, wavelength=args.wavelength_um, n_ref=n_ref, xp=xp)
    h_half = linear_kernel(grid["fxy2"], dz=0.5 * float(args.dz_um), wavelength=args.wavelength_um, n_ref=n_ref, xp=xp)

    s_cn, off_cn, diag_cn, lam_y_cn = prepare_cn_operator(dt=float(args.dt), mobility=float(args.mobility), dx=float(grid["du"]), dy=float(grid["dv"]), Ny=int(grid["Ny"]), xp=xp, dtype=real_dtype)

    samples = []
    sample_every = max(1, int(args.sample_every))
    sample_set = set(range(0, grid["Nz"], sample_every))
    sample_set.add(grid["Nz"] - 1)

    start = time.perf_counter()

    def optics_half_step(A):
        return hop_linear_inplace(A, h_half, xp=xp)

    def optics_step(A, theta_k, k):
        A, I_b, I_a, I_mid = advance_slice_with_midintensity(A, theta_k, kernel=h_sub, dz=float(args.dz_um), wavelength=float(args.wavelength_um), n_ref=n_ref, ne=float(args.ne), no=float(args.no), Nsub=Nsub, coherent=False, theta_weights=None, xp=xp)
        return A, I_mid

    def theta_step(theta_k, I_mid, theta_prev, theta_next, k):
        out = cn_trapezoid_picard_step(theta_k, I_mid, I_mid, b=b, bi=float(args.bi), dt=float(args.dt), mobility=float(args.mobility), dx=float(grid["du"]), dy=float(grid["dv"]), s=s_cn, off=off_cn, diag=diag_cn, max_iter=int(args.picard_iters), tol_update=float(args.picard_tol_up), clamp=(float(args.theta_min), float(args.theta_max)), tridiag_solver=tridiag_solver, xp=xp)
        out[0, :] = float(args.theta_bc)
        out[-1, :] = float(args.theta_bc)
        return out

    def observer(payload):
        jt = payload["jt"]
        k = payload["k"]
        if jt == int(args.Nt) and k in sample_set:
            I_mid = payload["I_mid"]
            mm = moment_metrics(I_mid, grid, xp=xp)
            theta_k = payload["theta_k"]
            samples.append([jt, k, (k + 1) * float(args.dz_um), mm["power"], mm["Imax"], mm["sx_um"], mm["sy_um"], mm["xc_um"], mm["yc_um"], float(asnumpy(xp.max(theta_k)))])

    res = run_td_zmarch(theta0, A0, optics_step=optics_step, theta_step=theta_step, controls=TDZMarchControls(Nt=int(args.Nt), observer_stride_z=sample_every, observer_stride_t=1), optics_half_step=optics_half_step, observer=observer)

    elapsed = time.perf_counter() - start

    final_I = total_intensity(res.A_last, coherent=False, xp=xp)
    final_mm = moment_metrics(final_I, grid, xp=xp)

    theta_final = res.theta
    theta_delta = theta_final - theta0

    metrics = dict(
        dtype=dtype_name,
        tridiag=str(args.tridiag),
        backend=getattr(xp, "__name__", str(xp)),
        Nx=grid["Nx"], Ny=grid["Ny"], Nz=grid["Nz"], Nt=int(args.Nt),
        z_um=float(args.z_um), dz_um=float(args.dz_um), Nsub=Nsub, dz_sub_um=dz_sub,
        elapsed_s=float(elapsed), sec_per_zslice=float(elapsed / max(1, grid["Nz"] * int(args.Nt))),
        b=float(b), bi=float(args.bi), n_ref=float(n_ref),
        initial_theta_max=float(asnumpy(xp.max(theta0))),
        final_theta_min=float(asnumpy(xp.min(theta_final))),
        final_theta_max=float(asnumpy(xp.max(theta_final))),
        max_theta_change=float(asnumpy(xp.max(xp.abs(theta_delta)))),
        final_power=final_mm["power"], final_Imax=final_mm["Imax"], final_sx_um=final_mm["sx_um"], final_sy_um=final_mm["sy_um"], final_xc_um=final_mm["xc_um"], final_yc_um=final_mm["yc_um"],
    )

    rows = np.asarray(samples, dtype=np.float64)

    if args.save_npz:
        save_path = Path(str(args.save_npz).format(dtype=dtype_name))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, metrics_keys=np.asarray(list(metrics.keys()), dtype=object), metrics_values=np.asarray(list(metrics.values()), dtype=object), rows=rows, row_columns=np.asarray(["jt", "k", "z_um", "power", "Imax", "sx_um", "sy_um", "xc_um", "yc_um", "theta_max"], dtype=object), final_I=asnumpy(final_I), final_theta=asnumpy(theta_final[-1]), x_um=asnumpy(grid["x_um"]), y_um=asnumpy(grid["y_um"]))
        metrics["saved_npz"] = str(save_path)

    return metrics, rows


def print_case(metrics, rows):
    print(f"  dtype                    : {metrics['dtype']}")
    for k, v in metrics.items():
        if k == "dtype":
            continue
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.12g}")
        else:
            print(f"  {k:24s}: {v}")

    if rows.size:
        print("  sampled_rows:")
        print("    jt    k      z_um      power        Imax        sx_um      sy_um      xc_um      yc_um    theta_max")
        for row in rows:
            print(f"    {int(row[0]):3d} {int(row[1]):4d} {row[2]:9.1f} {row[3]:.6e} {row[4]:.6e} {row[5]:9.4f} {row[6]:9.4f} {row[7]:9.4f} {row[8]:9.4f} {row[9]:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "cupy", "numpy"], default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64", "both"], default="both")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="reference")
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--xaper-um", type=float, default=75.0)
    ap.add_argument("--yaper-um", type=float, default=100.0)
    ap.add_argument("--z-um", type=float, default=500.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--Nt", type=int, default=20)
    ap.add_argument("--dt", type=float, default=7.5e-4)
    ap.add_argument("--waist-um", type=float, default=3.0)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--tilt-x-rad-per-um", type=float, default=0.0)
    ap.add_argument("--tilt-y-rad-per-um", type=float, default=0.0)
    ap.add_argument("--Vapp", type=float, default=0.9144)
    ap.add_argument("--delta-eps", type=float, default=13.0)
    ap.add_argument("--K-N", type=float, default=7.0e-12)
    ap.add_argument("--b", type=float, default=None)
    ap.add_argument("--bi", type=float, default=214.28571428571428)
    ap.add_argument("--mobility", type=float, default=1.0)
    ap.add_argument("--theta-bc", type=float, default=0.0)
    ap.add_argument("--theta-center", type=float, default=None)
    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=math.pi / 2)
    ap.add_argument("--picard-iters", type=int, default=4)
    ap.add_argument("--picard-tol-up", type=float, default=1e-6)
    ap.add_argument("--wavelength-um", type=float, default=0.633)
    ap.add_argument("--ne", type=float, default=1.7)
    ap.add_argument("--no", type=float, default=1.5)
    ap.add_argument("--dn-max-est", type=float, default=0.02)
    ap.add_argument("--dz-opt-max-phi", type=float, default=0.30)
    ap.add_argument("--max-substeps", type=int, default=16)
    ap.add_argument("--sample-every", type=int, default=25)
    ap.add_argument("--save-npz", default=None, help="Optional path. May include {dtype}.")
    args = ap.parse_args()

    dtypes = ["float32", "float64"] if args.dtype == "both" else [args.dtype]

    print("52_lc_3um_beam_precision_compare")
    all_metrics = []
    for dtname in dtypes:
        metrics, rows = run_case(args, dtname)
        all_metrics.append(metrics)
        print_case(metrics, rows)

    if len(all_metrics) == 2:
        m32, m64 = all_metrics
        print("  comparison_float64_minus_float32:")
        for key in ["final_theta_max", "max_theta_change", "final_Imax", "final_sx_um", "final_sy_um", "final_power"]:
            print(f"    {key:22s}: {float(m64[key]) - float(m32[key]): .12e}")

    print("52_lc_3um_beam_precision_compare: PASS")


if __name__ == "__main__":
    main()
