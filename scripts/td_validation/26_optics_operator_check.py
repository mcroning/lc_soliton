#!/usr/bin/env python3
"""
26_optics_operator_check.py

Validate the lowest-level optics_engine operators.

This test is independent of legacy code.  It checks mathematical identities
and FFT-grid/kernel properties before we build launch/propagation layers.
"""

import argparse
import csv
from pathlib import Path

import cupy as cp

from lc_soliton.optics_engine.operators import (
    neff_from_theta,
    delta_n_from_theta,
    transverse_k_grids,
    angular_spectrum_kernel,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--dx", type=float, default=0.2935420744)  # 75/(256-1) um
    ap.add_argument("--dy", type=float, default=0.390625)      # 100/256 um
    ap.add_argument("--dz", type=float, default=5.0)
    ap.add_argument("--wavelength", type=float, default=0.633)
    ap.add_argument("--ne", type=float, default=1.7)
    ap.add_argument("--no", type=float, default=1.5)
    ap.add_argument("--n0", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Nx = int(args.Nx)
    Ny = int(args.Ny)

    rows = []

    # ------------------------------------------------------------------
    # 1. n_eff identities
    # ------------------------------------------------------------------
    th_test = cp.asarray([0.0, 0.25 * cp.pi, 0.5 * cp.pi], dtype=cp.float64)
    neff = neff_from_theta(th_test, ne=args.ne, no=args.no, xp=cp)

    expected_0 = float(args.no)
    expected_pi2 = float(args.ne)
    expected_pi4 = float(
        (args.ne * args.no)
        / cp.sqrt(0.5 * args.ne * args.ne + 0.5 * args.no * args.no)
    )

    neff_errors = {
        "neff_theta0_error": abs(float(neff[0]) - expected_0),
        "neff_theta_pi4_error": abs(float(neff[1]) - expected_pi4),
        "neff_theta_pi2_error": abs(float(neff[2]) - expected_pi2),
    }

    # ------------------------------------------------------------------
    # 2. delta_n median subtraction
    # ------------------------------------------------------------------
    theta_grid = cp.linspace(0.0, 0.9, Nx * Ny, dtype=cp.float64).reshape(Nx, Ny)
    dn = delta_n_from_theta(theta_grid, ne=args.ne, no=args.no, n_ref=None, xp=cp)

    dn_stats = {
        "dn_median": float(cp.median(dn)),
        "dn_min": float(cp.min(dn)),
        "dn_max": float(cp.max(dn)),
    }

    # ------------------------------------------------------------------
    # 3. FFT k grids
    # ------------------------------------------------------------------
    kx, ky = transverse_k_grids(Nx, Ny, dx=args.dx, dy=args.dy, xp=cp)

    kgrid_stats = {
        "kx_shape_ok": bool(kx.shape == (Nx, 1)),
        "ky_shape_ok": bool(ky.shape == (1, Ny)),
        "kx0": float(kx[0, 0]),
        "ky0": float(ky[0, 0]),
        "kx_abs_max": float(cp.max(cp.abs(kx))),
        "ky_abs_max": float(cp.max(cp.abs(ky))),
        "kx_nyquist_expected": float(cp.pi / args.dx),
        "ky_nyquist_expected": float(cp.pi / args.dy),
    }

    # ------------------------------------------------------------------
    # 4. Angular spectrum kernel
    # ------------------------------------------------------------------
    H = angular_spectrum_kernel(
        Nx,
        Ny,
        dx=args.dx,
        dy=args.dy,
        wavelength=args.wavelength,
        dz=args.dz,
        n0=args.n0,
        xp=cp,
    )

    k0 = 2 * cp.pi / args.wavelength
    k = args.n0 * k0
    q2 = kx * kx + ky * ky
    propagating = q2 <= k * k

    absH = cp.abs(H)
    absH_prop = absH[propagating]

    kernel_stats = {
        "kernel_shape_ok": bool(H.shape == (Nx, Ny)),
        "propagating_fraction": float(cp.mean(propagating.astype(cp.float64))),
        "absH_min": float(cp.min(absH)),
        "absH_max": float(cp.max(absH)),
        "absH_prop_min": float(cp.min(absH_prop)),
        "absH_prop_max": float(cp.max(absH_prop)),
        "absH_prop_rms_error_from_1": float(
            cp.sqrt(cp.mean((absH_prop - 1.0) ** 2))
        ),
    }

    summary = {}
    summary.update(neff_errors)
    summary.update(dn_stats)
    summary.update(kgrid_stats)
    summary.update(kernel_stats)

    print()
    print("Optics operator validation")
    print("-" * 72)
    for k, v in summary.items():
        print(f"{k}: {v}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "operator_check.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    with open(outdir / "operator_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print()
    print("[wrote]", outdir / "operator_check.txt")
    print("[wrote]", outdir / "operator_check.csv")


if __name__ == "__main__":
    main()
