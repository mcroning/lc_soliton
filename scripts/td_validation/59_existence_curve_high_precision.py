#!/usr/bin/env python
"""59: high-precision existence curve using the clean LC workflow."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_powers(text: str) -> tuple[float, ...]:
    vals = tuple(float(p) for p in text.replace(",", " ").split())
    if not vals:
        raise ValueError("empty power list")
    return vals


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def fmt(x, nd=6):
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def main() -> None:
    _ensure_src_on_path()

    from lc.request import StaticRequest
    from lc.numerics.backend import BackendSpec
    from lc.numerics.grid import GridSpec
    from lc.physics.beam import single_gaussian
    from lc.physics.liquid_crystal import LCSpec, LCCell, LCMaterial, LCBias, resolved_b
    from lc.workflows.existence_curve import ExistenceCurveRequest, run_existence_curve

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="cupy")
    ap.add_argument("--precision", choices=["float32", "float64"], default="float64")
    ap.add_argument("--tridiag", choices=["reference", "fast"], default="fast")
    ap.add_argument("--powers", default="0.1 0.2 0.5 1.0 2.0 4.0")
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--z-um", type=float, default=3000.0)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--Vapp", type=float, default=0.9144)
    ap.add_argument("--K-N", type=float, default=7.0e-12)
    ap.add_argument("--delta-eps", type=float, default=13.0)
    ap.add_argument("--ne", type=float, default=1.7)
    ap.add_argument("--no", type=float, default=1.5)
    ap.add_argument("--thickness-um", type=float, default=75.0)
    ap.add_argument("--y-aperture-um", type=float, default=100.0)
    ap.add_argument("--theta-bc", type=float, default=0.0)
    ap.add_argument("--wavelength-um", type=float, default=0.633)
    ap.add_argument("--waist-um", type=float, default=3.0)
    ap.add_argument("--max-outer", type=int, default=20)
    ap.add_argument("--static-inner-steps", type=int, default=1)
    ap.add_argument("--static-max-steps", type=int, default=50)
    ap.add_argument("--static-resid-every", type=int, default=10)
    ap.add_argument("--static-relax-omega", type=float, default=0.3)
    ap.add_argument("--tol-rms", type=float, default=5e-3)
    ap.add_argument("--tol-max", type=float, default=2e-2)
    ap.add_argument("--tol-residual-rms", type=float, default=5e-3)
    ap.add_argument("--tol-residual-max", type=float, default=2e-1)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    powers = parse_powers(args.powers)
    lc = LCSpec(
        material=LCMaterial(name="generic", ne=args.ne, no=args.no, K_N=args.K_N, delta_eps=args.delta_eps),
        cell=LCCell(thickness_um=args.thickness_um, y_aperture_um=args.y_aperture_um, interaction_length_um=args.z_um, theta_bc=args.theta_bc),
        bias=LCBias(Vapp=args.Vapp),
        mobility=1.0,
    )

    base = StaticRequest(
        lc=lc,
        beams=single_gaussian(wavelength_um=args.wavelength_um, power=1.0, waist_um=args.waist_um),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(Nx=args.Nx, Ny=args.Ny, dz_um=args.dz_um, x_aperture_um=args.thickness_um, y_aperture_um=args.y_aperture_um, z_length_um=args.z_um),
        max_outer=args.max_outer,
        static_inner_steps=args.static_inner_steps,
        static_max_steps=args.static_max_steps,
        static_resid_every=args.static_resid_every,
        static_relax_omega=args.static_relax_omega,
        tol_rms=args.tol_rms,
        tol_max=args.tol_max,
        tol_residual_rms=args.tol_residual_rms,
        tol_residual_max=args.tol_residual_max,
        tridiag=args.tridiag,
    )

    result = run_existence_curve(ExistenceCurveRequest(base=base, powers=powers))

    print("59_existence_curve_high_precision")
    print(f"  backend      : {args.backend}")
    print(f"  precision    : {args.precision}")
    print(f"  tridiag      : {args.tridiag}")
    print("  workflow     : static_zstack")
    print(f"  continuation : {'enabled' if result.metrics.get('continuation', True) else 'disabled'}")
    if result.results:
        shape = result.results[0].theta.shape
        print(f"  theta stack  : {shape[0]} x {shape[1]} x {shape[2]}")
    print(f"  grid         : {args.Nx} x {args.Ny}, z={args.z_um:g} um, dz={args.dz_um:g} um")
    print(f"  LC           : V={args.Vapp:g} V, b={resolved_b(lc):.12g}, theta_bc={args.theta_bc:g}")
    print(f"  powers       : {' '.join(fmt(p, 4) for p in powers)}")
    print(f"  static steps : max={args.static_max_steps}, resid_every={args.static_resid_every}, omega={args.static_relax_omega:g}")
    print(f"  residual tol : rms={args.tol_residual_rms:g}, max={args.tol_residual_max:g}")
    print()
    print(" P_req   conv outer  P_out       theta_max   Imax       sx_um    sy_um    dtheta_rms  dtheta_max  res_rms     res_max     elapsed_s")
    print("-" * 136)

    table_rows: list[dict] = []
    for row, res in zip(result.samples, result.results or []):
        out = {
            "P_req_mW": row.get("requested_power", ""),
            "converged": row.get("converged", False),
            "outer_steps": res.metrics.get("outer_steps", ""),
            "P_out": row.get("output_power", res.metrics.get("power", "")),
            "theta_max": res.metrics.get("theta_max", ""),
            "Imax": res.metrics.get("Imax", ""),
            "sx_um": res.metrics.get("sx_um", ""),
            "sy_um": res.metrics.get("sy_um", ""),
            "dtheta_rms": res.metrics.get("dtheta_rms", ""),
            "dtheta_max": res.metrics.get("dtheta_max", ""),
            "residual_rms": res.metrics.get("residual_rms", ""),
            "residual_max": res.metrics.get("residual_max", ""),
            "elapsed_s": res.metrics.get("elapsed_s", ""),
        }
        table_rows.append(out)
        print(
            f"{fmt(out['P_req_mW'],4):>6s} {fmt(out['converged']):>6s} {str(out['outer_steps']):>5s} "
            f"{fmt(out['P_out'],8):>10s} {fmt(out['theta_max'],8):>11s} {fmt(out['Imax'],8):>10s} "
            f"{fmt(out['sx_um'],6):>8s} {fmt(out['sy_um'],6):>8s} {fmt(out['dtheta_rms'],6):>11s} "
            f"{fmt(out['dtheta_max'],6):>11s} {fmt(out['residual_rms'],6):>10s} {fmt(out['residual_max'],6):>10s} "
            f"{fmt(out['elapsed_s'],5):>10s}"
        )

    print()
    print(f"  num_points      : {result.metrics['num_points']}")
    print(f"  converged_count : {result.metrics['converged_count']}")
    if args.csv:
        path = Path(args.csv)
        write_csv(path, table_rows)
        print(f"  csv             : {path}")
    assert result.metrics["num_points"] == len(powers)
    print("59_existence_curve_high_precision: PASS")


if __name__ == "__main__":
    main()
