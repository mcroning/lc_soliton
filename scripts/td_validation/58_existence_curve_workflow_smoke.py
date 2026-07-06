#!/usr/bin/env python
"""58: command-line validation/driver for the clean existence-curve workflow.

This script exercises lc.workflows.existence_curve.run_existence_curve().

Default solver: clean production soliton solver.
Reference solver: strict z-stack via --solver static_zstack.
"""

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


def fmt(x, nd=6) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    ap.add_argument("--solver", choices=["soliton", "static_zstack"], default="soliton")
    ap.add_argument("--powers", default="0.5 1.0 2.0 4.0")

    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--z-um", type=float, default=250.0)
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

    ap.add_argument("--max-outer", type=int, default=80)
    ap.add_argument("--theta-steps", type=int, default=50)
    ap.add_argument("--field-mix", type=float, default=0.25)
    ap.add_argument("--theta-mix", type=float, default=1.0)
    ap.add_argument("--tol-field", type=float, default=1e-4)
    ap.add_argument("--tol-theta", type=float, default=1e-5)
    ap.add_argument("--tol-residual-rms", type=float, default=1e-3)
    ap.add_argument("--tol-residual-max", type=float, default=1e-2)

    ap.add_argument("--static-max-steps", type=int, default=50)
    ap.add_argument("--static-resid-every", type=int, default=10)
    ap.add_argument("--static-relax-omega", type=float, default=0.3)

    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    powers = parse_powers(args.powers)

    lc = LCSpec(
        material=LCMaterial(
            name="generic",
            ne=args.ne,
            no=args.no,
            K_N=args.K_N,
            delta_eps=args.delta_eps,
        ),
        cell=LCCell(
            thickness_um=args.thickness_um,
            y_aperture_um=args.y_aperture_um,
            interaction_length_um=args.z_um,
            theta_bc=args.theta_bc,
        ),
        bias=LCBias(Vapp=args.Vapp),
        mobility=1.0,
    )

    base = StaticRequest(
        lc=lc,
        beams=single_gaussian(
            wavelength_um=args.wavelength_um,
            power=1.0,
            waist_um=args.waist_um,
        ),
        backend=BackendSpec(args.backend, args.precision, verbose=True),
        grid=GridSpec(
            Nx=args.Nx,
            Ny=args.Ny,
            dz_um=args.dz_um,
            x_aperture_um=args.thickness_um,
            y_aperture_um=args.y_aperture_um,
            z_length_um=args.z_um,
        ),
        max_outer=args.max_outer,
        static_max_steps=args.static_max_steps,
        static_resid_every=args.static_resid_every,
        static_relax_omega=args.static_relax_omega,
        tol_residual_rms=args.tol_residual_rms,
        tol_residual_max=args.tol_residual_max,
        tridiag=args.tridiag,
    )

    req = ExistenceCurveRequest(
        base=base,
        powers=powers,
        solver=args.solver,
        continuation=True,
        soliton_max_outer=args.max_outer,
        theta_steps_per_outer=args.theta_steps,
        field_mix=args.field_mix,
        theta_mix=args.theta_mix,
        tol_field=args.tol_field,
        tol_theta=args.tol_theta,
        tol_residual_rms=args.tol_residual_rms,
        tol_residual_max=args.tol_residual_max,
    )

    result = run_existence_curve(req)

    print("58_existence_curve_workflow")
    print(f"  backend        : {args.backend}")
    print(f"  precision      : {args.precision}")
    print(f"  tridiag        : {args.tridiag}")
    print(f"  solver         : {args.solver}")
    print(f"  grid           : {args.Nx} x {args.Ny}, z={args.z_um:g} um, dz={args.dz_um:g} um")
    print(f"  LC             : V={args.Vapp:g} V, b={resolved_b(lc):.12g}, theta_bc={args.theta_bc:g}")
    print(f"  powers         : {' '.join(fmt(p, 4) for p in powers)}")
    print(f"  max_outer      : {args.max_outer}")
    print(f"  theta_steps    : {args.theta_steps}")
    print(f"  field_mix      : {args.field_mix:g}")
    print(
        f"  tolerances     : field={args.tol_field:g}, theta={args.tol_theta:g}, "
        f"res_rms={args.tol_residual_rms:g}, res_max={args.tol_residual_max:g}"
    )
    print()

    print(
        " P_req  seed    dir cont conv outer"
        "  P_out      beta      theta_max   Imax"
        "      sx_um    sy_um"
        "    res_rms    res_max   field_rel  elapsed_s"
    )
    print("-" * 170)

    table_rows: list[dict] = []
    for i, (row, res) in enumerate(zip(result.samples, result.results or [])):
        out = {
            "P_req_mW": row.get("requested_power", ""),
            "seed_power": None if i == 0 else float(powers[i - 1]),
            "direction": row.get("continuation_direction", ""),
            "continuation": row.get("continuation_used", False),
            "used_initial_A": row.get("used_initial_A", False),
            "used_initial_theta": row.get("used_initial_theta", False),
            "converged": row.get("converged", False),
            "outer_steps": res.metrics.get("outer_steps", ""),
            "P_out": row.get("output_power", res.metrics.get("power", "")),
            "beta": res.metrics.get("beta", ""),
            "theta_max": res.metrics.get("theta_max", ""),
            "Imax": res.metrics.get("Imax", ""),
            "sx_um": res.metrics.get("sx_um", ""),
            "sy_um": res.metrics.get("sy_um", ""),
            "residual_rms": res.metrics.get("residual_rms", ""),
            "residual_max": res.metrics.get("residual_max", ""),
            "field_rel": res.metrics.get("field_rel", ""),
            "elapsed_s": res.metrics.get("elapsed_s", ""),
        }

        table_rows.append(out)

        print(
            f"{fmt(out['P_req_mW'],4):>6s} "
            f"{fmt(out['seed_power'],4):>6s} "
            f"{str(out['direction']):>5s} "
            f"{fmt(out['continuation']):>4s} "
            f"{fmt(out['converged']):>4s} "
            f"{str(out['outer_steps']):>5s} "
            f"{fmt(out['P_out'],8):>10s} "
            f"{fmt(out['beta'],8):>10s} "
            f"{fmt(out['theta_max'],8):>11s} "
            f"{fmt(out['Imax'],8):>10s} "
            f"{fmt(out['sx_um'],6):>8s} "
            f"{fmt(out['sy_um'],6):>8s} "
            f"{fmt(out['residual_rms'],6):>10s} "
            f"{fmt(out['residual_max'],6):>10s} "
            f"{fmt(out['field_rel'],6):>11s} "
            f"{fmt(out['elapsed_s'],5):>10s}"
            )

    print()
    print(f"  num_points      : {result.metrics['num_points']}")
    print(f"  converged_count : {result.metrics['converged_count']}")

    if args.csv:
        path = Path(args.csv)
        write_csv(path, table_rows)
        print(f"  csv             : {path}")

    assert result.kind == "ExistenceCurveResult"
    assert len(result.samples) == len(powers)
    assert len(result.results or []) == len(powers)
    assert result.metrics["num_points"] == len(powers)

    print("58_existence_curve_workflow: PASS")


if __name__ == "__main__":
    main()
