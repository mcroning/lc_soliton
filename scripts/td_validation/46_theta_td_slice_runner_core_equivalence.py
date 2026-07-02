#!/usr/bin/env python3
"""46_theta_td_slice_runner_core_equivalence.py

Validate the extracted z-stack TD theta-slice module against the trusted
``runner_core.py`` implementation for one theta update.

This is the theta analogue of test 45.  It compares the numerical update used
inside the production TD z-loop:

    theta_new[k] = td_update_theta_from_midintensity(...)

for the same theta slice, z-neighbor slices, midpoint intensity, context, and
prepared IE operator.

The test intentionally does not propagate optics and does not run a full TD
workflow.  It anchors the theta-zstack port to the working runner logic.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    root = _repo_root()
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_runner_core(path_or_module: str | None):
    """Load trusted runner_core by dotted module or file path."""
    _ensure_src_on_path()

    explicit = path_or_module or os.environ.get("RUNNER_CORE_PATH")
    if explicit is None:
        candidates = [
            "lc_soliton.validated_core.runner_core",
            "lc_soliton.runner_core",
            "lc_soliton.core.runner_core",
            "lc_soliton.legacy_validated.runner_core",
            "runner_core",
        ]
        errors: list[str] = []
        for name in candidates:
            try:
                return importlib.import_module(name)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {type(e).__name__}: {e}")
        raise ImportError("Could not import runner_core. Tried:\n  " + "\n  ".join(errors))

    if (
        "/" not in explicit
        and "\\" not in explicit
        and not explicit.endswith(".py")
    ):
        return importlib.import_module(explicit)

    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"runner_core path does not exist: {p}")

    # If runner_core uses relative imports, prefer deriving a package module
    # name when the path lies under src/lc_soliton.
    parts = p.parts
    if "src" in parts and "lc_soliton" in parts:
        try:
            i = parts.index("lc_soliton")
            rel = list(parts[i:])
            rel[-1] = rel[-1].removesuffix(".py")
            return importlib.import_module(".".join(rel))
        except Exception:
            pass

    spec = importlib.util.spec_from_file_location("_trusted_runner_core", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _asnumpy(a):
    try:
        import cupy as cupy  # type: ignore

        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _rel_l2(a, b) -> float:
    aa = _asnumpy(a).astype(np.float64, copy=False)
    bb = _asnumpy(b).astype(np.float64, copy=False)
    return float(np.linalg.norm((aa - bb).ravel()) / (np.linalg.norm(bb.ravel()) + 1e-300))


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(_asnumpy(a) - _asnumpy(b))))


def _make_ctx(*, Nx, Ny, du, dv, dz, b, bi, mobility, theta_bc, theta_z_gamma, theta_clamp, true_td_predictor_only, true_td_full_pred_optics, picard_iters, picard_tol_up):
    return SimpleNamespace(
        Nx=int(Nx),
        Ny=int(Ny),
        du=float(du),
        dv=float(dv),
        dz=float(dz),
        b=float(b),
        bi=float(bi),
        mobility=float(mobility),
        theta_bc=float(theta_bc),
        theta_z_gamma=float(theta_z_gamma),
        theta_clamp=theta_clamp,
        true_td_predictor_only=bool(true_td_predictor_only),
        true_td_full_pred_optics=bool(true_td_full_pred_optics),
        picard_iters=int(picard_iters),
        picard_tol_up=float(picard_tol_up),
        td_theta_relax_steps=2,
        td_theta_relax_omega=1.0,
        _cn_ky_cache={},
    )


def _make_fields(cp, *, Nx: int, Ny: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, Nx, dtype=np.float32)
    y = (np.arange(Ny, dtype=np.float32) - Ny / 2) * (2.0 / Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    base = 0.18 + 0.25 * np.sin(np.pi * (X + 1.0) / 2.0) ** 2
    theta_ref = base + 0.035 * np.cos(2 * np.pi * Y) + 0.015 * rng.normal(size=(Nx, Ny)).astype(np.float32)
    tp = theta_ref + 0.012 * np.sin(2 * np.pi * Y).astype(np.float32)
    tn = theta_ref - 0.009 * np.cos(2 * np.pi * Y).astype(np.float32)
    I_mid = 0.07 * np.exp(-(X**2 / 0.11 + Y**2 / 0.17)).astype(np.float32)
    I_mid += 0.003 * rng.random((Nx, Ny), dtype=np.float32)

    # Dirichlet x boundaries match theta_bc.
    for arr in (theta_ref, tp, tn):
        arr[0, :] = 0.0
        arr[-1, :] = 0.0

    return cp.asarray(theta_ref.astype(np.float32)), cp.asarray(tp.astype(np.float32)), cp.asarray(tn.astype(np.float32)), cp.asarray(I_mid.astype(np.float32))


def _run_case(old, new, *, args, case_name: str, true_td: bool, gamma_z: float, predictor_only: bool):
    cp = old.cp

    ctx_old = _make_ctx(
        Nx=args.Nx,
        Ny=args.Ny,
        du=args.du,
        dv=args.dv,
        dz=args.dz,
        b=args.b,
        bi=args.bi,
        mobility=args.mobility,
        theta_bc=args.theta_bc,
        theta_z_gamma=gamma_z,
        theta_clamp=(args.theta_min, args.theta_max),
        true_td_predictor_only=predictor_only,
        true_td_full_pred_optics=False,
        picard_iters=args.picard_iters,
        picard_tol_up=args.picard_tol_up,
    )
    ctx_new = _make_ctx(
        Nx=args.Nx,
        Ny=args.Ny,
        du=args.du,
        dv=args.dv,
        dz=args.dz,
        b=args.b,
        bi=args.bi,
        mobility=args.mobility,
        theta_bc=args.theta_bc,
        theta_z_gamma=gamma_z,
        theta_clamp=(args.theta_min, args.theta_max),
        true_td_predictor_only=predictor_only,
        true_td_full_pred_optics=False,
        picard_iters=args.picard_iters,
        picard_tol_up=args.picard_tol_up,
    )

    theta_ref, tp, tn, I_mid = _make_fields(cp, Nx=args.Nx, Ny=args.Ny, seed=args.seed)

    a_old, off_old, diag_old, lam_old = old.prepare_ie_ky_operator(
        dt=args.dt,
        mobility=ctx_old.mobility,
        du=ctx_old.du,
        dv=ctx_old.dv,
        Ny=ctx_old.Ny,
    )
    a_new, off_new, diag_new, lam_new = new.prepare_ie_ky_operator(
        dt=args.dt,
        mobility=ctx_new.mobility,
        du=ctx_new.du,
        dv=ctx_new.dv,
        Ny=ctx_new.Ny,
        xp=cp,
        dtype=theta_ref.dtype,
    )

    out_old = old.td_update_theta_from_midintensity(
        ctx_old,
        theta_ref,
        tp,
        tn,
        I_mid,
        dt_j=args.dt,
        true_td=true_td,
        a_ie=a_old,
        s=a_old,
        off=off_old,
        diag=diag_old,
        lam_y=lam_old,
        Nsub=1,
        dz_sub=args.dz,
        h_sub=None,
        amp_for_pred=None,
        I_b=None,
        I_a=None,
        Ahat=None,
        plan_f=None,
        plan_i=None,
    )

    out_new = new.td_update_theta_from_midintensity(
        ctx_new,
        theta_ref,
        tp,
        tn,
        I_mid,
        dt_j=args.dt,
        true_td=true_td,
        a_ie=a_new,
        s=a_new,
        off=off_new,
        diag=diag_new,
        lam_y=lam_new,
        Nsub=1,
        dz_sub=args.dz,
        h_sub=None,
        amp_for_pred=None,
        I_b=None,
        I_a=None,
        Ahat=None,
        plan_f=None,
        plan_i=None,
        xp=cp,
    )

    return {
        f"{case_name}_ie_a_rel": _rel_l2(a_new, a_old),
        f"{case_name}_ie_off_abs": _max_abs(off_new, off_old),
        f"{case_name}_ie_diag_rel": _rel_l2(diag_new, diag_old),
        f"{case_name}_theta_rel_l2": _rel_l2(out_new, out_old),
        f"{case_name}_theta_max_abs": _max_abs(out_new, out_old),
        f"{case_name}_theta_min": float(np.min(_asnumpy(out_new))),
        f"{case_name}_theta_max": float(np.max(_asnumpy(out_new))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-core", default=None, help="Dotted module or path for trusted runner_core")
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=48)
    parser.add_argument("--du", type=float, default=2.0 / 63.0)
    parser.add_argument("--dv", type=float, default=2.0 / 48.0)
    parser.add_argument("--dz", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=7.5e-4)
    parser.add_argument("--b", type=float, default=1.718606658)
    parser.add_argument("--bi", type=float, default=2.5)
    parser.add_argument("--mobility", type=float, default=1.0)
    parser.add_argument("--theta-bc", type=float, default=0.0)
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=1.55)
    parser.add_argument("--picard-iters", type=int, default=4)
    parser.add_argument("--picard-tol-up", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--tol", type=float, default=0.0, help="maximum allowed rel/max discrepancy; default exact")
    parser.add_argument("--cases", default="true_noz,predictor,gammaz,quasistatic", help="comma-separated cases")
    args = parser.parse_args()

    _ensure_src_on_path()
    old = _load_runner_core(args.runner_core)
    from lc_soliton.theta_engine import td_zstack as new

    case_specs = {
        "true_noz": dict(true_td=True, gamma_z=0.0, predictor_only=False),
        "predictor": dict(true_td=True, gamma_z=0.0, predictor_only=True),
        "gammaz": dict(true_td=True, gamma_z=0.04, predictor_only=False),
        "quasistatic": dict(true_td=False, gamma_z=0.04, predictor_only=False),
    }
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]

    metrics: dict[str, float] = {}
    for case in selected:
        if case not in case_specs:
            raise ValueError(f"unknown case {case!r}; choices={sorted(case_specs)}")
        metrics.update(_run_case(old, new, args=args, case_name=case, **case_specs[case]))

    print("46_theta_td_slice_runner_core_equivalence")
    print(f"  grid              : {args.Nx} x {args.Ny}")
    print(f"  dt                : {args.dt:g}")
    print(f"  b                 : {args.b:.9g}")
    print(f"  bi                : {args.bi:.9g}")
    print(f"  cases             : {','.join(selected)}")
    for k, v in metrics.items():
        if k.endswith("theta_min") or k.endswith("theta_max"):
            print(f"  {k:<28}: {v:.9g}")
        else:
            print(f"  {k:<28}: {v:.16e}")

    tol = float(args.tol)
    check = {k: v for k, v in metrics.items() if not (k.endswith("theta_min") or k.endswith("theta_max"))}
    failures = {k: v for k, v in check.items() if v > tol}
    if failures:
        raise AssertionError(f"td_zstack differs from runner_core beyond tol={tol}: {failures}")

    print("46_theta_td_slice_runner_core_equivalence: PASS")


if __name__ == "__main__":
    main()
