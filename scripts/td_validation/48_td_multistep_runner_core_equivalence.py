#!/usr/bin/env python3
"""Multi-step TD z-stack equivalence against trusted runner_core.

This test starts from the multi-step TD comparison validated in test 47 and
repeats it for several physical-time steps.  It is intentionally written by
reusing the trusted runner_core orchestration block and replacing only the
extracted numerical kernels with the new package modules.

Each TD step resets the optical launch field, copies the previous theta stack
as theta_ref, performs the old half-step/z-loop/half-step optical pass, and
updates theta_full[k] from the midpoint intensity.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    for p in (str(root), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_runner_core(path_or_module: str | None):
    _ensure_src_on_path()
    if path_or_module is None:
        return importlib.import_module("lc_soliton.validated_core.runner_core")

    if (
        "/" not in path_or_module
        and "\\" not in path_or_module
        and not path_or_module.endswith(".py")
    ):
        return importlib.import_module(path_or_module)

    p = Path(path_or_module).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"runner_core path does not exist: {p}")

    # If possible, load as package module to support relative imports.
    root = Path.cwd().resolve()
    try:
        rel = p.relative_to(root / "src")
        modname = ".".join(rel.with_suffix("").parts)
        return importlib.import_module(modname)
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location("_trusted_runner_core", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _asnumpy(a: Any) -> np.ndarray:
    if hasattr(a, "get"):
        return np.asarray(a.get())
    try:
        from lc_soliton.core.backend import asnumpy
        return np.asarray(asnumpy(a))
    except Exception:
        return np.asarray(a)


def _rel_l2(a: Any, b: Any) -> float:
    aa = _asnumpy(a)
    bb = _asnumpy(b)
    den = np.linalg.norm(bb.ravel()) + 1e-300
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _max_abs(a: Any, b: Any) -> float:
    return float(np.max(np.abs(_asnumpy(a) - _asnumpy(b))))


def _scalar(x: Any) -> float:
    if hasattr(x, "get"):
        return float(x.get())
    return float(x)


def _make_ctx(xp: Any, *, Nx: int, Ny: int, Nz: int, dz: float, dt: float, true_td: bool):
    dx = 75.0 / Nx
    dy = 100.0 / Ny
    du = 2.0 / (Nx - 1)
    dv = 2.0 * (100.0 / 75.0) / Ny
    lm = 0.633
    ne = 1.7
    no = 1.5
    refin = 1.5
    kout = 2.0 * np.pi / lm

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    fxy2 = xp.asarray(FX * FX + FY * FY)

    return SimpleNamespace(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=float(dx),
        dy=float(dy),
        du=float(du),
        dv=float(dv),
        dz=float(dz),
        dt=float(dt),
        lm=float(lm),
        ne=float(ne),
        no=float(no),
        refin=float(refin),
        kout=float(kout),
        fxy2=fxy2,
        _h_cache={},
        windowxy=None,
        coh=False,
        b=1.718606658,
        bi=2.5,
        mobility=1.0,
        theta_z_gamma=0.025,
        theta_bc=0.0,
        theta_clamp=(0.0, 1.55),
        picard_iters=4,
        picard_tol_up=1e-6,
        true_td_predictor_only=False,
        true_td_full_pred_optics=False,
        td_theta_relax_steps=2,
        td_theta_relax_omega=0.8,
        dn_max_est=0.02,
        dz_opt_max_phi=0.3,
        max_substeps=8,
    )


def _initial_amp(xp: Any, ctx: Any) -> Any:
    x = np.linspace(-37.5, 37.5, ctx.Nx, endpoint=True)
    y = np.linspace(-50.0, 50.0, ctx.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    w = 4.0
    g0 = np.exp(-((X + 1.2) ** 2 + (Y - 0.8) ** 2) / (2 * w * w))
    g1 = 0.27 * np.exp(-((X - 2.5) ** 2 + (Y + 1.7) ** 2) / (2 * (1.25*w) ** 2))
    phase = np.exp(1j * 0.04 * X)
    amp = np.stack([g0 * phase, g1 * np.conj(phase)], axis=0).astype(np.complex64)
    amp /= np.sqrt(np.sum(np.abs(amp) ** 2) * ctx.dx * ctx.dy)
    return xp.asarray(amp.astype(np.complex64))


def _initial_theta_full(xp: Any, ctx: Any) -> Any:
    x = np.linspace(-1.0, 1.0, ctx.Nx, endpoint=True)
    y = np.linspace(-1.0, 1.0, ctx.Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    base = 0.45 * (1 - X * X)
    ripple = 0.017 * np.cos(2 * np.pi * Y) * (1 - X * X)
    zmod = np.linspace(-0.02, 0.02, ctx.Nz)[:, None, None]
    th = base[None, :, :] + ripple[None, :, :] + zmod
    th[:, 0, :] = 0.0
    th[:, -1, :] = 0.0
    return xp.asarray(th.astype(np.float32))


def _run_old_one_zpass(old: Any, ctx: Any, amp0: Any, theta0: Any, *, dt: float, true_td: bool, Nsub: int, dz_sub: float, h_sub: Any, h_half: Any):
    xp = old.cp
    amp = xp.array(amp0, copy=True)
    theta_full = xp.array(theta0, copy=True)
    theta_ref = theta_full.copy()

    I_b = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)
    I_a = xp.empty_like(I_b)
    I_mid = xp.empty_like(I_b)
    I_mid_store = xp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=xp.float32)

    Ahat = xp.empty_like(amp)
    plan_f = None
    plan_i = None

    if true_td:
        a_ie, off, diag, lam_y = old.prepare_ie_ky_operator(dt=dt, mobility=ctx.mobility, du=ctx.du, dv=ctx.dv, Ny=ctx.Ny)
        s = None
    else:
        s, off, diag, lam_y = old.prepare_cn_ky_operator(dt=dt, mobility=ctx.mobility, du=ctx.du, dv=ctx.dv, Ny=ctx.Ny)
        a_ie = None

    old.hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

    for k in range(ctx.Nz):
        old.intens_into(I_b, amp, coh=ctx.coh)
        theta_use = theta_ref[k]
        tp = theta_ref[k - 1] if k > 0 else theta_ref[k]
        tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else theta_ref[k]
        old.td_get_slice_mid_intensity(
            ctx, amp, theta_use, I_b, I_a, I_mid,
            Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
            Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
            frozen_I_mid_k=None,
        )
        I_mid_store[k] = I_mid
        theta_full[k] = old.td_update_theta_from_midintensity(
            ctx, theta_ref[k], tp, tn, I_mid,
            dt_j=dt, true_td=true_td,
            a_ie=a_ie, s=s, off=off, diag=diag, lam_y=lam_y,
            Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
            amp_for_pred=None, I_b=I_b, I_a=I_a, Ahat=Ahat,
            plan_f=plan_f, plan_i=plan_i,
        )

    old.hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)
    return amp, theta_full, I_mid_store


def _run_new_one_zpass(new_opt: Any, new_th: Any, ctx: Any, amp0: Any, theta0: Any, *, dt: float, true_td: bool, Nsub: int, dz_sub: float, h_sub: Any, h_half: Any):
    xp = new_opt.cp if hasattr(new_opt, "cp") else np
    amp = xp.array(amp0, copy=True)
    theta_full = xp.array(theta0, copy=True)
    theta_ref = theta_full.copy()

    I_b = xp.empty((ctx.Nx, ctx.Ny), dtype=xp.float32)
    I_a = xp.empty_like(I_b)
    I_mid = xp.empty_like(I_b)
    I_mid_store = xp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=xp.float32)

    Ahat = xp.empty_like(amp)
    plan_f = None
    plan_i = None

    if true_td:
        prepared = new_th.prepare_ie_ky_operator(dt=dt, mobility=ctx.mobility, du=ctx.du, dv=ctx.dv, Ny=ctx.Ny, xp=xp, dtype=xp.float32)
    else:
        prepared = new_th.prepare_cn_ky_operator(dt=dt, mobility=ctx.mobility, du=ctx.du, dv=ctx.dv, Ny=ctx.Ny, xp=xp, dtype=xp.float32)
    a_or_s, off, diag, lam_y = prepared
    a_ie = a_or_s if true_td else None
    s = None if true_td else a_or_s

    new_opt.hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)

    for k in range(ctx.Nz):
        new_opt.intens_into(I_b, amp, coh=ctx.coh)
        theta_use = theta_ref[k]
        tp = theta_ref[k - 1] if k > 0 else theta_ref[k]
        tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else theta_ref[k]
        new_opt.td_get_slice_mid_intensity(
            ctx, amp, theta_use, I_b, I_a, I_mid,
            Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
            Ahat=Ahat, plan_f=plan_f, plan_i=plan_i,
            frozen_I_mid_k=None,
        )
        I_mid_store[k] = I_mid
        theta_full[k] = new_th.td_update_theta_from_midintensity(
            ctx, theta_ref[k], tp, tn, I_mid,
            dt_j=dt, true_td=true_td,
            a_ie=a_ie, s=s, off=off, diag=diag, lam_y=lam_y,
            Nsub=Nsub, dz_sub=dz_sub, h_sub=h_sub,
            amp_for_pred=None, I_b=I_b, I_a=I_a, Ahat=Ahat,
            plan_f=plan_f, plan_i=plan_i,
            xp=xp,
        )

    new_opt.hop_linear_inplace(amp, h_half, ctx.windowxy, Ahat, plan_f=plan_f, plan_i=plan_i)
    return amp, theta_full, I_mid_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner-core", default=None, help="trusted runner_core module path or dotted module")
    ap.add_argument("--Nx", type=int, default=64)
    ap.add_argument("--Ny", type=int, default=48)
    ap.add_argument("--Nz", type=int, default=9)
    ap.add_argument("--Nt", type=int, default=6)
    ap.add_argument("--dz-um", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=7.5e-4)
    ap.add_argument("--tol", type=float, default=2e-7)
    ap.add_argument("--quasistatic", action="store_true", help="compare true_td=False branch")
    args = ap.parse_args()

    old = _load_runner_core(args.runner_core)
    _ensure_src_on_path()
    from lc_soliton.optics_engine import splitstep as new_opt
    from lc_soliton.theta_engine import td_zstack as new_th

    xp = old.cp
    ctx_old = _make_ctx(xp, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, dz=args.dz_um, dt=args.dt, true_td=not args.quasistatic)
    ctx_new = _make_ctx(xp, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, dz=args.dz_um, dt=args.dt, true_td=not args.quasistatic)

    # Choose/cross-check kernels and substeps using trusted convention.
    Nsub_old, dz_sub_old, phi_est = old.choose_optics_substeps(
        ctx_old.dz,
        kout=ctx_old.kout,
        dn_max_est=ctx_old.dn_max_est,
        dz_opt_max_phi=ctx_old.dz_opt_max_phi,
        max_substeps=ctx_old.max_substeps,
    )
    Nsub_new, dz_sub_new, _ = new_opt.choose_optics_substeps(
        ctx_new.dz,
        kout=ctx_new.kout,
        dn_max_est=ctx_new.dn_max_est,
        dz_opt_max_phi=ctx_new.dz_opt_max_phi,
        max_substeps=ctx_new.max_substeps,
    )
    assert Nsub_old == Nsub_new
    assert abs(dz_sub_old - dz_sub_new) < 1e-15

    h_sub_old = old.get_h_for_dz(ctx_old, dz_sub_old)
    h_half_old = old.get_h_for_dz(ctx_old, 0.5 * ctx_old.dz)
    h_sub_new = new_opt.get_h_for_dz(ctx_new, dz_sub_new)
    h_half_new = new_opt.get_h_for_dz(ctx_new, 0.5 * ctx_new.dz)

    amp0 = _initial_amp(xp, ctx_old)
    theta0 = _initial_theta_full(xp, ctx_old)

    true_td = not bool(args.quasistatic)

    old_theta = xp.array(theta0, copy=True)
    new_theta = xp.array(theta0, copy=True)
    old_amp = xp.array(amp0, copy=True)
    new_amp = xp.array(amp0, copy=True)
    old_I_mid = None
    new_I_mid = None

    step_rows = []
    max_metrics = {
        "amp_rel_l2": 0.0,
        "amp_max_abs": 0.0,
        "theta_rel_l2": 0.0,
        "theta_max_abs": 0.0,
        "I_mid_rel_l2": 0.0,
        "I_mid_max_abs": 0.0,
        "power_abs_err": 0.0,
    }

    for jt in range(1, int(args.Nt) + 1):
        old_amp, old_theta, old_I_mid = _run_old_one_zpass(
            old, ctx_old, amp0, old_theta,
            dt=args.dt, true_td=true_td, Nsub=Nsub_old, dz_sub=dz_sub_old,
            h_sub=h_sub_old, h_half=h_half_old,
        )
        new_amp, new_theta, new_I_mid = _run_new_one_zpass(
            new_opt, new_th, ctx_new, amp0, new_theta,
            dt=args.dt, true_td=true_td, Nsub=Nsub_new, dz_sub=dz_sub_new,
            h_sub=h_sub_new, h_half=h_half_new,
        )

        old_power = _scalar(xp.sum(old.intens(old_amp, ctx_old.coh)) * ctx_old.dx * ctx_old.dy)
        new_power = _scalar(xp.sum(new_opt.intens(new_amp, ctx_new.coh)) * ctx_new.dx * ctx_new.dy)
        row = {
            "jt": jt,
            "amp_rel_l2": _rel_l2(new_amp, old_amp),
            "amp_max_abs": _max_abs(new_amp, old_amp),
            "theta_rel_l2": _rel_l2(new_theta, old_theta),
            "theta_max_abs": _max_abs(new_theta, old_theta),
            "I_mid_rel_l2": _rel_l2(new_I_mid, old_I_mid),
            "I_mid_max_abs": _max_abs(new_I_mid, old_I_mid),
            "old_power": old_power,
            "new_power": new_power,
            "power_abs_err": abs(new_power - old_power),
            "old_theta_max": float(np.max(_asnumpy(old_theta))),
            "new_theta_max": float(np.max(_asnumpy(new_theta))),
        }
        step_rows.append(row)
        for k in max_metrics:
            max_metrics[k] = max(max_metrics[k], float(row[k]))

    metrics = {
        "kernel_sub_rel_l2": _rel_l2(h_sub_new, h_sub_old),
        "kernel_half_rel_l2": _rel_l2(h_half_new, h_half_old),
        **{f"max_{k}": v for k, v in max_metrics.items()},
        "final_old_power": step_rows[-1]["old_power"],
        "final_new_power": step_rows[-1]["new_power"],
        "final_old_theta_max": step_rows[-1]["old_theta_max"],
        "final_new_theta_max": step_rows[-1]["new_theta_max"],
    }

    print("48_td_multistep_runner_core_equivalence")
    print(f"  grid              : {args.Nx} x {args.Ny} x {args.Nz}")
    print(f"  branch            : {'quasistatic' if args.quasistatic else 'true_td'}")
    print(f"  Nt                : {args.Nt}")
    print(f"  dt                : {args.dt:g}")
    print(f"  dz_um             : {args.dz_um:g}")
    print(f"  Nsub              : {Nsub_old}")
    print(f"  dz_sub_um         : {dz_sub_old:g}")
    print(f"  phi_est           : {phi_est:g}")
    print("  per_step:")
    print("    jt   amp_rel       theta_rel     theta_abs     I_mid_rel     power_err")
    for r in step_rows:
        print(
            f"    {r['jt']:2d}  {r['amp_rel_l2']:.3e}  {r['theta_rel_l2']:.3e}  "
            f"{r['theta_max_abs']:.3e}  {r['I_mid_rel_l2']:.3e}  {r['power_abs_err']:.3e}"
        )
    for k, v in metrics.items():
        print(f"  {k:<24}: {v:.16e}" if isinstance(v, float) else f"  {k:<24}: {v}")

    failures = {
        k: v for k, v in metrics.items()
        if (k.endswith("rel_l2") or k.endswith("max_abs") or k.endswith("power_abs_err"))
        and float(v) > args.tol
    }
    if failures:
        raise AssertionError(f"multi-step TD differs from runner_core beyond tol={args.tol}: {failures}")
    print("48_td_multistep_runner_core_equivalence: PASS")


if __name__ == "__main__":
    main()
