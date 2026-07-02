#!/usr/bin/env python3
"""45_optics_slice_runner_core_equivalence.py

Validate the extracted optics split-step module against the trusted
``runner_core.py`` implementation for one TD z-slice.

This is intentionally a narrow oracle test.  It compares exactly the optics
operations that the production TD z-stack port will rely on:

- intensity before the slice
- angular-spectrum kernel for the substep dz
- LC nonlinear phase + linear hop ordering
- intensity after the slice
- midpoint intensity
- advanced optical field

The test does not update theta and does not run the full TD workflow.
"""

from __future__ import annotations
def _load_runner_core(path_or_module):
    import importlib
    import importlib.util
    from pathlib import Path

    if path_or_module is None:
        return importlib.import_module("lc_soliton.validated_core.runner_core")

    # Dotted module path, e.g. lc_soliton.validated_core.runner_core
    if (
        "/" not in path_or_module
        and "\\" not in path_or_module
        and not path_or_module.endswith(".py")
    ):
        return importlib.import_module(path_or_module)

    # File path fallback
    p = Path(path_or_module)
    if not p.exists():
        raise FileNotFoundError(f"runner_core path does not exist: {p}")

    spec = importlib.util.spec_from_file_location("runner_core_ref", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
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





def _asnumpy(a):
    try:
        import cupy as cupy  # type: ignore

        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _rel_l2(a, b) -> float:
    aa = _asnumpy(a).astype(np.complex128 if np.iscomplexobj(_asnumpy(a)) else np.float64)
    bb = _asnumpy(b).astype(np.complex128 if np.iscomplexobj(_asnumpy(b)) else np.float64)
    return float(np.linalg.norm((aa - bb).ravel()) / (np.linalg.norm(bb.ravel()) + 1e-300))


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(_asnumpy(a) - _asnumpy(b))))


def _make_ctx(cp, *, Nx: int, Ny: int, xaper_um: float, yaper_um: float, dz_um: float, wavelength_um: float, n_ref: float, ne: float, no: float, coherent: bool):
    dx = float(xaper_um) / (Nx - 1)
    dy = float(yaper_um) / Ny
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    fxy2_np = fx[:, None] ** 2 + fy[None, :] ** 2
    fxy2 = cp.asarray(fxy2_np, dtype=cp.float32) if hasattr(cp, "asarray") else fxy2_np.astype(np.float32)
    return SimpleNamespace(
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        dz=float(dz_um),
        dz_um=float(dz_um),
        lm=float(wavelength_um),
        refin=float(n_ref),
        ne=float(ne),
        no=float(no),
        kout=float(2.0 * np.pi / wavelength_um),
        fxy2=fxy2,
        windowxy=None,
        coh=bool(coherent),
        _h_cache={},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-core", default=None, help="Path to trusted runner_core.py if not importable")
    parser.add_argument("--Nx", type=int, default=64)
    parser.add_argument("--Ny", type=int, default=48)
    parser.add_argument("--xaper-um", type=float, default=75.0)
    parser.add_argument("--yaper-um", type=float, default=100.0)
    parser.add_argument("--dz-um", type=float, default=5.0)
    parser.add_argument("--Nsub", type=int, default=3)
    parser.add_argument("--wavelength-um", type=float, default=0.633)
    parser.add_argument("--n-ref", type=float, default=1.5)
    parser.add_argument("--ne", type=float, default=1.7)
    parser.add_argument("--no", type=float, default=1.5)
    parser.add_argument("--coherent", action="store_true")
    parser.add_argument("--tol", type=float, default=0.0, help="relative tolerance; default expects exact/roundoff equality")
    args = parser.parse_args()

    _ensure_src_on_path()
    old = _load_runner_core(args.runner_core)

    from lc_soliton.optics_engine import splitstep as new

    # Use the same backend object as runner_core so CPU/GPU behavior matches.
    cp = old.cp
    rng = np.random.default_rng(45)

    ctx_old = _make_ctx(cp, Nx=args.Nx, Ny=args.Ny, xaper_um=args.xaper_um, yaper_um=args.yaper_um,
                        dz_um=args.dz_um, wavelength_um=args.wavelength_um, n_ref=args.n_ref,
                        ne=args.ne, no=args.no, coherent=args.coherent)
    ctx_new = _make_ctx(cp, Nx=args.Nx, Ny=args.Ny, xaper_um=args.xaper_um, yaper_um=args.yaper_um,
                        dz_um=args.dz_um, wavelength_um=args.wavelength_um, n_ref=args.n_ref,
                        ne=args.ne, no=args.no, coherent=args.coherent)

    amp_np = (
        rng.normal(size=(2, args.Nx, args.Ny))
        + 1j * rng.normal(size=(2, args.Nx, args.Ny))
    ).astype(np.complex64)
    # Make component powers unequal so coherent/incoherent paths are both sensitive.
    amp_np[1] *= np.complex64(0.37 - 0.11j)

    x = np.linspace(-args.xaper_um / 2, args.xaper_um / 2, args.Nx, dtype=np.float32)
    y = (np.arange(args.Ny, dtype=np.float32) - args.Ny / 2) * (args.yaper_um / args.Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    theta_np = (
        0.52
        + 0.22 * np.exp(-(X**2 + Y**2) / (2 * 8.0**2))
        + 0.03 * np.sin(2 * np.pi * Y / args.yaper_um)
    ).astype(np.float32)

    amp_old = cp.asarray(amp_np.copy())
    amp_new = cp.asarray(amp_np.copy())
    theta = cp.asarray(theta_np)

    dz_sub = float(args.dz_um) / int(args.Nsub)
    h_old = old.get_h_for_dz(ctx_old, dz_sub)
    h_new = new.get_h_for_dz(ctx_new, dz_sub)

    I_b_old = cp.empty((args.Nx, args.Ny), dtype=cp.float32)
    I_a_old = cp.empty_like(I_b_old)
    I_mid_old = cp.empty_like(I_b_old)
    I_b_new = cp.empty_like(I_b_old)
    I_a_new = cp.empty_like(I_b_old)
    I_mid_new = cp.empty_like(I_b_old)

    old.intens_into(I_b_old, amp_old, coh=bool(args.coherent))
    new.intens_into(I_b_new, amp_new, coh=bool(args.coherent))

    old.td_get_slice_mid_intensity(
        ctx_old,
        amp_old,
        theta,
        I_b_old,
        I_a_old,
        I_mid_old,
        Nsub=int(args.Nsub),
        dz_sub=dz_sub,
        h_sub=h_old,
        Ahat=None,
        plan_f=None,
        plan_i=None,
    )
    new.td_get_slice_mid_intensity(
        ctx_new,
        amp_new,
        theta,
        I_b_new,
        I_a_new,
        I_mid_new,
        Nsub=int(args.Nsub),
        dz_sub=dz_sub,
        h_sub=h_new,
        Ahat=None,
        plan_f=None,
        plan_i=None,
    )

    metrics = {
        "kernel_rel_l2": _rel_l2(h_new, h_old),
        "kernel_max_abs": _max_abs(h_new, h_old),
        "I_b_rel_l2": _rel_l2(I_b_new, I_b_old),
        "I_b_max_abs": _max_abs(I_b_new, I_b_old),
        "A_after_rel_l2": _rel_l2(amp_new, amp_old),
        "A_after_max_abs": _max_abs(amp_new, amp_old),
        "I_a_rel_l2": _rel_l2(I_a_new, I_a_old),
        "I_a_max_abs": _max_abs(I_a_new, I_a_old),
        "I_mid_rel_l2": _rel_l2(I_mid_new, I_mid_old),
        "I_mid_max_abs": _max_abs(I_mid_new, I_mid_old),
    }

    print("45_optics_slice_runner_core_equivalence")
    print(f"  grid              : {args.Nx} x {args.Ny}")
    print(f"  Nsub              : {args.Nsub}")
    print(f"  dz_um             : {args.dz_um:g}")
    print(f"  dz_sub_um         : {dz_sub:g}")
    print(f"  coherent          : {bool(args.coherent)}")
    for k, v in metrics.items():
        print(f"  {k:<18}: {v:.16e}")

    tol = float(args.tol)
    # Exact equality is expected for the extraction on the same backend.  Permit
    # tiny roundoff only if the caller supplied --tol.
    failures = {k: v for k, v in metrics.items() if v > tol}
    if failures:
        raise AssertionError(f"splitstep differs from runner_core beyond tol={tol}: {failures}")

    print("45_optics_slice_runner_core_equivalence: PASS")


if __name__ == "__main__":
    main()
