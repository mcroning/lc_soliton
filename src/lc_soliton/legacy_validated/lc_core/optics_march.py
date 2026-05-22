# lc_core/optics_march.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Any

from .optics import (
    intensity,
    total_power,
    linear_step,
    strang_step_with_fixed_theta,
)


@dataclass
class OpticsMarchResult:
    mode: str
    Nz: int
    dz_um: float
    coh: bool
    apply_window_each_step: bool
    use_fixed_theta_phase: bool

    P_start: float
    P_end: float
    rel_power_change: float | None
    Imax_start: float
    Imax_end: float

    Ixz: Any | None = None
    Iyz: Any | None = None
    xz_stride: int = 1

    power_trace: list[float] | None = None
    imax_trace: list[float] | None = None


def march_optics_only(
    ctx,
    *,
    amp0=None,
    mode: Literal["linear", "fixed_theta_strang"] = "linear",
    theta_full=None,
    apply_window_each_step: bool = False,
    store_slices: bool = True,
    xz_stride: int = 1,
    keep_final_amp: bool = True,
):
    """
    Optics-only z-march.

    This intentionally does NOT solve theta. It tests the propagation/marching,
    cache reuse, optional sponge, and slice bookkeeping.

    Parameters
    ----------
    ctx:
        LCContextGPU-like object.
    amp0:
        Optional starting field. Defaults to ctx.amp0.
    mode:
        "linear":
            repeated angular-spectrum steps only.

        "fixed_theta_strang":
            L(dz/2) -> N(theta[k], dz) -> L(dz/2)
            using theta_full[k] as a prescribed fixed theta stack.

    theta_full:
        Required for fixed_theta_strang unless ctx.theta_full is used.
    apply_window_each_step:
        Applies ctx.windowxy after each completed z step.
        This will intentionally reduce power if sponge/window < 1.
    store_slices:
        If True, stores Ixz and Iyz every xz_stride steps.
    xz_stride:
        Store every this many z steps.
    keep_final_amp:
        If True, attach final_amp to returned result as a dynamic attribute.
        This avoids forcing dataclass serialization of a CuPy array.

    Returns
    -------
    OpticsMarchResult
    """
    cp = ctx.xp

    if amp0 is None:
        amp = ctx.amp0.copy()
    else:
        amp = amp0.copy()

    if theta_full is None:
        theta_full = getattr(ctx, "theta_full", None)

    if mode == "fixed_theta_strang" and theta_full is None:
        raise ValueError("theta_full is required for mode='fixed_theta_strang'.")

    xz_stride = max(1, int(xz_stride))
    nframes = (int(ctx.Nz) + xz_stride - 1) // xz_stride if store_slices else 0

    if store_slices:
        Ixz = cp.zeros((nframes, ctx.Nz, ctx.Nx), dtype=cp.float32)
        Iyz = cp.zeros((nframes, ctx.Nz, ctx.Ny), dtype=cp.float32)
    else:
        Ixz = None
        Iyz = None

    power_trace = []
    imax_trace = []

    I_start = intensity(amp, coh=ctx.coh).astype(cp.float32, copy=False)
    P_start = total_power(I_start, ctx.dx, ctx.dy)
    Imax_start = float(cp.max(I_start).get())

    slot = 0

    for k in range(int(ctx.Nz)):
        if mode == "linear":
            amp, _info = linear_step(
                ctx,
                amp,
                float(ctx.dz),
                apply_window=bool(apply_window_each_step),
            )

        elif mode == "fixed_theta_strang":
            theta_k = theta_full[k]
            amp, _infos = strang_step_with_fixed_theta(
                ctx,
                amp,
                theta_k,
                float(ctx.dz),
                apply_window=bool(apply_window_each_step),
            )

        else:
            raise ValueError(f"Unknown optics march mode: {mode!r}")

        I = intensity(amp, coh=ctx.coh).astype(cp.float32, copy=False)
        P = total_power(I, ctx.dx, ctx.dy)
        Imax = float(cp.max(I).get())

        power_trace.append(float(P))
        imax_trace.append(float(Imax))

        if store_slices and ((k % xz_stride) == 0):
            Ixz[slot, k, :] = I[:, ctx.Ny // 2]
            Iyz[slot, k, :] = I[ctx.Nx // 2, :]
            slot += 1

    P_end = power_trace[-1] if power_trace else P_start
    Imax_end = imax_trace[-1] if imax_trace else Imax_start

    result = OpticsMarchResult(
        mode=mode,
        Nz=int(ctx.Nz),
        dz_um=float(ctx.dz),
        coh=bool(ctx.coh),
        apply_window_each_step=bool(apply_window_each_step),
        use_fixed_theta_phase=(mode == "fixed_theta_strang"),
        P_start=float(P_start),
        P_end=float(P_end),
        rel_power_change=float((P_end - P_start) / P_start) if P_start != 0 else None,
        Imax_start=float(Imax_start),
        Imax_end=float(Imax_end),
        Ixz=Ixz,
        Iyz=Iyz,
        xz_stride=int(xz_stride),
        power_trace=power_trace,
        imax_trace=imax_trace,
    )

    if keep_final_amp:
        result.final_amp = amp

    return result


def optics_march_summary(result: OpticsMarchResult) -> dict:
    """
    JSON-safe summary. Omits CuPy arrays.
    """
    d = asdict(result)
    d.pop("Ixz", None)
    d.pop("Iyz", None)

    # traces can get long; keep them in memory unless caller explicitly saves them.
    d.pop("power_trace", None)
    d.pop("imax_trace", None)

    d["has_final_amp"] = hasattr(result, "final_amp")
    d["has_Ixz"] = result.Ixz is not None
    d["has_Iyz"] = result.Iyz is not None

    if result.Ixz is not None:
        d["Ixz_shape"] = list(result.Ixz.shape)
    if result.Iyz is not None:
        d["Iyz_shape"] = list(result.Iyz.shape)

    return d


def save_optics_march_outputs(result: OpticsMarchResult, run_dir):
    """
    Save JSON summary and, if present, slice arrays as compressed NumPy files.

    This copies slice arrays to CPU, so use intentionally.
    """
    import json
    from pathlib import Path
    import numpy as np

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = optics_march_summary(result)
    (run_dir / "optics_march_summary.json").write_text(json.dumps(summary, indent=2))

    if result.Ixz is not None or result.Iyz is not None:
        out = {}
        if result.Ixz is not None:
            out["Ixz"] = result.Ixz.get()
        if result.Iyz is not None:
            out["Iyz"] = result.Iyz.get()
        np.savez_compressed(run_dir / "optics_march_slices.npz", **out)

    if result.power_trace is not None or result.imax_trace is not None:
        trace = {}
        if result.power_trace is not None:
            trace["power"] = np.asarray(result.power_trace, dtype=np.float64)
        if result.imax_trace is not None:
            trace["Imax"] = np.asarray(result.imax_trace, dtype=np.float64)
        np.savez_compressed(run_dir / "optics_march_trace.npz", **trace)

    return summary
