# lc_core/compare_runs.py
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class FieldDiffStats:
    max_abs: float
    rms: float
    rel_rms: float | None
    p95_abs: float
    p99_abs: float


@dataclass
class RunComparison:
    label_a: str
    label_b: str
    theta_stats: dict
    I_stats: dict | None
    theta_z_trace: dict
    I_z_trace: dict | None


def _xp_of(arr):
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy as cp
        return cp
    import numpy as np
    return np


def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def field_diff_stats(A, B, *, reference=None) -> dict:
    """
    Basic difference statistics for two same-shaped arrays.

    rel_rms is rms(A-B) / rms(reference), where reference defaults to B.
    """
    xp = _xp_of(A)

    A64 = A.astype(xp.float64, copy=False)
    B64 = B.astype(xp.float64, copy=False)
    D = A64 - B64

    absD = xp.abs(D)
    rms = xp.sqrt(xp.mean(D * D))

    if reference is None:
        reference = B

    R = reference.astype(xp.float64, copy=False)
    rms_ref = xp.sqrt(xp.mean(R * R))
    rel = rms / rms_ref if float(rms_ref.get() if hasattr(rms_ref, "get") else rms_ref) > 0 else None

    absD_np = _to_numpy(absD)

    stats = FieldDiffStats(
        max_abs=float(_to_numpy(xp.max(absD))),
        rms=float(_to_numpy(rms)),
        rel_rms=float(_to_numpy(rel)) if rel is not None else None,
        p95_abs=float(np.percentile(absD_np, 95)),
        p99_abs=float(np.percentile(absD_np, 99)),
    )
    return asdict(stats)


def z_trace_diff(A, B, *, dx=None, dy=None):
    """
    Per-z max/rms difference trace for arrays shaped (Nz,Nx,Ny).
    """
    xp = _xp_of(A)

    A64 = A.astype(xp.float64, copy=False)
    B64 = B.astype(xp.float64, copy=False)
    D = A64 - B64

    absD = xp.abs(D)
    max_z = xp.max(absD, axis=(1, 2))
    rms_z = xp.sqrt(xp.mean(D * D, axis=(1, 2)))

    out = dict(
        max_abs=_to_numpy(max_z).astype(float).tolist(),
        rms=_to_numpy(rms_z).astype(float).tolist(),
    )

    if dx is not None and dy is not None:
        # L2 over transverse plane for each z.
        l2_z = xp.sqrt(xp.sum(D * D, axis=(1, 2)) * xp.float64(dx * dy))
        out["l2_xy"] = _to_numpy(l2_z).astype(float).tolist()

    return out


def compare_static_td(
    ctx,
    static_result,
    td_result,
    *,
    label_static="static",
    label_td="td",
    compare_I=True,
    allow_crop=False,
):
    """
    Compare strict static z-march result with TD result.

    Expects:
        static_result.theta_full
        td_result.theta_full

    Optionally:
        static_result.I_mid_store
        td_result.I_mid_store
    """
    Nz_compare, warning = validate_comparable_shapes(
        static_result,
        td_result,
        allow_crop=allow_crop,
    )
    
    theta_static = static_result.theta_full[:Nz_compare]
    theta_td = td_result.theta_full[:Nz_compare]

    theta_stats = field_diff_stats(theta_td, theta_static, reference=theta_static)
    theta_z = z_trace_diff(theta_td, theta_static, dx=ctx.dx, dy=ctx.dy)

    I_stats = None
    I_z = None

    if (
        compare_I
        and getattr(static_result, "I_mid_store", None) is not None
        and getattr(td_result, "I_mid_store", None) is not None
    ):
    
        I_static = static_result.I_mid_store[:Nz_compare]
        I_td = td_result.I_mid_store[:Nz_compare]
    
        I_stats = field_diff_stats(
            I_td,
            I_static,
            reference=I_static,
        )
    
        I_z = z_trace_diff(
            I_td,
            I_static,
            dx=ctx.dx,
            dy=ctx.dy,
        )

    comp = RunComparison(
        label_a=label_static,
        label_b=label_td,
        theta_stats=theta_stats,
        I_stats=I_stats,
        theta_z_trace=theta_z,
        I_z_trace=I_z,
    )

    out = asdict(comp)
    out["comparison_warning"] = warning
    out["Nz_compare"] = int(Nz_compare)
    
    return out


def plot_z_trace(comparison, ctx=None, *, field="theta", path=None):
    """
    Plot per-z RMS and max absolute difference.
    """
    import matplotlib.pyplot as plt

    trace_key = f"{field}_z_trace"
    trace = comparison[trace_key]
    if trace is None:
        raise ValueError(f"No trace available for {field!r}")

    Nz = len(trace["rms"])
    if ctx is not None:
        z = np.arange(Nz) * float(ctx.dz)
        xlabel = "z (um)"
    else:
        z = np.arange(Nz)
        xlabel = "slice k"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(z, trace["rms"], label="RMS")
    ax.semilogy(z, trace["max_abs"], label="max |diff|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{field} difference")
    ax.set_title(f"{field} difference vs z")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_slice_difference(ctx, A, B, k, *, title=None, path=None, percentile=99, cmap="RdBu_r"):
    """
    Plot A[k]-B[k] as an x-y map.
    """
    import matplotlib.pyplot as plt

    D = _to_numpy(A[k] - B[k])
    x = _to_numpy(ctx.x)
    y = _to_numpy(ctx.y)

    vmax = np.percentile(np.abs(D), percentile)
    vmax = max(float(vmax), 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        D.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    if title is None:
        title = f"Difference map, k={k}, z={k*ctx.dz:.1f} um"
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

    return fig


def save_static_td_comparison(
    ctx,
    static_result,
    td_result,
    run_dir,
    *,
    klist=None,
    allow_crop=True,
):
    """
    Save static-vs-TD comparison JSON and figures into run_dir/comparisons/.
    """
    run_dir = Path(run_dir)
    outdir = run_dir / "comparisons"
    outdir.mkdir(parents=True, exist_ok=True)

    comparison = compare_static_td(
        ctx,
        static_result,
        td_result,
        allow_crop=allow_crop,
    )
    
    (outdir / "static_vs_td_summary.json").write_text(
        json.dumps(comparison, indent=2)
    )
    
    plot_z_trace(
        comparison,
        ctx=ctx,
        field="theta",
        path=outdir / "theta_diff_vs_z.png",
    )
    
    if comparison["I_z_trace"] is not None:
        plot_z_trace(
            comparison,
            ctx=ctx,
            field="I",
            path=outdir / "I_mid_diff_vs_z.png",
        )
    
    Nz_plot = int(comparison["Nz_compare"])
    
    theta_static_plot = static_result.theta_full[:Nz_plot]
    theta_td_plot = td_result.theta_full[:Nz_plot]
    
    I_static_plot = None
    I_td_plot = None
    
    if (
        getattr(static_result, "I_mid_store", None) is not None
        and getattr(td_result, "I_mid_store", None) is not None
    ):
        I_static_plot = static_result.I_mid_store[:Nz_plot]
        I_td_plot = td_result.I_mid_store[:Nz_plot]
    
    if klist is None:
        klist = [0, Nz_plot // 4, Nz_plot // 2, 3 * Nz_plot // 4, Nz_plot - 1]
    
    for k in klist:
        
        plot_slice_difference(
            ctx,
            theta_td_plot,
            theta_static_plot,
            int(k),
            title=f"TD - static theta, k={int(k)}, z={int(k)*ctx.dz:.1f} um",
            path=outdir / f"theta_diff_k{int(k):04d}.png",
        )

        if getattr(static_result, "I_mid_store", None) is not None and getattr(td_result, "I_mid_store", None) is not None:
            plot_slice_difference(
                ctx,
                td_result.I_mid_store,
                static_result.I_mid_store,
                int(k),
                title=f"TD - static I_mid, k={int(k)}, z={int(k)*ctx.dz:.1f} um",
                path=outdir / f"I_mid_diff_k{int(k):04d}.png",
            )
            
    comparison["comparison_dir"] = str(outdir)
    
    print(f"[compare_runs] saved comparison outputs to: {outdir}")
    
    return comparison

def load_saved_run_arrays(run_dir, *, xp=None):
    """
    Load saved theta_full and I_mid_store from a run directory.

    If xp is cupy, arrays are moved to GPU.
    """
    run_dir = Path(run_dir)

    theta = np.load(run_dir / "arrays" / "theta_full.npy")
    I_mid = np.load(run_dir / "arrays" / "I_mid_store.npy")

    if xp is not None:
        theta = xp.asarray(theta)
        I_mid = xp.asarray(I_mid)

    return dict(
        theta_full=theta,
        I_mid_store=I_mid,
    )


def saved_arrays_as_result(run_dir, *, xp=None):
    """
    Return a SimpleNamespace compatible with compare_static_td().
    """
    from types import SimpleNamespace

    arr = load_saved_run_arrays(run_dir, xp=xp)

    return SimpleNamespace(
        theta_full=arr["theta_full"],
        I_mid_store=arr["I_mid_store"],
    )

def validate_comparable_shapes(static_result, td_result, *, allow_crop=False):
    s_shape = tuple(static_result.theta_full.shape)
    t_shape = tuple(td_result.theta_full.shape)

    if s_shape != t_shape:
        msg = (
            "Incommensurate theta arrays: "
            f"static theta_full shape={s_shape}, TD theta_full shape={t_shape}. "
            "These runs likely have different rlen/dz/Nz or grid settings. "
            "Rerun with matching config, or pass allow_crop=True explicitly."
        )

        if not allow_crop:
            raise ValueError(msg)

        Nz = min(s_shape[0], t_shape[0])
        if s_shape[1:] != t_shape[1:]:
            raise ValueError(
                msg + " Cropping only supports different Nz with same transverse grid."
            )

        return Nz, msg

    return s_shape[0], None
