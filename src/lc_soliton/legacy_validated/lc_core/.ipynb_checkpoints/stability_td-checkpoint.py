from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


@dataclass
class TDStabilityMetrics:
    theta_rms: float
    theta_max_abs: float
    I_rms: float | None
    I_max_abs: float | None
    centroid_wiggle_um: float | None
    centroid_rms_um: float | None


def field_distance(A, B):
    A = _to_numpy(A).astype(np.float64, copy=False)
    B = _to_numpy(B).astype(np.float64, copy=False)
    D = A - B

    return dict(
        rms=float(np.sqrt(np.mean(D * D))),
        max_abs=float(np.max(np.abs(D))),
    )


def centroid_x_vs_z(ctx, Ixz):
    Ixz = _to_numpy(Ixz).astype(np.float64, copy=False)  # (Nz, Nx)
    x = _to_numpy(ctx.x).astype(np.float64, copy=False)

    Pz = Ixz.sum(axis=1) + 1e-300
    xc = (Ixz * x[None, :]).sum(axis=1) / Pz

    return xc


def centroid_wiggle_metrics(ctx, Ixz, *, baseline_slices=20):
    xc = centroid_x_vs_z(ctx, Ixz)

    base = np.median(xc[:baseline_slices])
    dx = xc - base

    return dict(
        centroid_x_um=xc.tolist(),
        centroid_wiggle_um=float(np.max(np.abs(dx))),
        centroid_rms_um=float(np.sqrt(np.mean(dx * dx))),
    )


def measure_td_stability(
    ctx,
    result,
    *,
    theta_ref,
    I_ref=None,
    frame=-1,
    baseline_slices=20,
):
    """
    Compare a TD result against a reference theta/I state.

    For TD results with movie stores, centroid metrics use the requested
    movie frame. For generic results, they use final I_mid_store.
    """
    theta_d = field_distance(result.theta_full, theta_ref)

    I_d = None
    if I_ref is not None and getattr(result, "I_mid_store", None) is not None:
        I_d = field_distance(result.I_mid_store, I_ref)

    centroid = None

    if hasattr(result, "stores") and getattr(result.stores, "Ixz", None) is not None:
        Ixz = result.stores.Ixz[frame]
        centroid = centroid_wiggle_metrics(
            ctx,
            Ixz,
            baseline_slices=baseline_slices,
        )
    elif getattr(result, "I_mid_store", None) is not None:
        I = _to_numpy(result.I_mid_store)
        Ixz = I[:, :, int(ctx.Ny) // 2]
        centroid = centroid_wiggle_metrics(
            ctx,
            Ixz,
            baseline_slices=baseline_slices,
        )

    metrics = TDStabilityMetrics(
        theta_rms=theta_d["rms"],
        theta_max_abs=theta_d["max_abs"],
        I_rms=I_d["rms"] if I_d is not None else None,
        I_max_abs=I_d["max_abs"] if I_d is not None else None,
        centroid_wiggle_um=centroid["centroid_wiggle_um"] if centroid is not None else None,
        centroid_rms_um=centroid["centroid_rms_um"] if centroid is not None else None,
    )

    out = asdict(metrics)

    if centroid is not None:
        out["centroid_x_um"] = centroid["centroid_x_um"]

    return out


def plot_centroid_wiggle(ctx, metrics, *, title="centroid x(z)", path=None):
    import matplotlib.pyplot as plt

    xc = np.asarray(metrics["centroid_x_um"], dtype=float)
    z = np.arange(len(xc)) * float(ctx.dz)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, xc)
    ax.set_xlabel("z (um)")
    ax.set_ylabel("x centroid (um)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

    return fig


def save_td_stability_report(ctx, result, run_dir, *, theta_ref, I_ref=None, frame=-1):
    run_dir = Path(run_dir)
    outdir = run_dir / "td_stability"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = measure_td_stability(
        ctx,
        result,
        theta_ref=theta_ref,
        I_ref=I_ref,
        frame=frame,
    )

    (outdir / "td_stability_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    if "centroid_x_um" in metrics:
        plot_centroid_wiggle(
            ctx,
            metrics,
            title="TD centroid wiggle",
            path=outdir / "centroid_x_vs_z.png",
        )

    print(f"[stability_td] saved report to: {outdir}")

    return metrics