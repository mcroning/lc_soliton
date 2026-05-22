from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pde_defect_ratio_slice(ctx, theta_k, I_k, theta_prev, theta_next, *, eps=1e-12):
    """
    Compute local fractional PDE defect:

        eta = |R| / ( |L_xy| + |L_z| + |drive| + eps )

    Interpretable as local fractional PDE imbalance.
    """
    cp = ctx.xp

    th = theta_k.astype(cp.float64, copy=False)
    I = I_k.astype(cp.float64, copy=False)
    tp = theta_prev.astype(cp.float64, copy=False)
    tn = theta_next.astype(cp.float64, copy=False)

    du2 = cp.float64(ctx.du) ** 2
    dv2 = cp.float64(ctx.dv) ** 2
    dz2 = cp.float64(ctx.dz) ** 2

    # L_xy: Dirichlet x, periodic y
    yp = cp.roll(th, -1, axis=1)
    ym = cp.roll(th, +1, axis=1)

    Lxy = cp.zeros_like(th)
    Lxy[1:-1, :] = (
        (th[2:, :] - 2 * th[1:-1, :] + th[:-2, :]) / du2
        +
        (yp[1:-1, :] - 2 * th[1:-1, :] + ym[1:-1, :]) / dv2
    )

    if float(ctx.theta_z_gamma) == 0.0:
        Lz = cp.zeros_like(th)
    else:
        Lz = cp.float64(ctx.theta_z_gamma) * (tn - 2 * th + tp) / dz2
        Lz[0, :] = 0.0
        Lz[-1, :] = 0.0

    drive = (cp.float64(ctx.b) + cp.float64(ctx.bi) * I) * cp.sin(2 * th)

    R = Lxy + Lz + drive
    denom = cp.abs(Lxy) + cp.abs(Lz) + cp.abs(drive) + cp.float64(eps)

    eta = cp.abs(R) / denom

    # Dirichlet rows are prescribed, not solved
    eta[0, :] = 0.0
    eta[-1, :] = 0.0

    return eta, R, dict(Lxy=Lxy, Lz=Lz, drive=drive)


def defect_stats(ctx, eta, I_k=None):
    cp = ctx.xp

    e = eta[1:-1, :].astype(cp.float64, copy=False)

    out = dict(
        eta_max=float(cp.max(e).get()),
        eta_rms=float(cp.sqrt(cp.mean(e**2)).get()),
        eta_p95=float(cp.percentile(e, 95).get()),
        eta_p99=float(cp.percentile(e, 99).get()),
    )

    if I_k is not None:
        w = I_k[1:-1, :].astype(cp.float64, copy=False)
        wmax = cp.max(w)
        if float(wmax.get()) > 0:
            w = w / wmax
            out["eta_rms_Iweighted"] = float(
                cp.sqrt(cp.sum(w * e**2) / cp.maximum(cp.sum(w), 1e-30)).get()
            )
        else:
            out["eta_rms_Iweighted"] = None

    return out


def defect_report_slice(ctx, theta_full, I_mid_store, k):
    tp = theta_full[k - 1] if k > 0 else theta_full[k]
    tn = theta_full[k + 1] if k < ctx.Nz - 1 else theta_full[k]

    eta, R, terms = pde_defect_ratio_slice(
        ctx,
        theta_full[k],
        I_mid_store[k],
        tp,
        tn,
    )

    stats = defect_stats(ctx, eta, I_mid_store[k])
    stats.update(
        dict(
            k=int(k),
            z_um=float(k * ctx.dz),
            interpretation="eta is local fractional PDE imbalance: 0.001 = 0.1%, 0.01 = 1%",
        )
    )

    return eta, R, terms, stats


def defect_report_z(ctx, theta_full, I_mid_store, *, k_stride=1):
    reports = []

    for k in range(0, int(ctx.Nz), int(k_stride)):
        _, _, _, stats = defect_report_slice(ctx, theta_full, I_mid_store, k)
        reports.append(stats)

    return reports


def plot_defect_map(ctx, eta, *, title="PDE defect ratio eta", path=None, vmax_percentile=99):
    import matplotlib.pyplot as plt

    A = eta.get()
    x = ctx.x.get()
    y = ctx.y.get()

    vmax = np.percentile(A[1:-1, :], vmax_percentile)
    vmax = max(vmax, 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        A.T,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="fractional PDE defect")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_defect_trace(reports, *, path=None):
    import matplotlib.pyplot as plt

    z = np.array([r["z_um"] for r in reports])
    rms = np.array([r["eta_rms"] for r in reports])
    iw = np.array([r["eta_rms_Iweighted"] for r in reports])
    p99 = np.array([r["eta_p99"] for r in reports])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(z, rms, label="RMS")
    ax.semilogy(z, iw, label="I-weighted RMS")
    ax.semilogy(z, p99, label="p99")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("fractional PDE defect")
    ax.set_title("PDE satisfaction quality vs z")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=160)
        plt.close(fig)
    else:
        plt.show()

    return fig


def save_defect_diagnostics(ctx, res_z, run_dir, *, klist=None, k_stride=1):
    run_dir = Path(run_dir)
    outdir = run_dir / "pde_defect_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    reports = defect_report_z(
        ctx,
        res_z.theta_full,
        res_z.I_mid_store,
        k_stride=k_stride,
    )

    plot_defect_trace(
        reports,
        path=outdir / "pde_defect_trace.png",
    )

    if klist is None:
        klist = [0, ctx.Nz // 4, ctx.Nz // 2, 3 * ctx.Nz // 4, ctx.Nz - 1]

    slice_stats = []
    for k in klist:
        eta, R, terms, stats = defect_report_slice(
            ctx,
            res_z.theta_full,
            res_z.I_mid_store,
            int(k),
        )

        plot_defect_map(
            ctx,
            eta,
            title=f"PDE defect ratio eta, k={k}, z={k*ctx.dz:.1f} um",
            path=outdir / f"pde_defect_k{k:04d}.png",
        )

        slice_stats.append(stats)

    summary = dict(
        definition="eta = |R| / (|Lxy| + |Lz| + |drive| + eps)",
        interpretation=dict(
            eta_1e_minus_3="0.1% local PDE imbalance",
            eta_1e_minus_2="1% local PDE imbalance",
            eta_1e_minus_1="10% local PDE imbalance",
        ),
        z_reports=reports,
        slice_stats=slice_stats,
    )

    (outdir / "pde_defect_summary.json").write_text(json.dumps(summary, indent=2))

    return summary