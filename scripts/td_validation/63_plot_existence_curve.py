#!/usr/bin/env python
"""63: plot clean LC soliton existence-curve CSVs.

Input is the CSV written by 58_existence_curve_workflow_smoke.py.

Example
-------
python scripts/td_validation/63_plot_existence_curve.py \
    runs/existence_curve_512_updown_from_0p5.csv

Outputs PNG figures and a compact PDF report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _direction_groups(df: pd.DataFrame):
    if "continuation_direction" not in df.columns:
        yield "all", df
        return

    seen = False
    for name in ("up", "down"):
        sub = df[df["continuation_direction"].astype(str) == name]
        if len(sub):
            seen = True
            yield name, sub

    other = df[~df["continuation_direction"].astype(str).isin(["up", "down"])]
    if len(other) or not seen:
        yield "other", other if len(other) else df


def _plot_xy(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    pdf: PdfPages,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label, sub in _direction_groups(df):
        if x not in sub.columns or y not in sub.columns:
            continue
        ss = sub[[x, y]].dropna()
        if len(ss) == 0:
            continue
        ax.plot(ss[x], ss[y], marker="o", label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_widths(df: pd.DataFrame, *, out_path: Path, pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label, sub in _direction_groups(df):
        for y, suffix in (("sx_um", "sx"), ("sy_um", "sy")):
            if "P_req_mW" not in sub.columns or y not in sub.columns:
                continue
            ss = sub[["P_req_mW", y]].dropna()
            if len(ss):
                ax.plot(ss["P_req_mW"], ss[y], marker="o", label=f"{label} {suffix}")

    ax.set_title("Beam widths vs power")
    ax.set_xlabel("Requested power (mW)")
    ax.set_ylabel("Width (um)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    pdf.savefig(fig)
    plt.close(fig)


def _plot_summary_table(df: pd.DataFrame, *, out_path: Path, pdf: PdfPages) -> None:
    cols = [
        "P_req_mW",
        "seed_power",
        "continuation_direction",
        "continuation",
        "converged",
        "outer_steps",
        "beta",
        "theta_max",
        "Imax",
        "sx_um",
        "sy_um",
        "residual_rms",
        "residual_max",
        "field_rel",
    ]
    cols = [c for c in cols if c in df.columns]
    show = df[cols].copy()

    for c in show.columns:
        if c in ("continuation_direction", "continuation", "converged"):
            continue
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.5g}")

    fig, ax = plt.subplots(figsize=(12, max(3.0, 0.38 * len(show) + 1.2)))
    ax.axis("off")
    ax.set_title("Existence curve table", pad=12)
    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pdf", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_dir = args.out_dir or csv_path.with_suffix("").with_name(csv_path.stem + "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = args.pdf or (out_dir / f"{csv_path.stem}_plots.pdf")

    df = pd.read_csv(csv_path)
    df = _to_num(
        df,
        [
            "P_req_mW",
            "seed_power",
            "outer_steps",
            "P_out",
            "beta",
            "theta_max",
            "Imax",
            "sx_um",
            "sy_um",
            "residual_rms",
            "residual_max",
            "field_rel",
            "elapsed_s",
        ],
    )

    with PdfPages(pdf_path) as pdf:
        _plot_summary_table(df, out_path=out_dir / "00_table.png", pdf=pdf)

        _plot_xy(df, x="P_req_mW", y="beta",
                 title="Beta vs power", xlabel="Requested power (mW)", ylabel="Beta",
                 out_path=out_dir / "01_beta_vs_power.png", pdf=pdf)
        _plot_xy(df, x="P_req_mW", y="theta_max",
                 title="Theta max vs power", xlabel="Requested power (mW)", ylabel="Theta max",
                 out_path=out_dir / "02_theta_max_vs_power.png", pdf=pdf)
        _plot_xy(df, x="P_req_mW", y="Imax",
                 title="Peak intensity vs power", xlabel="Requested power (mW)", ylabel="Imax",
                 out_path=out_dir / "03_Imax_vs_power.png", pdf=pdf)
        _plot_widths(df, out_path=out_dir / "04_widths_vs_power.png", pdf=pdf)
        _plot_xy(df, x="P_req_mW", y="residual_rms",
                 title="Residual RMS vs power", xlabel="Requested power (mW)", ylabel="Residual RMS",
                 out_path=out_dir / "05_residual_rms_vs_power.png", pdf=pdf, logy=True)
        _plot_xy(df, x="P_req_mW", y="field_rel",
                 title="Field update vs power", xlabel="Requested power (mW)", ylabel="Field relative update",
                 out_path=out_dir / "06_field_rel_vs_power.png", pdf=pdf, logy=True)
        _plot_xy(df, x="P_req_mW", y="outer_steps",
                 title="Outer iterations vs power", xlabel="Requested power (mW)", ylabel="Outer iterations",
                 out_path=out_dir / "07_outer_steps_vs_power.png", pdf=pdf)
        _plot_xy(df, x="theta_max", y="beta",
                 title="Beta vs theta max", xlabel="Theta max", ylabel="Beta",
                 out_path=out_dir / "08_beta_vs_theta_max.png", pdf=pdf)

    print("63_plot_existence_curve")
    print(f"  input  : {csv_path}")
    print(f"  outdir : {out_dir}")
    print(f"  pdf    : {pdf_path}")
    print("63_plot_existence_curve: PASS")


if __name__ == "__main__":
    main()
