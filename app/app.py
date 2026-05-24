"""
Minimal LC Soliton GUI prototype.
"""

import json
from PIL import Image

from pathlib import Path

import streamlit as st

from lc_soliton import (
    LCParams,
    run_static,
    load_reference_case,
    available_engine_modes,
    run_reference_case,
    available_reference_cases,
)



st.set_page_config(
    page_title="LC Soliton",
    layout="wide",
)

st.title("LC Soliton Simulator")

st.caption(
    "Available engine modes: "
    + ", ".join(available_engine_modes())
)

st.sidebar.header("Grid")

Nx = st.sidebar.slider("Nx", 32, 512, 64, step=32)
Ny = st.sidebar.slider("Ny", 32, 512, 64, step=32)
Nz = st.sidebar.slider("Nz", 4, 200, 16)

st.sidebar.header("Geometry")

xaper_um = st.sidebar.number_input("xaper (µm)", value=75.0)
yaper_um = st.sidebar.number_input("yaper (µm)", value=150.0)
dz_um = st.sidebar.number_input("dz (µm)", value=5.0)

st.sidebar.header("Physics")

b = st.sidebar.number_input("b", value=1.0)
bi = st.sidebar.number_input("bi", value=0.2)

st.sidebar.header("Beam")

waist_x_um = st.sidebar.number_input("waist x (µm)", value=8.0)
waist_y_um = st.sidebar.number_input("waist y (µm)", value=8.0)



st.header("Trusted reference case")

reference_names = available_reference_cases()
selected_reference = st.selectbox(
    "Reference case",
    reference_names,
    index=reference_names.index("strict_static_centroid_drift")
    if "strict_static_centroid_drift" in reference_names
    else 0,
)

if st.button("Show trusted strict-static reference case"):
    data = load_reference_case(selected_reference)

    st.subheader("Reference figures")

    figures = data.get("figures", {})

    xz_png = Path(figures["xz_reference"])
    centroid_png = Path(figures["centroid_reference"])

    if xz_png.exists():
        st.image(Image.open(xz_png), caption="xz intensity reference")

    if centroid_png.exists():
        st.image(Image.open(centroid_png), caption="x centroid drift reference")

    if "trusted_metrics" in data:
        st.subheader("Trusted metrics")
        st.json(data["trusted_metrics"])

    if "summary" in data:
        st.subheader("Static z summary")
        st.json(data["summary"])

    reports_path = (
        Path(data["case_dir"]) / "static_z_reports.json"
    )

    if reports_path.exists():
        import pandas as pd

        reports = json.loads(reports_path.read_text())
        st.subheader("z-march reports")
        st.dataframe(pd.DataFrame(reports))

st.subheader("Regenerate trusted reference case")

if st.button("Validate trusted reference case"):
    with st.spinner("Running trusted reference validation..."):
        try:
            summary = run_reference_case(selected_reference)
        except Exception as exc:
            st.error("Trusted reference validation failed")
            st.exception(exc)
        else:
            st.success("Trusted reference validation passed")
            st.json(summary)

run_button = st.button("Run strict static case")

if run_button:

    params = LCParams(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        xaper_um=xaper_um,
        yaper_um=yaper_um,
        dz_um=dz_um,
        b=b,
        bi=bi,
        waist_x_um=waist_x_um,
        waist_y_um=waist_y_um,
    )

    run_dir = Path("runs/streamlit_demo")

    with st.spinner("Running simulation..."):

        result = run_static(
            params,
            run_dir=run_dir,
            save_slices=True,
            save_full=False,
            progress=None,
        )

        st.success("Run complete")

    st.subheader("Result")
    st.json(result)

    st.subheader("Generated files")

    files = sorted(run_dir.glob("*"))
    for f in files:
        st.write(f.name)

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        st.subheader("Metadata")
        st.json(metadata_path.read_text())

    scalar_log = run_dir / "scalar_log.csv"
    if scalar_log.exists():
        import pandas as pd

        st.subheader("Scalar log")
        df = pd.read_csv(scalar_log)
        st.dataframe(df)

        if "z_um" in df.columns and "Imax" in df.columns:
            st.line_chart(df.set_index("z_um")["Imax"])

    st.write(result)
