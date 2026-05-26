"""
Minimal LC Soliton GUI prototype.
"""

import json

from dataclasses import asdict

from PIL import Image

from pathlib import Path

import streamlit as st

from lc_soliton import (
    LCParams,
    SimulationRequest,
    OutputRequest,
    RuntimeRequest,
    SolverRequest,
    run_engine,
    run_static,
    load_reference_case,
    summarize_reference_case,
    available_engine_modes,
    describe_engine_modes,
    run_reference_case,
    available_reference_cases,
    describe_engine_modes,
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



mode_descriptions = describe_engine_modes()
st.caption(
    "Static and time-dependent liquid-crystal soliton simulation modes."
)

mode_labels = {
    "static": "Static",
    "time_dependent": "Time Dependent",
    "time_dependent_dual_grid": "Time Dependent (Dual Grid)",
}




mode_options = available_engine_modes()

selected_mode_label = st.sidebar.selectbox(
    "Simulation mode",
    [mode_labels.get(mode, mode) for mode in mode_options],
)

selected_mode = {
    mode_labels.get(mode, mode): mode
    for mode in mode_options
}[selected_mode_label]

st.sidebar.caption(mode_descriptions.get(selected_mode, ""))

request_path = run_dir / "request.json"

if request_path.exists():
    with st.expander("Saved simulation request"):
        st.json(json.loads(request_path.read_text()))

with st.sidebar.expander("Mode notes"):
    if selected_mode == "static":
        st.write(
            "Static mode computes a self-consistent z-marched steady-state solution."
        )
    elif selected_mode == "time_dependent":
        st.write(
            "Time-dependent mode evolves the LC director response on the full grid."
        )
    elif selected_mode == "time_dependent_dual_grid":
        st.write(
            "Dual-grid time-dependent mode evolves the optics on the full grid while solving the LC director on a coarser grid."
        )

st.sidebar.header("Solver")

static_max_steps = st.sidebar.number_input(
    "Static max steps",
    value=2,
    min_value=1,
    step=1,
)

if selected_mode in {"time_dependent", "time_dependent_dual_grid"}:
    Nt = st.sidebar.number_input("Time steps", value=100, min_value=1, step=1)
    dt = st.sidebar.number_input("dt", value=5e-4, format="%.6f")
    t_stride = st.sidebar.number_input("Output stride", value=1, min_value=1, step=1)
else:
    Nt = 100
    dt = 5e-4
    t_stride = 1



st.header("Trusted reference case")

reference_names = available_reference_cases()
selected_reference = st.selectbox(
    "Reference case",
    reference_names,
    index=reference_names.index("static_centroid_drift")
    if "static_centroid_drift" in reference_names
    else 0,
)

case_dir = Path(
    "validation/reference_cases/static_centroid_drift"
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

    if "reports" in data:
        import pandas as pd
    
        st.subheader("z-march reports")
        st.dataframe(pd.DataFrame(data["reports"]))

st.subheader("Validate trusted reference case")

summary = summarize_reference_case(selected_reference)

st.caption(
    f"Reference summary: "
    f"max_residual_rms={summary.get('max_residual_rms', 'n/a')}"
)

if st.button("Validate trusted reference case"):
    with st.spinner("Running trusted reference validation..."):
        try:
            result = run_reference_case(selected_reference)
        except Exception as exc:
            st.error("Trusted reference validation failed")
            st.exception(exc)
        else:
            st.success("Trusted reference validation passed")

            if isinstance(result, dict):
                st.json(result)

run_button = st.button(f"Run {selected_mode_label} case")

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


    status_box = st.empty()
    status_box.info(f"Launching {selected_mode_label} simulation...")

    
    with st.spinner("Running simulation..."):

        request = SimulationRequest(
            mode=selected_mode,
            params=asdict(params),
            solver=SolverRequest(
                static_max_steps=int(static_max_steps),
                Nt=int(Nt),
                dt=float(dt),
                t_stride=int(t_stride),
            ),
            output=OutputRequest(
                run_dir=str(run_dir),
                save_slices=True,
                save_full=False,
            ),
            runtime=RuntimeRequest(
                backend="auto",
                progress=True,
            ),
        )

        result = run_engine(request)
        status_box.success("Simulation finished successfully.")
    st.success("Run complete")


    st.subheader("Run summary")
        
    metadata_path = run_dir / "metadata.json"
    
    if metadata_path.exists():
        import json
    
        metadata = json.loads(metadata_path.read_text())
    
        col1, col2, col3 = st.columns(3)
    
        col1.metric("Mode", metadata.get("mode", "n/a"))
        col2.metric("Grid", f"{metadata.get('Nx')}×{metadata.get('Ny')}×{metadata.get('Nz')}")
        col3.metric("Backend", metadata.get("backend", "n/a"))
    
        st.caption(
            f"dz = {metadata.get('dz_um', 'n/a')} µm, "
            f"nsub = {metadata.get('nsub', 'n/a')}"
        )

    
    with st.expander("Raw result dictionary"):
        st.json(result)
    
    with st.expander("Generated files"):
        files = sorted(run_dir.glob("*"))
        for f in files:
            st.write(f.name)

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with st.expander("Metadata"):
            st.json(json.loads(metadata_path.read_text()))

    scalar_log = run_dir / "scalar_log.csv"
    if scalar_log.exists():
        import pandas as pd

        st.subheader("Scalar diagnostics")
        df = pd.read_csv(scalar_log)
        
        with st.expander("Scalar log table"):
            st.dataframe(df)

    if "Imax" in df.columns:
        st.subheader("Intensity summary")
    
        if selected_mode == "static":
            if "z_um" in df.columns:
                st.caption("Static run: maximum intensity versus propagation distance.")
                st.line_chart(df.set_index("z_um")["Imax"])
            else:
                st.line_chart(df["Imax"])
    
        else:
            st.caption(
                "Time-dependent run: plotting final-z maximum intensity versus time/output step."
            )
    
            if "k" in df.columns:
                final_k = df["k"].max()
                df_final = df[df["k"] == final_k].copy()
            else:
                df_final = df.copy()
    
            if "t" in df_final.columns:
                st.line_chart(df_final.set_index("t")["Imax"])
            elif "jt" in df_final.columns:
                st.line_chart(df_final.set_index("jt")["Imax"])
            else:
                st.line_chart(df_final["Imax"])

    st.write(result)
