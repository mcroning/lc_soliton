"""
Minimal LC Soliton GUI prototype.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from PIL import Image

from lc_soliton import (
    SimulationRequest,
    OutputRequest,
    RuntimeRequest,
    SolverRequest,
    GridRequest,
    GeometryRequest,
    MaterialRequest,
    LaunchRequest,
    BoundaryRequest,
    run_engine,
    load_reference_case,
    summarize_reference_case,
    available_engine_modes,
    describe_engine_modes,
    run_reference_case,
    available_reference_cases,
)

st.set_page_config(page_title="LC Soliton", layout="wide")

st.warning(
    "Current CPU static engine is a package smoke path and does not yet reproduce "
    "the trusted strict-static bridge confinement. GPU bridge validation is pending."
)

st.title("LC Soliton Simulator")
st.caption("Available engine modes: " + ", ".join(available_engine_modes()))

mode_descriptions = describe_engine_modes()

mode_labels = {
    "static": "Static",
    "time_dependent": "Time Dependent",
    "time_dependent_dual_grid": "Time Dependent (Dual Grid)",
}

display_mode_labels = {
    "static": "Static",
    "strict_static": "Static",
    "time_dependent": "Time Dependent",
    "td_predictor_only": "Time Dependent",
    "time_dependent_dual_grid": "Time Dependent (Dual Grid)",
    "dg_td_predictor": "Time Dependent (Dual Grid)",
}

# -------------------------
# Sidebar controls
# -------------------------

st.sidebar.header("Grid")
Nx = st.sidebar.slider("Nx", 32, 512, 256, step=32)
Ny = st.sidebar.slider("Ny", 32, 512, 256, step=32)
Nz = st.sidebar.slider("Nz", 4, 600, 50)

st.sidebar.header("Geometry")
xaper_um = st.sidebar.number_input("xaper (µm)", value=75.0)
yaper_um = st.sidebar.number_input("yaper (µm)", value=100.0)
dz_um = st.sidebar.number_input("dz (µm)", value=5.0)

st.sidebar.header("Physics")
b = st.sidebar.number_input("b", value=2.487025357142857, format="%.6f")
bi = st.sidebar.number_input("bi", value=214.28571428571414, format="%.6f")

st.sidebar.header("Beam")
waist_x_um = st.sidebar.number_input("waist x (µm)", value=3.0)
waist_y_um = st.sidebar.number_input("waist y (µm)", value=3.0)

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

with st.sidebar.expander("Mode notes"):
    if selected_mode == "static":
        st.write("Static mode computes a self-consistent z-marched steady-state solution.")
    elif selected_mode == "time_dependent":
        st.write("Time-dependent mode evolves the LC director response on the full grid.")
    elif selected_mode == "time_dependent_dual_grid":
        st.write(
            "Dual-grid TD mode evolves optics on the full grid while solving "
            "the LC director on a coarser grid."
        )

st.sidebar.header("Solver")

static_max_steps = st.sidebar.number_input(
    "Static max steps",
    value=100,
    min_value=1,
    step=1,
)

if selected_mode in {"time_dependent", "time_dependent_dual_grid"}:
    Nt = st.sidebar.number_input("Time steps", value=100, min_value=1, step=1)
    dt = st.sidebar.number_input("dt", value=5e-4, format="%.6f")
    t_stride = st.sidebar.number_input("Output stride", value=1, min_value=1, step=1)
else:
    Nt = 1
    dt = 5e-4
    t_stride = 1

# -------------------------
# Reference case
# -------------------------

st.header("Trusted reference case")

reference_names = available_reference_cases()
selected_reference = st.selectbox(
    "Reference case",
    reference_names,
    index=reference_names.index("strict_static_centroid_drift")
    if "strict_static_centroid_drift" in reference_names
    else 0,
)

summary = summarize_reference_case(selected_reference)
reference_data = load_reference_case(selected_reference)

st.caption(
    f"Reference summary: max_residual_rms={summary.get('max_residual_rms', 'n/a')}"
)

cfg = reference_data.get("config", {})
trusted_material = cfg.get("material", {})
trusted_grid = cfg.get("grid", {})
trusted_launch = cfg.get("launch", {})

st.subheader("Trusted static-case parameters")
col1, col2, col3 = st.columns(3)
col1.metric("b", trusted_material.get("b", "n/a"))
col2.metric("bi", trusted_material.get("bi", "n/a"))
col3.metric("dz (µm)", trusted_grid.get("dz_um", "n/a"))

with st.expander("Trusted config"):
    st.json(cfg)

with st.expander("Reference JSON"):
    st.json(reference_data)

if st.button("Show trusted strict-static reference case"):
    st.subheader("Reference figures")

    figures = reference_data.get("figures", {})

    xz_png = Path(figures.get("xz_reference", ""))
    centroid_png = Path(figures.get("centroid_reference", ""))

    if xz_png.exists():
        st.image(Image.open(xz_png), caption="xz intensity reference")
    else:
        st.info("No xz reference image found.")

    if centroid_png.exists():
        st.image(Image.open(centroid_png), caption="x centroid drift reference")
    else:
        st.info("No centroid reference image found.")

    if "trusted_metrics" in reference_data:
        st.subheader("Trusted metrics")
        st.json(reference_data["trusted_metrics"])

    if "summary" in reference_data:
        st.subheader("Static z summary")
        st.json(reference_data["summary"])

    if "reports" in reference_data:
        import pandas as pd

        st.subheader("z-march reports")
        st.dataframe(pd.DataFrame(reference_data["reports"]))

st.subheader("Validate trusted reference case")

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

# -------------------------
# GUI run
# -------------------------

run_button = st.button(f"Run {selected_mode_label} case")
progress_bar = st.empty()
status_box = st.empty()


def gui_progress(message):
    text = str(message)
    status_box.info(text)

    m = re.search(r"z\s+(\d+)/(\d+)", text)
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        if total > 0:
            progress_bar.progress(min(max(current / total, 0.0), 1.0))
            return

    m = re.search(r"time step\s+(\d+)/(\d+)", text.lower())
    if m:
        current = int(m.group(1))
        total = int(m.group(2))
        if total > 0:
            progress_bar.progress(min(max(current / total, 0.0), 1.0))
            return


if run_button:
    run_dir = Path("runs/streamlit_demo")

    progress_bar.progress(0)
    status_box.info(f"Launching {selected_mode_label} simulation...")

    request = SimulationRequest(
        mode=selected_mode,
        grid=GridRequest(Nx=int(Nx), Ny=int(Ny), Nz=int(Nz)),
        geometry=GeometryRequest(
            xaper_um=float(xaper_um),
            yaper_um=float(yaper_um),
            dz_um=float(dz_um),
            wavelength_um=0.633,
        ),
        material=MaterialRequest(
            ne=1.7,
            no=1.5,
            b=float(b),
            bi=float(bi),
            mobility=1.0,
            theta_bc=0.0,
            theta_bias_amp=0.1,
            theta_clamp_min=-1.2,
            theta_clamp_max=1.2,
            theta_z_gamma=0.0,
        ),
        launch=LaunchRequest(
            power_mW=1.0,
            waist_x_um=float(waist_x_um),
            waist_y_um=float(waist_y_um),
            separation_um=0.0,
            coherent=False,
        ),
        boundary=BoundaryRequest(use_sponge=True, windowedge=0.1),
        solver=SolverRequest(
            static_max_steps=int(static_max_steps),
            Nt=int(Nt),
            dt=float(dt),
            t_stride=int(t_stride),
            dtau_static=0.01,
            static_tol_rms=0.005,
            static_tol_max=0.01,
            static_selfcons_passes=3,
            static_mix=0.3,
            dz_opt_max_phi=0.3,
            dn_max_est=0.02,
            max_substeps=16,
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

    st.subheader("Actual GUI request sent to engine.py")
    st.json(asdict(request))

    with st.spinner("Running simulation..."):
        result = run_engine(request, progress_callback=gui_progress)

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    st.session_state["last_run_dir"] = str(run_dir)
    st.session_state["last_metadata"] = metadata
    st.session_state["last_result"] = result
    st.session_state["last_mode"] = selected_mode

    status_box.success("Simulation finished successfully.")
    progress_bar.progress(100)
    st.success("Run complete")

# -------------------------
# Persistent result display
# -------------------------

if "last_run_dir" in st.session_state:
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = Path(st.session_state["last_run_dir"])
    metadata = st.session_state.get("last_metadata", {})
    result = st.session_state.get("last_result", {})
    result_mode = st.session_state.get("last_mode", selected_mode)

    metadata_path = run_dir / "metadata.json"

    st.subheader("Run summary")

    if metadata:
        col1, col2, col3 = st.columns(3)
        raw_mode = metadata.get("mode", result_mode)

        col1.metric("Mode", display_mode_labels.get(raw_mode, raw_mode))
        col2.metric(
            "Grid",
            f"{metadata.get('Nx')}×{metadata.get('Ny')}×{metadata.get('Nz')}",
        )
        col3.metric("Backend", metadata.get("backend", "n/a"))

        st.caption(
            f"dz = {metadata.get('dz_um', 'n/a')} µm, "
            f"nsub = {metadata.get('nsub', 'n/a')}"
        )

    summary_tab, diagnostics_tab, files_tab = st.tabs(
        ["Summary", "Diagnostics", "Files / provenance"]
    )

    with summary_tab:
        st.subheader("Run summary")
        st.write("Simulation completed successfully.")

        if metadata:
            st.json(metadata)

        if isinstance(result, dict):
            st.subheader("Engine result")
            st.json(result)

        st.subheader("Final yz intensity cross section")

        iyz_path = run_dir / "Iyz.dat"

        if iyz_path.exists() and metadata and iyz_path.stat().st_size > 0:
            Nz_meta = int(metadata["Nz"])
            Ny_meta = int(metadata["Ny"])
            n_values = iyz_path.stat().st_size // 4
            Nt_out = n_values // (Nz_meta * Ny_meta)

            if Nt_out > 0:
                Iyz = np.memmap(
                    iyz_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(Nt_out, Nz_meta, Ny_meta),
                )

                final_yz = np.asarray(Iyz[-1])

                z_max_um = (float(metadata["Nz"]) - 1) * float(metadata["dz_um"])
                yaper_meta_um = float(metadata["yaper_um"])

                fig, ax = plt.subplots(figsize=(9, 4))
                im = ax.imshow(
                    final_yz.T,
                    aspect="auto",
                    origin="lower",
                    extent=[0.0, z_max_um, -0.5 * yaper_meta_um, 0.5 * yaper_meta_um],
                )

                ax.set_xlabel("z (µm)")
                ax.set_ylabel("y (µm)")
                ax.set_title("Final yz intensity")
                fig.colorbar(im, ax=ax, label="I")

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Iyz.dat exists but contains no complete frames.")
        else:
            st.info("No saved Iyz.dat found for this run.")

        st.subheader("Slice movie viewer")

        movie_options = {
            "Iyz": run_dir / "Iyz.dat",
            "Ixz": run_dir / "Ixz.dat",
            "dthetayz": run_dir / "dthetayz.dat",
            "dthetaxz": run_dir / "dthetaxz.dat",
        }

        available_movies = {
            name: path
            for name, path in movie_options.items()
            if path.exists() and path.stat().st_size > 0
        }

        if available_movies and metadata:
            movie_name = st.selectbox(
                "Movie slice",
                list(available_movies.keys()),
                key="movie_slice_selector",
            )

            movie_path = available_movies[movie_name]

            Nz_meta = int(metadata["Nz"])
            Nx_meta = int(metadata["Nx"])
            Ny_meta = int(metadata["Ny"])

            n_values = movie_path.stat().st_size // 4

            if movie_name.endswith("yz"):
                frame_shape = (Nz_meta, Ny_meta)
                frame_size = Nz_meta * Ny_meta
            else:
                frame_shape = (Nz_meta, Nx_meta)
                frame_size = Nz_meta * Nx_meta

            nframes = n_values // frame_size
       
            if nframes <= 0:
                st.info(f"{movie_name} exists but contains no complete frames.")
            else:
                frame_key = f"{movie_name}_movie_frame"
            
                if frame_key not in st.session_state:
                    st.session_state[frame_key] = nframes - 1
            
                st.session_state[frame_key] = min(
                    max(int(st.session_state[frame_key]), 0),
                    int(nframes - 1),
                )
            
                c1, c2, c3, c4, c5 = st.columns(5)
            
                with c1:
                    if st.button("⏮ First", key=f"{movie_name}_first"):
                        st.session_state[frame_key] = 0
            
                with c2:
                    if st.button("◀ Previous", key=f"{movie_name}_prev"):
                        st.session_state[frame_key] = max(
                            0,
                            int(st.session_state[frame_key]) - 1,
                        )
            
                with c3:
                    st.metric("Frame", f"{st.session_state[frame_key]} / {nframes - 1}")
            
                with c4:
                    if st.button("Next ▶", key=f"{movie_name}_next"):
                        st.session_state[frame_key] = min(
                            int(nframes - 1),
                            int(st.session_state[frame_key]) + 1,
                        )
            
                with c5:
                    if st.button("Last ⏭", key=f"{movie_name}_last"):
                        st.session_state[frame_key] = int(nframes - 1)
            
                if nframes == 1:
                    frame = 0
                    st.caption("Single-frame dataset")
                else:
                    frame = st.slider(
                        "Frame",
                        min_value=0,
                        max_value=int(nframes - 1),
                        value=int(st.session_state[frame_key]),
                        key=f"{movie_name}_frame_slider",
                    )
                    st.session_state[frame_key] = int(frame)
            
                frame = int(st.session_state[frame_key])
            
                arr = np.memmap(
                    movie_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(nframes, *frame_shape),
                )
            
                data = np.asarray(arr[frame])
            
                fig, ax = plt.subplots(figsize=(9, 4))
                
                if movie_name.endswith("yz"):
                    extent = [
                        0.0,
                        (Nz_meta - 1) * float(metadata["dz_um"]),
                        -0.5 * float(metadata["yaper_um"]),
                        0.5 * float(metadata["yaper_um"]),
                    ]
                    ax.set_ylabel("y (µm)")
                else:
                    extent = [
                        0.0,
                        (Nz_meta - 1) * float(metadata["dz_um"]),
                        -0.5 * float(metadata["xaper_um"]),
                        0.5 * float(metadata["xaper_um"]),
                    ]
                    ax.set_ylabel("x (µm)")
                
                imshow_kwargs = dict(
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                )
                
                if movie_name in {"Iyz", "Ixz"}:
                    default_vmax = float(np.nanmax(arr))
                    if not np.isfinite(default_vmax) or default_vmax <= 0:
                        default_vmax = 1.0
                
                    I_vmax = st.number_input(
                        f"{movie_name} fixed color max",
                        value=default_vmax,
                        min_value=0.0,
                        format="%.6g",
                        key=f"{movie_name}_vmax",
                    )
                
                    imshow_kwargs["vmin"] = 0.0
                    imshow_kwargs["vmax"] = float(I_vmax)
                
                im = ax.imshow(data.T, **imshow_kwargs)
                
                ax.set_xlabel("z (µm)")
                ax.set_title(f"{movie_name}, frame {frame}/{nframes - 1}")
                fig.colorbar(im, ax=ax, label=movie_name)
                
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No saved slice movie files found for this run.")

    with diagnostics_tab:
        scalar_log = run_dir / "scalar_log.csv"

        if scalar_log.exists():
            import pandas as pd

            st.subheader("Scalar diagnostics")
            df = pd.read_csv(scalar_log)

            if result_mode == "static":
                st.info(
                    "Static result view: z-dependent quantities are shown along "
                    "the propagation direction."
                )
            elif result_mode == "time_dependent":
                st.info(
                    "Time-dependent result view: scalar summaries are reduced to "
                    "a final-z time trace."
                )
            elif result_mode == "time_dependent_dual_grid":
                st.info(
                    "Dual-grid TD result view: scalar summaries are reduced to a "
                    "final-z time trace; director dynamics were solved on the coarse grid."
                )

            if "Imax" in df.columns:
                st.subheader("Intensity summary")

                if result_mode == "static":
                    if "z_um" in df.columns:
                        st.caption("Static run: maximum intensity versus propagation distance.")
                        st.line_chart(df.set_index("z_um")["Imax"])
                    else:
                        st.line_chart(df["Imax"])
                else:
                    st.caption(
                        "Time-dependent run: maximum intensity at the final z-slice "
                        "versus time/output step."
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

            with st.expander("Scalar log columns"):
                st.write(list(df.columns))

            with st.expander("Scalar log table"):
                st.dataframe(df)
        else:
            st.info("No scalar_log.csv found for this run.")

    with files_tab:
        with st.expander("Saved simulation request"):
            request_path = run_dir / "request.json"
            if request_path.exists():
                st.json(json.loads(request_path.read_text()))
            else:
                st.write("No request.json found.")

        with st.expander("Generated files"):
            for f in sorted(run_dir.glob("*")):
                st.write(f.name)

        if metadata_path.exists():
            with st.expander("Metadata"):
                st.json(metadata)

        with st.expander("Raw result dictionary"):
            st.json(result)