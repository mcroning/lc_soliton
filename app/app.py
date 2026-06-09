"""
LC Soliton Streamlit GUI.
"""

from __future__ import annotations

import json
import re
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

from lc_soliton.physics.lc.bias import (
    compute_b_from_voltage,
    compute_bi_from_power,
    reported_freedericksz_voltage,
)
from lc_soliton.eigensoliton_runner import run_eigensoliton_existence_curve

def read_scalar_log_optional(scalar_log, st_obj=None):
    import pandas as pd
    from pandas.errors import EmptyDataError, ParserError

    if scalar_log is None or not scalar_log.exists() or scalar_log.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(scalar_log)
    except EmptyDataError:
        return None
    except ParserError as exc:
        if st_obj is not None:
            st_obj.warning(
                "scalar_log.csv has a partial/corrupt row, probably from a stopped run. "
                f"Reading valid rows only. Details: {exc}"
            )
        try:
            return pd.read_csv(scalar_log, engine="python", on_bad_lines="skip")
        except Exception:
            return None


def valid_saved_frames_from_scalar_log(df):
    if df is None or "jt" not in df.columns or len(df) == 0:
        return None
    jt = df["jt"].dropna()
    if len(jt) == 0:
        return None
    return int(jt.max()) + 1


def should_stop():
    return bool(st.session_state.get("stop_requested", False))


st.set_page_config(page_title="LC Soliton", layout="wide")
st.title("LC Soliton Simulator")

mode_options = available_engine_modes()
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

st.sidebar.header("Workflow")

workflow = st.sidebar.radio(
    "Workflow",
    [
        "Single run",
        "Existence curve sweep",
        "Log RMS / width stability monitor",
    ],
    index=0,
)

if workflow == "Existence curve sweep":
    st.sidebar.info("Placeholder: continuation over P or V. Not wired yet.")
    sweep_param = st.sidebar.selectbox("Sweep parameter", ["P_mW", "V_bias"])
    sweep_values_text = st.sidebar.text_input(
        "Sweep values",
        value="0.1, 0.2, 0.5, 1.0, 2.0, 4.0",
    )

elif workflow == "Log RMS / width stability monitor":
    st.sidebar.info("Placeholder: grid, dz, and dt sensitivity studies. Not wired yet.")
    monitor_param = st.sidebar.selectbox(
        "Study parameter",
        ["dt", "dz_um", "Nx/Ny", "theta_bc", "P_mW", "V_bias"],
    )
    monitor_values_text = st.sidebar.text_input(
        "Study values",
        value="0.001, 0.0005, 0.00025",
    )

st.sidebar.header("Run")

selected_mode_label = st.sidebar.selectbox(
    "Simulation mode",
    [mode_labels.get(mode, mode) for mode in mode_options],
)

selected_mode = {
    mode_labels.get(mode, mode): mode
    for mode in mode_options
}[selected_mode_label]

st.sidebar.caption(mode_descriptions.get(selected_mode, ""))

run_dir = Path(st.sidebar.text_input("Run folder", value="runs/streamlit_demo"))
save_slices = st.sidebar.checkbox("Save xz/yz movie slices", value=True)
save_full = st.sidebar.checkbox("Save full final theta", value=False)

st.sidebar.header("Grid")
Nx = st.sidebar.slider("Nx", 32, 512, 256, step=32)
Ny = st.sidebar.slider("Ny", 32, 512, 256, step=32)
Nz = st.sidebar.slider("Nz", 4, 600, 50)

st.sidebar.header("Geometry")
xaper_um = st.sidebar.number_input("x aperture / cell thickness d (µm)", value=75.0)
yaper_um = st.sidebar.number_input("y aperture (µm)", value=100.0)
dz_um = st.sidebar.number_input("dz (µm)", value=5.0)

st.sidebar.header("LC physics")
V_bias = st.sidebar.number_input("Bias voltage V (V)", value=1.10, min_value=0.0, format="%.6g")
P_mW = st.sidebar.number_input("Optical power P (mW)", value=1.0, min_value=0.0, format="%.6g")
theta_bc = st.sidebar.number_input("Boundary / pretilt θ_bc (rad)", value=0.0, format="%.6g")

with st.sidebar.expander("Material constants"):
    K_SI = st.number_input("Elastic constant K (SI)", value=7e-12, format="%.6g")
    De_rel = st.number_input("Dielectric anisotropy Δε", value=13.0, format="%.6g")
    ne = st.number_input("ne", value=1.7, format="%.6g")
    no = st.number_input("no", value=1.5, format="%.6g")

b_from_V = compute_b_from_voltage(V_bias, K=K_SI, De=De_rel)
bi_from_P = compute_bi_from_power(P_mW, d_um=float(xaper_um), K=K_SI, ne=ne, no=no)

with st.sidebar.expander("Advanced: override derived b and bi"):
    use_b_override = st.checkbox("Override b", value=False)
    b_override = st.number_input("b override", value=float(b_from_V), format="%.6g", disabled=not use_b_override)

    use_bi_override = st.checkbox("Override bi", value=False)
    bi_override = st.number_input("bi override", value=float(bi_from_P), format="%.6g", disabled=not use_bi_override)

b = float(b_override) if use_b_override else float(b_from_V)
bi = float(bi_override) if use_bi_override else float(bi_from_P)
V_F = reported_freedericksz_voltage(K=K_SI, De=De_rel)

st.sidebar.caption(f"Derived b = {b:.6g}")
st.sidebar.caption(f"Derived bi = {bi:.6g}")
st.sidebar.caption(f"Zero-pretilt Freedericksz V_F ≈ {V_F:.6g} V")

st.sidebar.header("Beam")
waist_x_um = st.sidebar.number_input("waist x (µm)", value=3.0)
waist_y_um = st.sidebar.number_input("waist y (µm)", value=3.0)
separation_um = st.sidebar.number_input("beam separation (µm)", value=0.0)
pair_angle_deg = st.sidebar.number_input("separation angle (deg)", value=0.0)
power_ratio = st.sidebar.number_input("P2/P1", value=0.0, min_value=0.0)

theta_out1_deg = st.sidebar.number_input("beam 1 polar angle (deg)", value=0.0)
phi1_deg = st.sidebar.number_input("beam 1 azimuth (deg)", value=0.0)
theta_out2_deg = st.sidebar.number_input("beam 2 polar angle (deg)", value=0.0)
phi2_deg = st.sidebar.number_input("beam 2 azimuth (deg)", value=0.0)
coherent = st.sidebar.checkbox("coherent beams", value=False)

st.sidebar.header("Solver")
static_max_steps = st.sidebar.number_input("Static max steps", value=100, min_value=1, step=1)

if selected_mode in {"time_dependent", "time_dependent_dual_grid"}:
    Nt = st.sidebar.number_input("Time steps", value=100, min_value=1, step=1)
    dt = st.sidebar.number_input("dt", value=5e-4, format="%.6g")
    t_stride = st.sidebar.number_input("Output stride", value=1, min_value=1, step=1)
else:
    Nt = 1
    dt = 5e-4
    t_stride = 1

with st.sidebar.expander("Advanced solver parameters"):
    dtau_static = st.number_input("dtau_static", value=0.01, format="%.6g")
    static_tol_rms = st.number_input("static_tol_rms", value=0.005, format="%.6g")
    static_tol_max = st.number_input("static_tol_max", value=0.01, format="%.6g")
    static_selfcons_passes = st.number_input("static_selfcons_passes", value=3, min_value=1, step=1)
    static_mix = st.number_input("static_mix", value=0.3, format="%.6g")
    dz_opt_max_phi = st.number_input("dz_opt_max_phi", value=0.3, format="%.6g")
    dn_max_est = st.number_input("dn_max_est", value=0.02, format="%.6g")
    max_substeps = st.number_input("max_substeps", value=16, min_value=1, step=1)


# -------------------------
# Main setup/status
# -------------------------

st.subheader("Current setup")

c1, c2, c3, c4 = st.columns(4)
c1.metric("V", f"{V_bias:.4g} V")
c2.metric("P", f"{P_mW:.4g} mW")
c3.metric("b", f"{b:.4g}")
c4.metric("bi", f"{bi:.4g}")

st.caption(
    f"Workflow: {workflow} · "
    f"θ_bc={theta_bc:.6g} rad · "
    f"d={xaper_um:.6g} µm · "
    f"grid={Nx}×{Ny}×{Nz} · "
    f"dz={dz_um:.6g} µm"
)

if workflow == "Existence curve sweep":
    st.info(
        "Existence curve sweep placeholder: this will launch continuation runs over "
        "P or V and save branch metrics such as β, width, residual, and convergence."
    )

elif workflow == "Log RMS / width stability monitor":
    st.info(
        "Stability monitor placeholder: this will run grid/dz/dt studies and plot "
        "log residual RMS, width, centroid drift, and convergence floors."
    )

col_run, col_stop = st.columns(2)

with col_run:
    run_button = st.button(
        "Run existence curve sweep" if workflow == "Existence curve sweep" else f"Run {selected_mode_label} case",
        disabled=(workflow not in {"Single run", "Existence curve sweep"}),
    )

with col_stop:
    if st.button("Request stop after current slice"):
        st.session_state["stop_requested"] = True

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


# -------------------------
# Run engine
# -------------------------

if run_button:
    st.session_state["stop_requested"] = False
    st.session_state["last_run_dir"] = str(run_dir)
    st.session_state["last_mode"] = selected_mode
    st.session_state["last_result"] = {}
    st.session_state["last_metadata"] = {}

    progress_bar.progress(0)
    status_box.info(f"Launching {selected_mode_label} simulation...")


    # Recompute from live GUI values immediately before request construction
    
    V_F = float(reported_freedericksz_voltage(K=float(K_SI), De=float(De_rel)))
    b_live = (3.141592653589793**2 / 8.0) * (float(V_bias) / V_F)**2

    bi_live = float(compute_bi_from_power(
        float(P_mW),
        d_um=float(xaper_um),
        K=float(K_SI),
        ne=float(ne),
        no=float(no),
    ))
    
    b_run = float(b_override) if use_b_override else b_live
    bi_run = float(bi_override) if use_bi_override else bi_live
    
    if workflow == "Existence curve sweep":
        sweep_values = [
            float(s.strip())
            for s in sweep_values_text.split(",")
            if s.strip()
        ]

        if sweep_param != "P_mW":
            raise ValueError("Eigensoliton existence curve currently supports P_mW sweeps only.")

        sweep_dir = run_dir / "existence_curve"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        status_box.info(
            f"Launching eigensoliton continuation with {len(sweep_values)} power points..."
        )
        progress_bar.progress(0)

        request = SimulationRequest(
            mode="strict_static",
            grid=GridRequest(Nx=int(Nx), Ny=int(Ny), Nz=int(Nz)),
            geometry=GeometryRequest(
                xaper_um=float(xaper_um),
                yaper_um=float(yaper_um),
                dz_um=float(dz_um),
                wavelength_um=0.633,
            ),
            material=MaterialRequest(
                ne=float(ne),
                no=float(no),
                b=float(b_run),
                bi=float(bi_run),
                mobility=1.0,
                theta_bc=float(theta_bc),
                theta_bias_amp=0.1,
                theta_clamp_min=-1.2,
                theta_clamp_max=1.2,
                theta_z_gamma=0.0,
                K=float(K_SI),
                De=float(De_rel),
            ),
            launch=LaunchRequest(
                power_mW=float(sweep_values[0]),
                waist_x_um=float(waist_x_um),
                waist_y_um=float(waist_y_um),
                separation_um=float(separation_um),
                pair_angle_deg=float(pair_angle_deg),
                theta_out1_deg=float(theta_out1_deg),
                theta_out2_deg=float(theta_out2_deg),
                phi1_deg=float(phi1_deg),
                phi2_deg=float(phi2_deg),
                power_ratio=float(power_ratio),
                coherent=bool(coherent),
            ),
            boundary=BoundaryRequest(use_sponge=True, windowedge=0.1),
            solver=SolverRequest(
                static_max_steps=int(static_max_steps),
                Nt=1,
                dt=float(dt),
                t_stride=int(t_stride),
                dtau_static=float(dtau_static),
                static_tol_rms=float(static_tol_rms),
                static_tol_max=float(static_tol_max),
                static_selfcons_passes=int(static_selfcons_passes),
                static_mix=float(static_mix),
                dz_opt_max_phi=float(dz_opt_max_phi),
                dn_max_est=float(dn_max_est),
                max_substeps=int(max_substeps),
            ),
            output=OutputRequest(
                run_dir=str(sweep_dir),
                save_slices=bool(save_slices),
                save_full=bool(save_full),
            ),
            runtime=RuntimeRequest(backend="auto", progress=True),
        )

        result = run_eigensoliton_existence_curve(
            request,
            sweep_values,
            run_dir=sweep_dir,
            branch_name="fundamental",
            mode_seed="00",
            w0_um=float(waist_x_um),
            checkpoint_prefix="lc_eigensoliton",
            save_profiles=False,
            live_plot=False,
            solve_kwargs=dict(
                max_outer=int(static_max_steps),
                theta_residual_tol_rms=float(static_tol_rms),
                theta_residual_tol_max=float(static_tol_max),
            ),
        )

        st.session_state["last_run_dir"] = str(sweep_dir)
        st.session_state["last_result"] = result
        st.session_state["last_metadata"] = {}

        status_box.success(
            f"Eigensoliton existence curve finished: {result['n_completed']} branch points."
        )
        progress_bar.progress(100)
        st.success(f"Wrote {result['csv']}")
        st.stop()






    
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
        ne=float(ne),
        no=float(no),
        b=float(b_run),
        bi=float(bi_run),
        mobility=1.0,
        theta_bc=float(theta_bc),
        theta_bias_amp=0.1,
        theta_clamp_min=-1.2,
        theta_clamp_max=1.2,
        theta_z_gamma=0.0,
        K=float(K_SI),
        De=float(De_rel),
    ),
        launch=LaunchRequest(
            power_mW=float(P_mW),
            waist_x_um=float(waist_x_um),
            waist_y_um=float(waist_y_um),
            separation_um=float(separation_um),
            pair_angle_deg=float(pair_angle_deg),
            theta_out1_deg=float(theta_out1_deg),
            theta_out2_deg=float(theta_out2_deg),
            phi1_deg=float(phi1_deg),
            phi2_deg=float(phi2_deg),
            power_ratio=float(power_ratio),
            coherent=bool(coherent),
        ),
        boundary=BoundaryRequest(use_sponge=True, windowedge=0.1),
        solver=SolverRequest(
            static_max_steps=int(static_max_steps),
            Nt=int(Nt),
            dt=float(dt),
            t_stride=int(t_stride),
            dtau_static=float(dtau_static),
            static_tol_rms=float(static_tol_rms),
            static_tol_max=float(static_tol_max),
            static_selfcons_passes=int(static_selfcons_passes),
            static_mix=float(static_mix),
            dz_opt_max_phi=float(dz_opt_max_phi),
            dn_max_est=float(dn_max_est),
            max_substeps=int(max_substeps),
        ),
        output=OutputRequest(
            run_dir=str(run_dir),
            save_slices=bool(save_slices),
            save_full=bool(save_full),
        ),
        runtime=RuntimeRequest(backend="auto", progress=True),
    )

    try:
        with st.spinner("Running simulation..."):
            result = run_engine(
                request,
                progress_callback=gui_progress,
                should_stop=should_stop,
            )

        st.session_state["last_result"] = result
        status_box.success("Simulation finished successfully.")
        progress_bar.progress(100)
        st.success("Run complete")

    except Exception as exc:
        st.session_state["last_result"] = {"error": str(exc)}
        status_box.error("Run stopped or failed before completion.")
        st.warning("Partial output may still be available below.")
        st.exception(exc)

    finally:
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        
        if metadata_path.exists():
            metadata.update({
                "V_bias": float(V_bias),
                "P_mW": float(P_mW),
                "K": float(K_SI),
                "De": float(De_rel),
                "b": float(b_run),
                "bi": float(bi_run),
                "use_sponge": True,
                "windowedge": 0.1,
            })
            metadata_path.write_text(json.dumps(metadata, indent=2))
        
        st.session_state["last_metadata"] = metadata


# -------------------------
# Results display
# -------------------------

if "last_run_dir" not in st.session_state:
    st.info("No completed or partial run selected yet.")
else:
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = Path(st.session_state["last_run_dir"])
    metadata = st.session_state.get("last_metadata", {})
    result = st.session_state.get("last_result", {})
    result_mode = st.session_state.get("last_mode", selected_mode)

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        st.session_state["last_metadata"] = metadata

    if metadata:
        raw_mode = metadata.get("mode", result_mode)
        st.caption(
            f"{display_mode_labels.get(raw_mode, raw_mode)} · "
            f"{metadata.get('Nx')}×{metadata.get('Ny')}×{metadata.get('Nz')} · "
            f"dz={metadata.get('dz_um', 'n/a')} µm · "
            f"backend={metadata.get('backend', 'n/a')} · "
            f"θ_bc={metadata.get('theta_bc', 'n/a')} rad"
        )

    scalar_log = run_dir / "scalar_log.csv"
    df_scalar = read_scalar_log_optional(scalar_log, st)
    valid_saved_frames = valid_saved_frames_from_scalar_log(df_scalar)

    summary_tab, diagnostics_tab, workflows_tab, files_tab = st.tabs(
        ["Summary", "Diagnostics", "Workflows", "Files / provenance"]
    )

    with summary_tab:
        try:
            if isinstance(result, dict):
                with st.expander("Engine result"):
                    st.json(result)

            st.subheader("Final saved intensity cross section")

            final_slice_choice = st.radio(
                "Final intensity slice",
                ["yz", "xz"],
                horizontal=True,
                key="final_intensity_slice_choice",
            )
            
            if final_slice_choice == "yz":
                final_path = run_dir / "Iyz.dat"
                final_name = "Iyz"
            else:
                final_path = run_dir / "Ixz.dat"
                final_name = "Ixz"
            
            if final_path.exists() and metadata and final_path.stat().st_size > 0:
                Nz_meta = int(metadata["Nz"])
                Nx_meta = int(metadata["Nx"])
                Ny_meta = int(metadata["Ny"])
            
                if final_slice_choice == "yz":
                    frame_shape = (Nz_meta, Ny_meta)
                    frame_size = Nz_meta * Ny_meta
                    transverse_extent = [
                        -0.5 * float(metadata["yaper_um"]),
                        0.5 * float(metadata["yaper_um"]),
                    ]
                    transverse_label = "y (µm)"
                else:
                    frame_shape = (Nz_meta, Nx_meta)
                    frame_size = Nz_meta * Nx_meta
                    transverse_extent = [
                        -0.5 * float(metadata["xaper_um"]),
                        0.5 * float(metadata["xaper_um"]),
                    ]
                    transverse_label = "x (µm)"
            
                n_values = final_path.stat().st_size // 4
                nframes = n_values // frame_size
            
                if valid_saved_frames is not None:
                    nframes = min(nframes, valid_saved_frames)
            
                if nframes > 0:
                    arr = np.memmap(
                        final_path,
                        dtype=np.float32,
                        mode="r",
                        shape=(nframes, *frame_shape),
                    )
            
                    final_data = np.asarray(arr[nframes - 1])
            
                    z_max_um = (Nz_meta - 1) * float(metadata["dz_um"])
            
                    fig, ax = plt.subplots(figsize=(9, 4))
                    im = ax.imshow(
                        final_data.T,
                        aspect="auto",
                        origin="lower",
                        extent=[
                            0.0,
                            z_max_um,
                            transverse_extent[0],
                            transverse_extent[1],
                        ],
                    )
            
                    ax.set_xlabel("z (µm)")
                    ax.set_ylabel(transverse_label)
                    ax.set_title(f"Final saved {final_name} intensity, frame {nframes - 1}")
                    fig.colorbar(im, ax=ax, label="I")
            
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"{final_name}.dat exists but contains no complete frames.")
            else:
                st.info(f"No saved {final_name}.dat found for this run.")

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

                if valid_saved_frames is not None:
                    nframes = min(nframes, valid_saved_frames)

                if nframes <= 0:
                    st.info(f"{movie_name}.dat exists but has no complete saved frames yet.")
                else:
                    expected_nt = int(metadata.get("Nt_out", nframes))
                    if nframes < expected_nt:
                        st.warning(
                            f"Partial run data: showing {nframes} saved frame(s) "
                            f"out of expected {expected_nt}."
                        )

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

                    imshow_kwargs = dict(aspect="auto", origin="lower", extent=extent)

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

        except Exception as e:
            st.error(f"Summary tab failed: {e}")

    with diagnostics_tab:
        try:
            st.subheader("Scalar diagnostics")
            df = df_scalar

            if df is None:
                st.warning(
                    "No scalar_log data available yet. "
                    "The run may have stopped before scalar diagnostics were written. "
                    "Partial I/theta movies may still be available in the Summary tab."
                )
            else:
                residual_col = None
                residual_label = None

                if "frac_rrms" in df.columns:
                    residual_col = "frac_rrms"
                    residual_label = "fractional residual RMS"
                elif "I_weighted_frac_rrms" in df.columns:
                    residual_col = "I_weighted_frac_rrms"
                    residual_label = "intensity-weighted fractional residual RMS"
                elif "rrms" in df.columns:
                    residual_col = "rrms"
                    residual_label = "residual RMS"

                if {"Imax", "rrms"}.issubset(df.columns):
                    df["Imax_weighted_rrms_proxy"] = df["Imax"] * df["rrms"]

                if residual_col is not None:
                    st.subheader("Residual quality")

                    if result_mode == "static":
                        if "z_um" in df.columns:
                            st.caption(f"Static run: {residual_label} versus z.")
                            st.line_chart(df.set_index("z_um")[residual_col])
                        else:
                            st.line_chart(df[residual_col])
                    else:
                        if {"jt", "z_um"}.issubset(df.columns):
                            pivot = df.pivot_table(
                                index="jt",
                                columns="z_um",
                                values=residual_col,
                                aggfunc="max",
                            ).sort_index()

                            residual_map = pivot.to_numpy(dtype=float)
                            z_values = pivot.columns.to_numpy(dtype=float)
                            jt_values = pivot.index.to_numpy(dtype=float)

                            fig, ax = plt.subplots(figsize=(9, 4))
                            im = ax.imshow(
                                residual_map,
                                aspect="auto",
                                origin="lower",
                                extent=[
                                    float(z_values.min()),
                                    float(z_values.max()),
                                    float(jt_values.min()),
                                    float(jt_values.max()),
                                ],
                            )
                            ax.set_xlabel("z (µm)")
                            ax.set_ylabel("saved time frame")
                            ax.set_title(f"{residual_label} versus z and time")
                            fig.colorbar(im, ax=ax, label=residual_col)
                            st.pyplot(fig)
                            plt.close(fig)

                            if len(jt_values) == 1:
                                frame_choice = int(jt_values[0])
                                st.caption("Single saved time frame.")
                            else:
                                frame_choice = st.slider(
                                    "Residual frame",
                                    min_value=int(jt_values.min()),
                                    max_value=int(jt_values.max()),
                                    value=int(jt_values.max()),
                                    key="residual_quality_frame",
                                )

                            if frame_choice in pivot.index:
                                st.caption(f"{residual_label} versus z for frame {frame_choice}.")
                                st.line_chart(pivot.loc[frame_choice])
                        else:
                            st.info("Need jt and z_um columns to show residual versus z and time.")
                else:
                    st.info("No residual columns found in scalar_log.csv.")

                if "Imax_weighted_rrms_proxy" in df.columns:
                    with st.expander("Legacy brightness-weighted residual proxy"):
                        st.caption(
                            "Imax × rrms is only a brightness-weighted proxy. "
                            "It can show intensity fringes and should not be used as the main convergence metric."
                        )

                        if result_mode == "static" and "z_um" in df.columns:
                            st.line_chart(df.set_index("z_um")["Imax_weighted_rrms_proxy"])
                        elif {"jt", "z_um"}.issubset(df.columns):
                            proxy_pivot = df.pivot_table(
                                index="jt",
                                columns="z_um",
                                values="Imax_weighted_rrms_proxy",
                                aggfunc="max",
                            ).sort_index()
                            st.line_chart(proxy_pivot.T)
                        else:
                            st.line_chart(df["Imax_weighted_rrms_proxy"])

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

        except Exception as e:
            st.error(f"Diagnostics tab failed: {e}")

    with workflows_tab:
        st.subheader("Workflow launchers")

        with st.expander("Existence curve sweep", expanded=True):
            st.write(
                "Placeholder: launch a power/voltage continuation sweep, save branch data, "
                "and plot β(P), width(P), residual(P), and convergence status."
            )

            st.text_input(
                "Power list for future sweep (mW)",
                value="0.1, 0.2, 0.5, 1.0, 2.0, 4.0",
                key="future_existence_powers",
            )

            st.text_input(
                "Voltage list for future sweep (V)",
                value=f"{V_bias:.6g}",
                key="future_existence_voltages",
            )

            st.button("Launch existence curve sweep — not wired yet", disabled=True)

        with st.expander("Log RMS / width stability monitor", expanded=True):
            st.write(
                "Placeholder: run grid/time-step sensitivity studies and monitor "
                "log residual RMS, beam width, centroid drift, and convergence floors."
            )

            st.text_input("Grid sizes", value="128, 256, 512", key="future_monitor_grids")
            st.text_input("dt values", value="0.001, 0.0005, 0.00025", key="future_monitor_dt")
            st.text_input("dz values (µm)", value="10, 5, 2", key="future_monitor_dz")

            st.button("Launch stability monitor — not wired yet", disabled=True)

    with files_tab:
        st.subheader("Files and provenance")

        st.write(f"Run directory: `{run_dir}`")

        if run_dir.exists():
            files = sorted(p.name for p in run_dir.iterdir())
            st.write(files)
        else:
            st.info("Run directory does not exist yet.")

        if metadata:
            with st.expander("metadata.json"):
                st.json(metadata)

        with st.expander("Reference validation / regression tests"):
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
            c1, c2, c3 = st.columns(3)
            c1.metric("b", cfg.get("material", {}).get("b", "n/a"))
            c2.metric("bi", cfg.get("material", {}).get("bi", "n/a"))
            c3.metric("dz (µm)", cfg.get("grid", {}).get("dz_um", "n/a"))

            if st.button("Validate selected reference case"):
                with st.spinner("Running trusted reference validation..."):
                    try:
                        ref_result = run_reference_case(selected_reference)
                    except Exception as exc:
                        st.error("Trusted reference validation failed")
                        st.exception(exc)
                    else:
                        st.success("Trusted reference validation passed")
                        if isinstance(ref_result, dict):
                            st.json(ref_result)
            show_trusted_config = st.checkbox(
                "Show trusted config",
                value=False,
                key="show_trusted_config",
            )
            
            if show_trusted_config:
                st.json(cfg)
            
            show_reference_json = st.checkbox(
                "Show full reference JSON",
                value=False,
                key="show_reference_json",
            )
            
            if show_reference_json:
                st.json(reference_data)

            if st.button("Show reference figures"):
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
