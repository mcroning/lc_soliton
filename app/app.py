"""
LC Soliton Streamlit GUI.
"""

from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

import os
from datetime import datetime

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

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
from lc_soliton.request_translate import request_to_lcparams_kwargs
from lc_soliton.core.context import LCParams, make_context
from lc_soliton.core.backend import asnumpy

from lc_soliton.validated_core.eigenmode_core import (
    list_run_profiles,
    read_profile_summary,
)


def request_to_gui_defaults(req: dict) -> dict:
    mode = req.get("mode")

    return {
        "selected_mode_label": mode_labels.get(mode, mode),
        # After loading a template, show the editable beam widgets.
        "launch_source": "Build Gaussian beam(s)",

        "Nx": req["grid"].get("Nx"),
        "Ny": req["grid"].get("Ny"),
        "Nz": req["grid"].get("Nz"),

        "xaper_um": req["geometry"].get("xaper_um"),
        "yaper_um": req["geometry"].get("yaper_um"),
        "dz_um": req["geometry"].get("dz_um"),

        "ne": req["material"].get("ne"),
        "no": req["material"].get("no"),
        "theta_bc": req["material"].get("theta_bc"),
        "K_SI": req["material"].get("K"),
        "De_rel": req["material"].get("De"),

        "P_mW": req["launch"].get("power_mW"),
        "waist_x_um": req["launch"].get("waist_x_um"),
        "waist_y_um": req["launch"].get("waist_y_um"),
        "separation_um": req["launch"].get("separation_um"),
        "pair_angle_deg": req["launch"].get("pair_angle_deg"),
        "power_ratio": req["launch"].get("power_ratio"),
        "theta_out1_deg": req["launch"].get("theta_out1_deg"),
        "theta_out2_deg": req["launch"].get("theta_out2_deg"),
        "phi1_deg": req["launch"].get("phi1_deg"),
        "phi2_deg": req["launch"].get("phi2_deg"),
        "coherent": req["launch"].get("coherent"),

        "static_max_steps": req["solver"].get("static_max_steps"),
        "Nt": req["solver"].get("Nt"),
        "dt": req["solver"].get("dt"),
        "t_stride": req["solver"].get("t_stride"),
        "dtau_static": req["solver"].get("dtau_static"),
        "static_tol_rms": req["solver"].get("static_tol_rms"),
        "static_tol_max": req["solver"].get("static_tol_max"),
        "static_selfcons_passes": req["solver"].get("static_selfcons_passes"),
        "static_mix": req["solver"].get("static_mix"),
        "dz_opt_max_phi": req["solver"].get("dz_opt_max_phi"),
        "dn_max_est": req["solver"].get("dn_max_est"),
        "max_substeps": req["solver"].get("max_substeps"),

        "save_slices": req["output"].get("save_slices"),
        "save_full": req["output"].get("save_full"),
    }


def gui_default(key: str, fallback):
    defaults = st.session_state.get("gui_template_defaults", {})
    value = defaults.get(key, fallback)
    return fallback if value is None else value


# Streamlit widget helpers: initialize session_state once, then let the
# widget own the key. This lets template loading set pending defaults
# before widgets are instantiated without triggering Streamlit errors.
def init_state_default(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def sb_text(label: str, key: str, default: str, **kwargs):
    init_state_default(key, default)
    return st.sidebar.text_input(label, key=key, **kwargs)


def sb_checkbox(label: str, key: str, default: bool = False, **kwargs):
    init_state_default(key, default)
    return st.sidebar.checkbox(label, key=key, **kwargs)


def sb_number(label: str, key: str, default, **kwargs):
    init_state_default(key, default)
    return st.sidebar.number_input(label, key=key, **kwargs)


def sb_slider(label: str, key: str, min_value, max_value, default, **kwargs):
    init_state_default(key, default)
    return st.sidebar.slider(label, min_value, max_value, key=key, **kwargs)


def sb_radio(label: str, key: str, options, default=None, **kwargs):
    init_state_default(key, default if default is not None else options[0])
    return st.sidebar.radio(label, options, key=key, **kwargs)


def sb_selectbox(label: str, key: str, options, default=None, **kwargs):
    init_state_default(key, default if default is not None else options[0])
    return st.sidebar.selectbox(label, options, key=key, **kwargs)
def list_child_dirs(path: Path):
    try:
        items = []
        for p in path.iterdir():
            if p.name.startswith("."):
                continue
            try:
                if p.is_dir():
                    items.append(p)
            except OSError:
                continue
        return sorted(items, key=lambda p: p.name.lower())
    except Exception:
        return []

            
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

def compute_gui_biased_material_values(
    *,
    V_bias,
    P_mW,
    K_SI,
    De_rel,
    ne,
    no,
    xaper_um,
    use_b_override,
    b_override,
    use_bi_override,
    bi_override,
):
    V_F = float(reported_freedericksz_voltage(K=float(K_SI), De=float(De_rel)))

    b_live = float(compute_b_from_voltage(float(V_bias), K=float(K_SI), De=float(De_rel)))

    bi_live = float(compute_bi_from_power(
        float(P_mW),
        d_um=float(xaper_um),
        K=float(K_SI),
        ne=float(ne),
        no=float(no),
    ))

    b_run = float(b_override) if use_b_override else b_live
    bi_run = float(bi_override) if use_bi_override else bi_live

    return V_F, b_live, bi_live, b_run, bi_run

st.set_page_config(page_title="LC Soliton", layout="wide")
st.title("LC Soliton Simulator")

mode_options = available_engine_modes()
mode_descriptions = describe_engine_modes()



mode_labels = {
    "static": "Static",
    "time_dependent": "Time Dependent",
    "time_dependent_dual_grid": "Time Dependent (Dual Grid)",
}

# Apply a template request before any widgets are instantiated.
# This avoids assigning to widget-owned keys after creation.
if "pending_template_request" in st.session_state:
    req = st.session_state.pop("pending_template_request")
    defaults = request_to_gui_defaults(req)
    for key, value in defaults.items():
        if value is not None:
            st.session_state[key] = value


display_mode_labels = {
    "static": "Static",
    "strict_static": "Static",
    "time_dependent": "Time Dependent",
    "td_predictor_only": "Time Dependent",
    "time_dependent_dual_grid": "Time Dependent (Dual Grid)",
    "dg_td_predictor": "Time Dependent (Dual Grid)",
}
#from streamlit_file_browser import st_file_browser



# -------------------------
# Sidebar controls
# -------------------------

st.sidebar.header("Workflow")

workflow = sb_radio(
    "Workflow",
    "workflow",
    [
        "Single run",
        "Existence curve sweep",
        "Log RMS / width stability monitor",
    ],
    default="Single run",
)

if workflow == "Existence curve sweep":
    st.sidebar.info("Placeholder: continuation over P or V. Not wired yet.")
    sweep_param = sb_selectbox("Sweep parameter", "sweep_param", ["P_mW", "V_bias"], default="P_mW")
    sweep_values_text = sb_text(
        "Sweep values",
        "sweep_values_text",
        "0.1, 0.2, 0.5, 1.0, 2.0, 4.0",
    )

elif workflow == "Log RMS / width stability monitor":
    st.sidebar.info("Placeholder: grid, dz, and dt sensitivity studies. Not wired yet.")
    monitor_param = sb_selectbox(
        "Study parameter",
        "monitor_param",
        ["dt", "dz_um", "Nx/Ny", "theta_bc", "P_mW", "V_bias"],
        default="dt",
    )
    monitor_values_text = sb_text(
        "Study values",
        "monitor_values_text",
        "0.001, 0.0005, 0.00025",
    )

st.sidebar.header("Run")

mode_label_options = [mode_labels.get(mode, mode) for mode in mode_options]
selected_mode_label = sb_selectbox(
    "Simulation mode",
    "selected_mode_label",
    mode_label_options,
    default=mode_label_options[0],
)

selected_mode = {
    mode_labels.get(mode, mode): mode
    for mode in mode_options
}[selected_mode_label]

st.sidebar.caption(mode_descriptions.get(selected_mode, ""))

run_label = sb_text("Run label", "run_label", "streamlit_run")

safe_label = "".join(
    c if c.isalnum() or c in "-_." else "_"
    for c in run_label
)

st.sidebar.subheader("Storage")

if "pending_output_root_text" in st.session_state:
    st.session_state["output_root_text"] = st.session_state.pop("pending_output_root_text")

if "pending_replay_root_text" in st.session_state:
    st.session_state["replay_root_text"] = st.session_state.pop("pending_replay_root_text")

if "output_root_text" not in st.session_state:
    st.session_state["output_root_text"] = str(
        Path.home() / "lc_soliton_runs"
    )

output_root = Path(
    sb_text("Output root", "output_root_text", str(Path.home() / "lc_soliton_runs"))
).expanduser()

if "replay_root_text" not in st.session_state:
    st.session_state["replay_root_text"] = st.session_state["output_root_text"]

replay_root = Path(
    sb_text("Read/template root", "replay_root_text", st.session_state["output_root_text"])
).expanduser()

try:
    output_root.mkdir(parents=True, exist_ok=True)

    testfile = output_root / ".write_test"
    testfile.write_text("ok")
    testfile.unlink()

    output_ok = True

except Exception as e:
    output_ok = False
    st.sidebar.error(f"Output root is not writable: {e}")

replay_ok = replay_root.exists()

if not replay_ok:
    st.sidebar.warning("Read/template root does not exist.")

with st.expander("Browse folders", expanded=False):
    if "mini_browser_path" not in st.session_state:
        st.session_state["mini_browser_path"] = str(Path.home())

    current = Path(st.session_state["mini_browser_path"]).expanduser()

    st.write(f"Current: `{current}`")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("⬆ Parent", key="mini_parent"):
            st.session_state["mini_browser_path"] = str(current.parent)
            st.rerun()

    with c2:
        if st.button("Use as output root", key="mini_use_output"):
            st.session_state["pending_output_root_text"] = str(current)
            st.rerun()
    
    with c3:
        if st.button("Use as read/template root", key="mini_use_replay"):
            st.session_state["pending_replay_root_text"] = str(current)
            st.rerun() 

    child_dirs = list_child_dirs(current)

    if child_dirs:
        picked = st.selectbox(
            "Subfolders",
            [p.name for p in child_dirs],
            key="mini_subfolder",
        )

        if st.button("Open selected folder", key="mini_open"):
            st.session_state["mini_browser_path"] = str(current / picked)
            st.rerun()
    else:
        st.info("No readable subfolders.")


    

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = output_root / f"{safe_label}_{timestamp}"

st.sidebar.caption(f"Output:\n`{output_root}`")
st.sidebar.caption(f"Read:\n`{replay_root}`")
st.sidebar.caption(f"New run directory:\n`{run_dir}`")

save_slices = sb_checkbox("Save xz/yz movie slices", "save_slices", True)
save_full = sb_checkbox("Save full final theta", "save_full", False)

st.sidebar.header("Grid")
Nx = sb_slider("Nx", "Nx", 32, 512, 256, step=32)
Ny = sb_slider("Ny", "Ny", 32, 512, 256, step=32)
Nz = sb_slider("Nz", "Nz", 4, 600, 50)

st.sidebar.header("Geometry")
xaper_um = sb_number(
    "x aperture / cell thickness d (µm)",
    "xaper_um",
    75.0,
)
yaper_um = sb_number("y aperture (µm)", "yaper_um", 100.0)
dz_um = sb_number("dz (µm)", "dz_um", 5.0)

st.sidebar.header("LC physics")
V_bias = sb_number(
    "Bias voltage V (V)",
    "V_bias",
    0.9144,
    min_value=0.0,
    format="%.6g",
)
P_mW = sb_number(
    "Optical power P (mW)",
    "P_mW",
    1.0,
    min_value=0.0,
    format="%.6g",
)
theta_bc = sb_number(
    "Boundary / pretilt θ_bc (rad)",
    "theta_bc",
    0.0,
    format="%.6g",
)

with st.sidebar.expander("Material constants"):
    init_state_default("K_SI", 7e-12)
    init_state_default("De_rel", 13.0)
    init_state_default("ne", 1.7)
    init_state_default("no", 1.5)
    K_SI = st.number_input("Elastic constant K (SI)", format="%.6g", key="K_SI")
    De_rel = st.number_input("Dielectric anisotropy Δε", format="%.6g", key="De_rel")
    ne = st.number_input("ne", format="%.6g", key="ne")
    no = st.number_input("no", format="%.6g", key="no")

b_live = float(compute_b_from_voltage(V_bias, K=K_SI, De=De_rel))
bi_live = float(compute_bi_from_power(P_mW, d_um=float(xaper_um), K=K_SI, ne=ne, no=no))

with st.sidebar.expander("Advanced: override derived b and bi"):
    use_b_override = sb_checkbox("Override b", "use_b_override", False)
    b_override = st.number_input(
        "b override",
        value=float(b_live),
        format="%.6g",
        disabled=not use_b_override,
        key="b_override",
    )

    use_bi_override = sb_checkbox("Override bi", "use_bi_override", False)
    bi_override = st.number_input(
        "bi override",
        value=float(bi_live),
        format="%.6g",
        disabled=not use_bi_override,
        key="bi_override",
    )

b_run = float(b_override) if use_b_override else b_live
bi_run = float(bi_override) if use_bi_override else bi_live
V_F = float(reported_freedericksz_voltage(K=K_SI, De=De_rel))

b = b_run
bi = bi_run

st.sidebar.caption(f"Voltage-derived b = {b_live:.6g}")
st.sidebar.caption(f"Power-derived bi = {bi_live:.6g}")
st.sidebar.caption(f"Using b = {b_run:.6g}")
st.sidebar.caption(f"Using bi = {bi_run:.6g}")
st.sidebar.caption(f"Zero-pretilt Freedericksz V_F ≈ {V_F:.6g} V")

params_preview = LCParams(
    Nx=int(Nx),
    Ny=int(Ny),
    Nz=1,
    xaper_um=float(xaper_um),
    yaper_um=float(yaper_um),
    dz_um=float(dz_um),
    wavelength_um=0.633,
    ne=float(ne),
    no=float(no),
    b=float(b_run),
    bi=float(bi_run),
    theta_bc=float(theta_bc),
    K=float(K_SI),
    De=float(De_rel),
)

ctx_preview, _ = make_context(params_preview)
theta_bias = asnumpy(ctx_preview.theta_bias_2d)

x_um = np.linspace(
    -float(xaper_um) / 2,
     float(xaper_um) / 2,
     theta_bias.shape[0],
)

mid_y = theta_bias.shape[1] // 2
theta0_deg = np.degrees(float(theta_bias[theta_bias.shape[0] // 2, mid_y]))

col_bias, col_bias_info = st.columns([2, 1])

with col_bias:
    fig, ax = plt.subplots(figsize=(3, 2))

    ax.plot(
        x_um,
        np.degrees(theta_bias[:, mid_y]),
    )

    theta_box = AnchoredText(
        f"$\\theta_0$ = {theta0_deg:.1f}°",
        loc="upper right",
        prop={"size": 9},
        frameon=True,
    )
    ax.add_artist(theta_box)

    ax.set_title("LC bias profile", fontsize=10)
    ax.set_xlim(x_um[0], x_um[-1])
    ax.set_ylim(0, 90)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("θ (deg)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.sidebar.header("Launch source")

launch_source = sb_radio(
    "Initial condition",
    "launch_source",
    [
        "Build Gaussian beam(s)",
        "Use saved eigensoliton profile",
        "Load saved run as template",
    ],
    default="Build Gaussian beam(s)",
)

selected_profile_path = None
selected_profile_summary = None
profile_theta_source = "Saved eigensoliton θ"

if launch_source == "Load saved run as template":
    template_run_dir = sb_text(
        "Template run folder",
        "template_run_dir",
        str(replay_root),
    )

    template_path = Path(template_run_dir).expanduser()
    template_request_path = template_path / "request.json"

    if not template_path.exists():
        st.sidebar.warning("Template folder does not exist.")
    elif not template_request_path.exists():
        st.sidebar.warning("No request.json found in template folder.")
    else:
        st.sidebar.success("Template request.json found.")

    if st.sidebar.button("Load template into GUI", key="load_template_button"):
        if not template_request_path.exists():
            st.sidebar.error("Cannot load template: request.json was not found.")
        else:
            req = json.loads(template_request_path.read_text())
            st.session_state["pending_template_request"] = req
            st.session_state["loaded_template_run_dir"] = str(template_path)
            st.rerun()

    waist_x_um = st.session_state.get("waist_x_um", 3.0)
    waist_y_um = st.session_state.get("waist_y_um", 3.0)
    separation_um = st.session_state.get("separation_um", 0.0)
    pair_angle_deg = st.session_state.get("pair_angle_deg", 0.0)
    power_ratio = st.session_state.get("power_ratio", 0.0)
    theta_out1_deg = st.session_state.get("theta_out1_deg", 0.0)
    phi1_deg = st.session_state.get("phi1_deg", 0.0)
    theta_out2_deg = st.session_state.get("theta_out2_deg", 0.0)
    phi2_deg = st.session_state.get("phi2_deg", 0.0)
    coherent = st.session_state.get("coherent", False)

elif launch_source == "Use saved eigensoliton profile":
    profile_run_dir = sb_text(
        "Eigensoliton run folder",
        "profile_run_dir",
        str(replay_root / "existence_curve"),
    )

    profiles = list_run_profiles(profile_run_dir)

    if not profiles:
        st.sidebar.warning("No eigensoliton profiles found.")
    else:
        summaries = [read_profile_summary(p) for p in profiles]
        labels = [
            f"{s['P_mW']:.4g} mW  β={s['beta']:.5g}"
            for s in summaries
        ]

        choice = st.sidebar.selectbox("Saved eigenmode", labels, key="saved_eigenmode")
        idx = labels.index(choice)

        selected_profile_path = str(profiles[idx])
        selected_profile_summary = summaries[idx]

        st.sidebar.caption(selected_profile_path)

        with st.sidebar.expander("Selected eigensoliton parameters", expanded=True):
            st.write(f"P = {selected_profile_summary['P_mW']:.6g} mW")
            st.write(f"β = {selected_profile_summary['beta']:.6g}")
            st.write(f"b = {selected_profile_summary['b']:.6g}")
            st.write(f"bi = {selected_profile_summary['bi']:.6g}")
            st.write(f"θ_bc = {selected_profile_summary['theta_bc']:.6g}")
            st.write(f"grid = {selected_profile_summary['A_shape']}")
            st.caption(
                "Saved-profile launches use the profile grid and physics parameters."
            )

    profile_theta_source = sb_radio(
        "Initial director θ",
        "profile_theta_source",
        ["Saved eigensoliton θ", "Bias θ only"],
        default="Saved eigensoliton θ",
    )

    waist_x_um = 3.0
    waist_y_um = 3.0
    separation_um = 0.0
    pair_angle_deg = 0.0
    power_ratio = 0.0
    theta_out1_deg = 0.0
    phi1_deg = 0.0
    theta_out2_deg = 0.0
    phi2_deg = 0.0
    coherent = False

else:
    st.sidebar.header("Beam")
    waist_x_um = sb_number("waist x (µm)", "waist_x_um", 3.0)
    waist_y_um = sb_number("waist y (µm)", "waist_y_um", 3.0)
    separation_um = sb_number("beam separation (µm)", "separation_um", 0.0)
    pair_angle_deg = sb_number("separation angle (deg)", "pair_angle_deg", 0.0)
    power_ratio = sb_number("P2/P1", "power_ratio", 0.0, min_value=0.0)

    theta_out1_deg = sb_number("beam 1 polar angle (deg)", "theta_out1_deg", 0.0)
    phi1_deg = sb_number("beam 1 azimuth (deg)", "phi1_deg", 0.0)
    theta_out2_deg = sb_number("beam 2 polar angle (deg)", "theta_out2_deg", 0.0)
    phi2_deg = sb_number("beam 2 azimuth (deg)", "phi2_deg", 0.0)
    coherent = sb_checkbox("coherent beams", "coherent", False)

st.sidebar.header("Solver")
static_max_steps = sb_number(
    "Static max steps",
    "static_max_steps",
    100,
    min_value=1,
    step=1,
)

if selected_mode in {"time_dependent", "time_dependent_dual_grid"}:
    Nt = sb_number("Time steps", "Nt", 100, min_value=1, step=1)
    dt = sb_number("dt", "dt", 5e-4, format="%.6g")
    t_stride = sb_number("Output stride", "t_stride", 1, min_value=1, step=1)
else:
    Nt = 1
    dt = 5e-4
    t_stride = 1

with st.sidebar.expander("Advanced solver parameters"):
    init_state_default("dtau_static", 0.01)
    init_state_default("static_tol_rms", 0.005)
    init_state_default("static_tol_max", 0.01)
    init_state_default("static_selfcons_passes", 3)
    init_state_default("static_mix", 0.3)
    init_state_default("dz_opt_max_phi", 0.3)
    init_state_default("dn_max_est", 0.02)
    init_state_default("max_substeps", 16)
    dtau_static = st.number_input("dtau_static", format="%.6g", key="dtau_static")
    static_tol_rms = st.number_input("static_tol_rms", format="%.6g", key="static_tol_rms")
    static_tol_max = st.number_input("static_tol_max", format="%.6g", key="static_tol_max")
    static_selfcons_passes = st.number_input(
        "static_selfcons_passes",
        min_value=1,
        step=1,
        key="static_selfcons_passes",
    )
    static_mix = st.number_input("static_mix", format="%.6g", key="static_mix")
    dz_opt_max_phi = st.number_input("dz_opt_max_phi", format="%.6g", key="dz_opt_max_phi")
    dn_max_est = st.number_input("dn_max_est", format="%.6g", key="dn_max_est")
    max_substeps = st.number_input("max_substeps", min_value=1, step=1, key="max_substeps")


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


if workflow == "Log RMS / width stability monitor":
    st.info(
        "Stability monitor placeholder: this will run grid/dz/dt studies and plot "
        "log residual RMS, width, centroid drift, and convergence floors."
    )

col_run, col_stop = st.columns(2)

with col_run:
    run_button = st.button(
        "Run existence curve sweep" if workflow == "Existence curve sweep" else f"Run {selected_mode_label} case",
        disabled=(workflow not in {"Single run", "Existence curve sweep"} or not output_ok),
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
def eig_progress(current, total, power_mW, phase="", outer=None, max_outer=None, beta=None, rrms=None, rmax=None):
    if outer is not None and max_outer is not None:
        frac = (current + outer / max_outer) / max(total, 1)
        msg = (
            f"Eigensoliton sweep {current + 1}/{total}: "
            f"P = {power_mW:.3g} mW, outer {outer}/{max_outer}"
        )
        if beta is not None and rrms is not None:
            msg += f", β={beta:.4g}, Rrms={rrms:.2e}"
    else:
        frac = current / max(total, 1)
        msg = f"Eigensoliton sweep {current}/{total}: P = {power_mW:.3g} mW"
    progress_bar.progress(min(max(frac, 0.0), 1.0))

    status_box.info(msg)


# -------------------------
# Run engine
# -------------------------

if run_button:
    st.session_state["stop_requested"] = False

    # Output destination for this new run
    st.session_state["last_run_dir"] = str(run_dir)
    st.session_state["last_output_root"] = str(output_root)

    # Input/replay source used for saved profiles or recreated runs
    st.session_state["last_replay_root"] = str(replay_root)

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
        st.write(f"Existence curve output: {sweep_dir}")

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
            
        
        if launch_source == "Use saved eigensoliton profile":
            if selected_profile_path is None:
                raise ValueError(
                    "Use saved eigensoliton profile was selected, but no profile was chosen."
                )
        
            setattr(request, "launch_profile_path", str(selected_profile_path))
        
        result = run_eigensoliton_existence_curve(
            request,
            sweep_values,
            run_dir=sweep_dir,
            progress_callback=eig_progress,
            branch_name="fundamental",
            mode_seed="00",
            w0_um=float(waist_x_um),
            checkpoint_prefix="lc_eigensoliton",
            save_profiles=True,
            live_plot=False,
            solve_kwargs=dict(
                max_outer=int(static_max_steps),
                theta_residual_tol_rms=float(static_tol_rms),
                theta_residual_tol_max=float(static_tol_max),
            ),
        )
        
        st.session_state["last_run_dir"] = str(sweep_dir)
        st.session_state["last_output_root"] = str(output_root)
        st.session_state["last_replay_root"] = str(replay_root)
        st.session_state["last_result"] = result
        st.session_state["last_metadata"] = {}
        
        df_curve = result.get("dataframe")


        

        if df_curve is not None and len(df_curve):
            st.subheader("Eigensoliton existence curve")

            df_good = df_curve[df_curve["theta_residual_rms"] < 5e-2].copy()
            df_bad = df_curve[df_curve["theta_residual_rms"] >= 5e-2].copy()

            if len(df_bad):
                st.warning(f"{len(df_bad)} branch point(s) exceeded residual tolerance and were excluded from plots.")

            st.line_chart(df_good.set_index("P_mW")[["beta"]])

            st.subheader("Mode widths")
            st.line_chart(df_good.set_index("P_mW")[["sx_um", "sy_um"]])

            st.dataframe(df_curve)

        status_box.success(
            f"Eigensoliton existence curve finished: {result['n_completed']} branch points."
        )
        progress_bar.progress(100)
        st.success(f"Wrote {result['csv']}")
        st.stop()

    P_request = float(P_mW)
    b_request = float(b_run)
    bi_request = float(bi_run)
    theta_bc_request = float(theta_bc)



    Nx_request = int(Nx)
    Ny_request = int(Ny)
    
    if (
        launch_source == "Use saved eigensoliton profile"
        and selected_profile_summary is not None
    ):
        P_request = float(selected_profile_summary["P_mW"])
        b_request = float(selected_profile_summary["b"])
        bi_request = float(selected_profile_summary["bi"])
        theta_bc_request = float(selected_profile_summary["theta_bc"])
    
        if selected_profile_summary.get("A_shape") is not None:
            Nx_request = int(selected_profile_summary["A_shape"][0])
            Ny_request = int(selected_profile_summary["A_shape"][1])
    print(
        "[request physics]",
        "P=", P_request,
        "b=", b_request,
        "bi=", bi_request,
        "theta_bc=", theta_bc_request,
    )



    request = SimulationRequest(
        mode=selected_mode,
        grid=GridRequest(Nx=int(Nx_request), Ny=int(Ny_request), Nz=int(Nz)),
        geometry=GeometryRequest(
            xaper_um=float(xaper_um),
            yaper_um=float(yaper_um),
            dz_um=float(dz_um),
            wavelength_um=0.633,
        ),
        material=MaterialRequest(
            ne=float(ne),    
            no=float(no),    
            b=float(b_request),    
            bi=float(bi_request),    
            mobility=1.0,    
            theta_bc=float(theta_bc_request),
            theta_bias_amp=0.1,
            theta_clamp_min=-1.2,
            theta_clamp_max=1.2,
            theta_z_gamma=0.0,
            K=float(K_SI),
            De=float(De_rel),
        ),
        launch=LaunchRequest(
            power_mW=float(P_request),
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
    if launch_source == "Use saved eigensoliton profile":
        if selected_profile_path is None:
            raise ValueError(
                "Use saved eigensoliton profile was selected, but no profile was chosen."
            )

        setattr(request, "launch_profile_path", selected_profile_path)
        setattr(request, "launch_profile_theta_source", profile_theta_source)
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

    if "last_replay_root" in st.session_state:
        replay_display_root = Path(st.session_state["last_replay_root"])

        if st.button("Display selected read/recreate folder"):
            st.session_state["last_run_dir"] = str(replay_display_root)
            st.session_state["last_result"] = {}
            st.session_state["last_metadata"] = {}
            st.rerun()

    
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
