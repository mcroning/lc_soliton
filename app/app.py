"""
Minimal LC Soliton GUI prototype.
"""

from pathlib import Path

import streamlit as st

from lc_soliton import LCParams, run_static


st.set_page_config(
    page_title="LC Soliton",
    layout="wide",
)

st.title("LC Soliton Simulator")

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

    st.write(result)
