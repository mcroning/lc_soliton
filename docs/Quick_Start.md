# LC Soliton Quick Start

This guide is for colleagues who want to run the current LC soliton package without digging through the development history.

The package currently supports:

- static self-consistent optical propagation
- time-dependent LC evolution
- optional dual-grid director solves inside the unified TD runner
- eigensoliton/existence-curve workflows under active development
- Streamlit GUI operation on local machines and cluster OnDemand sessions

## 1. Install

From the repository root:

```bash
python -m pip install -e .
```

On the Tufts cluster:

```bash
conda activate /cluster/tufts/cglab/mcroning/condaenv/prenv
python -m pip install -e .
```

## 2. Launch the GUI

```bash
python -m streamlit run app/app.py
```

On Tufts Open OnDemand:

```bash
./ondemand_lc_streamlit
```

## 3. First Static Test Case

Recommended settings:

- Nx = 256
- Ny = 256
- Nz = 50
- x aperture = 75 µm
- y aperture = 100 µm
- dz = 5 µm
- wavelength = 0.633 µm
- V_bias = 0.915 V
- P_mW = 1.0
- waist_x = 3 µm
- waist_y = 3 µm
- theta_bc = 0

Typical diagnostics:

```text
b            ~ 1.7186
bi           ~ 214
theta_max    ~ 0.89 rad
rrms         ~ 3e-3
```

## 4. Output Files

Typical run outputs:

```text
metadata.json
scalar_log.csv
environment.json
```

## 5. Known Limitations

- High-power static cases may be difficult to converge
- Theta > pi/4 regimes require careful interpretation
- Eigensoliton workflows are still being polished

## 6. More Documentation

- docs/gui.md
- docs/public_api.md
- docs/eigensoliton_profiles.md
- docs/theory.md

## 7. Next Step

Once the package is installed and validated, run the guided tutorial:

- [First Simulation Tutorial](tutorial_first_simulation.md)

This tutorial walks through a complete 1 mW LC soliton simulation using the Streamlit GUI.