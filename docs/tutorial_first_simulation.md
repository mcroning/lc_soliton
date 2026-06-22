# Tutorial: Your First LC Soliton Simulation

This tutorial walks through a complete liquid-crystal soliton simulation using the Streamlit GUI.

Expected time:

- GPU workstation: a few seconds
- CPU-only laptop or VM: about 1–2 minutes

## Goal

Launch a 1 mW beam into a biased liquid-crystal cell and observe self-focusing.

At the end of the run you should see:

- a localized optical beam
- a self-induced director perturbation
- residual RMS around 0.003
- stable propagation through the cell

## Step 1: Launch the GUI

From the repository root:

```bash
streamlit run app/app.py
```

Open the displayed URL in a browser.

## Step 2: Enter Parameters

### Grid

```text
Nx = 256
Ny = 256
Nz = 50
```

### Geometry

```text
xaper_um = 75
yaper_um = 100
dz_um = 5
wavelength_um = 0.633
```

### Material

```text
ne = 1.7
no = 1.5
V_bias = 1.10
theta_bc = 0
```

The GUI should display approximately:

```text
b ≈ 1.72
bi ≈ 214
```

### Beam

```text
Power = 1.0 mW
waist_x_um = 3
waist_y_um = 3
coherent = False
```

### Solver

```text
Mode = Static
```

## Step 3: Run

Press:

```text
Run Simulation
```

A new run directory will be created.

Typical runtime:

- GPU: a few seconds
- CPU-only systems: about 1 minute

The reference Windows VM required approximately 70 seconds.

## Step 4: Inspect Results

The Summary tab should show:

- intensity cross sections
- director cross sections
- residual diagnostics

Typical values are:

```text
theta_max ≈ 0.89 rad
residual RMS ≈ 0.003
```

Small differences between systems are expected.

## Step 5: Explore

Try increasing power:

```text
0.5 mW
1.0 mW
2.0 mW
```

Observe:

- stronger self-focusing
- larger director reorientation
- larger propagation constant shift

## Validation

To verify the installation independently:

```bash
lc-soliton --validate-reference
```

Successful validation indicates that the package reproduces the trusted reference case.

## Next Steps

Explore:

- time-dependent evolution
- dual-grid acceleration
- eigensoliton profiles
- existence-curve workflows

Additional documentation:

```text
docs/gui.md
docs/eigensoliton_profiles.md
docs/theory.md
```
