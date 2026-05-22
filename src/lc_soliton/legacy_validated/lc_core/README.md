# LC Soliton Production Runner

## Current stable workflow

1. Edit `prdata` in a notebook.
2. Build `cfg` with `config_from_prdata(prdata)`.
3. Run either:
   - `run_td_experiment(...)`
   - `run_static_experiment(...)`
4. Outputs are saved under `/cluster/tufts/cglab/mcroning/lc_runs`.

## Key modules

- `lc_core/config.py` — converts legacy `prdata` into structured config.
- `lc_core/gpu_context.py` — builds GPU context and runtime stores.
- `lc_core/pipeline.py` — user-facing experiment runners.
- `lc_core/static_z_march.py` — strict static bridge.
- `lc_core/td_runner.py` — true-TD bridge.
- `lc_core/residual_quality.py` — PDE defect diagnostics.
- `lc_core/io_arrays.py` — saves full arrays.
- `lc_core/io_movies.py` — saves movie slices.
- `legacy/lc_offload_tools.py` — temporary bridge to validated legacy kernels.

## Run outputs

Each run directory contains:

- `config.json`
- `environment.json`
- summary JSON files
- `arrays/`
- optional `movies/`
- static runs also include `pde_defect_diagnostics/`

## Development rule

Keep new features in standalone modules. Do not merge into one monolithic runner.