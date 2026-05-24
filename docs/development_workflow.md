# Development workflow

## Philosophy

Public APIs should remain stable while internal implementations evolve.

Notebook experiments should graduate into:
1. validated implementation code
2. tests
3. public API exposure
4. GUI integration

## Typical workflow

### 1. Create feature branch

    git checkout -b feature-name

### 2. Develop incrementally

- implement feature
- test manually
- add automated tests
- validate reference behavior

### 3. Run validation

    pytest -q tests
    lc-soliton --validate-reference

### 4. Keep imports lightweight

Avoid heavy imports at package top level.

### 5. Preserve reproducibility

Public runners should emit:
- config
- summaries
- environment.json
- validation metrics

### 6. Prefer public APIs

GUI and examples should call:
- run_static
- run_td
- run_dg_td

rather than legacy internal machinery directly.

## Portability goals

The package should remain suitable for:
- local CPU execution
- CUDA GPU execution
- Slurm clusters
- Docker / Apptainer deployment
- future public releases
