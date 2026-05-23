# Portability roadmap

The package should remain suitable for future Docker / Apptainer / Singularity deployment.

## Design constraints

- Avoid hardcoded cluster paths in package code.
- Keep GPU dependencies optional.
- Keep CPU-only tests runnable without CUDA.
- Keep GUI code separate from compute backends.
- Prefer public APIs over legacy bridge calls.
- Store run provenance in JSON-compatible formats.
- Keep reference cases small enough for git.
- Keep large outputs outside git.

## Future container targets

1. CPU-only package + GUI
2. GPU-enabled package with CuPy/CUDA
3. HPC Apptainer image for Slurm execution
4. Lightweight public demo app for reference-case browsing

## Near-term action items

- Replace hardcoded paths in examples and validation scripts.
- Add environment export files.
- Separate GUI viewer mode from compute mode.
- Add public API for trusted reference-case execution.
