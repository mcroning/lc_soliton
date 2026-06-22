# Current Project Status

## Package Status

The LC Soliton package is operational and suitable for internal research use.

Implemented:

- Static self-consistent LC propagation
- Time-dependent LC evolution
- Unified TD runner
- Optional dual-grid acceleration
- Streamlit GUI
- Run provenance and metadata capture
- Reference-case validation
- Eigensoliton profile generation and replay

## Current Strengths

- Installable package
- GPU acceleration via CuPy
- Public package API
- Streamlit interface
- Reproducible run folders
- Trusted reference cases
- Validation framework

## Active Development Areas

- Existence-curve workflow polish
- Stability-analysis workflow polish
- High-power static convergence
- Documentation and usability improvements

## Known Limitations

- Some high-drive static cases remain challenging
- θ > π/4 regimes require careful interpretation
- Legacy validated modules remain transitional
- Cluster GUI access depends on local HPC configuration

## Recommended Entry Points

New users:

- Quick_Start.md
- gui.md

Developers:

- public_api.md
- project_map.md
- development_workflow.md