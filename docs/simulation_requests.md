# Simulation requests

Simulation requests are JSON-friendly descriptions of a run.

They are represented in Python by:

    SimulationRequest
    OutputRequest
    RuntimeRequest

## Example request

    {
      "mode": "strict_static",
      "params": {
        "Nx": 32,
        "Ny": 32,
        "Nz": 2,
        "static_max_steps": 2
      },
      "output": {
        "run_dir": "runs/example_request",
        "save_slices": false,
        "save_full": false
      },
      "runtime": {
        "backend": "auto",
        "progress": false
      }
    }

Run it with:

    lc-soliton --run-request request.json

## Purpose

Simulation requests provide a portable execution format for:

- CLI runs
- GUI runs
- Slurm jobs
- future container execution
- reference-case regeneration
