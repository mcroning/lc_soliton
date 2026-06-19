"""
Canonical simulation request objects.

These are lightweight, JSON-friendly descriptions of what to run.
They are intended to sit above the current implementation layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


EngineMode = Literal[
    "static",
    "time_dependent",
]


from dataclasses import dataclass, field



@dataclass
class GeometryRequest:
    xaper_um: float = 75.0
    yaper_um: float = 100.0
    dz_um: float = 5.0
    wavelength_um: float = 0.633

@dataclass
class MaterialRequest:
    ne: float = 1.7
    no: float = 1.5
    b: float = 2.49
    bi: float = 214.29
    mobility: float = 1.0
    theta_bc: float = 0.0
    theta_bias_amp: float = 0.1
    theta_clamp_min: float = -1.2
    theta_clamp_max: float = 1.2
    theta_z_gamma: float = 0.0
    K: float = 1.2e-11
    De: float = 10.3

@dataclass
class LaunchRequest:
    power_mW: float = 1.0
    waist_um: float = 4.0
    waist_x_um: float | None = None
    waist_y_um: float | None = None
    separation_um: float = 0.0
    coherent: bool = False
    pair_angle_deg: float = 0.0
    theta_out1_deg: float = 0.0
    theta_out2_deg: float = 0.0
    phi1_deg: float = 0.0
    phi2_deg: float = 0.0
    power_ratio: float = 0.0
@dataclass
class BoundaryRequest:
    use_sponge: bool = True
    windowedge: float = 0.1


@dataclass
class OutputRequest:
    run_dir: str = "runs/lc_soliton_run"
    save_slices: bool = True
    save_full: bool = False


@dataclass
class RuntimeRequest:
    backend: str = "auto"
    progress: bool = True

@dataclass
class GridRequest:
    Nx: int = 512
    Ny: int = 512
    Nz: int = 100
    use_dual_grid: bool = False
    dual_grid_factor: int = 2

@dataclass
class SolverRequest:
    static_max_steps: int = 250
    Nt: int = 100
    dt: float = 5e-4
    t_stride: int = 1

    dz_opt_max_phi: float = 0.3
    dn_max_est: float = 0.02
    max_substeps: int = 16

    dtau_static: float = 0.02
    static_tol_rms: float = 0.001
    static_tol_max: float = 0.01
    static_selfcons_passes: int = 3
    static_mix: float = 0.6
    strict_max_outer_passes: int = 8



@dataclass
class SimulationRequest:
    mode: str = "static"
    grid: GridRequest = field(default_factory=GridRequest)
    geometry: GeometryRequest = field(default_factory=GeometryRequest)
    material: MaterialRequest = field(default_factory=MaterialRequest)
    launch: LaunchRequest = field(default_factory=LaunchRequest)
    boundary: BoundaryRequest = field(default_factory=BoundaryRequest)
    solver: SolverRequest = field(default_factory=SolverRequest)
    output: OutputRequest = field(default_factory=OutputRequest)
    runtime: RuntimeRequest = field(default_factory=RuntimeRequest)

    # Deprecated bridge only. Do not use in new callers.
    params: dict | None = None
    def to_dict(self) -> dict:
        data = asdict(self)
        if data.get("params") is None:
            data.pop("params", None)
        return data
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationRequest":
        data = dict(data)
    
        if "params" in data and data["params"]:
            from .request_translate import legacy_params_to_request_dict
            data = legacy_params_to_request_dict(data)
    
        output = OutputRequest(**data.get("output", {}))
        runtime = RuntimeRequest(**data.get("runtime", {}))
        grid = GridRequest(**data.get("grid", {}))
        material = MaterialRequest(**data.get("material", {}))
        geometry = GeometryRequest(**data.get("geometry", {}))
        solver = SolverRequest(**data.get("solver", {}))
        launch = LaunchRequest(**data.get("launch", {}))
        boundary = BoundaryRequest(**data.get("boundary", {}))
    
        return cls(
            mode=data.get("mode", "static"),
            grid=grid,
            material=material,
            geometry=geometry,
            solver=solver,
            launch=launch,
            boundary=boundary,
            output=output,
            runtime=runtime,
            params=None,
        )


def save_request(request: SimulationRequest, path: str | Path) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request.to_dict(), indent=2))
    return path


def load_request(path: str | Path) -> SimulationRequest:
    import json

    path = Path(path)
    return SimulationRequest.from_dict(json.loads(path.read_text()))


__all__ = [
    "EngineMode",
    "GridRequest",
    "GeometryRequest",
    "MaterialRequest",
    "LaunchRequest",
    "BoundaryRequest",
    "SolverRequest",
    "OutputRequest",
    "RuntimeRequest",
    "SimulationRequest",
    "save_request",
    "load_request",
]



