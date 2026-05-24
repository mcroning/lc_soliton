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
    "strict_static",
    "td_predictor_only",
    "dg_td_predictor",
]


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

@dataclass
class SolverRequest:
    static_max_steps: int = 100
    Nt: int = 100
    dt: float = 5e-4
    t_stride: int = 1

@dataclass
class MaterialRequest:
    ne: float = 1.7
    no: float = 1.5
    K: float = 12e-12
    De: float = 10.3

@dataclass
class GeometryRequest:
    xaper_um: float = 75.0
    yaper_um: float = 1000.0
    dz_um: float = 20.0

@dataclass
class SimulationRequest:
    """
    Canonical high-level request for an LC soliton simulation.
    """

    mode: EngineMode = "strict_static"
    grid: GridRequest = field(default_factory=GridRequest)
    params: dict[str, Any] = field(default_factory=dict)
    output: OutputRequest = field(default_factory=OutputRequest)
    runtime: RuntimeRequest = field(default_factory=RuntimeRequest)
    solver: SolverRequest = field(default_factory=SolverRequest)
    material: MaterialRequest = field(default_factory=MaterialRequest)
    geometry: GeometryRequest = field(default_factory=GeometryRequest)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationRequest":
        output = OutputRequest(**data.get("output", {}))
        runtime = RuntimeRequest(**data.get("runtime", {}))
        grid = GridRequest(**data.get("grid", {}))
        material = MaterialRequest(**data.get("material", {}))
        geometry = GeometryRequest(**data.get("geometry", {}))
        solver = SolverRequest(**data.get("solver", {}))

        return cls(
            mode=data.get("mode", "strict_static"),
            grid=grid,
            material=material,
            geometry=geometry,
            solver=solver,
            params=dict(data.get("params", {})),
            output=output,
            runtime=runtime,
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
    "OutputRequest",
    "RuntimeRequest",
    "SimulationRequest",
    "save_request",
    "load_request",
]



