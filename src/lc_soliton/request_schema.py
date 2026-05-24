"""
Request schema introspection helpers.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from .request import SimulationRequest


def dataclass_schema(cls: type) -> dict[str, Any]:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass, got {cls!r}")

    return {
        field.name: {
            "type": getattr(field.type, "__name__", str(field.type)),
            "default": None if field.default.__class__.__name__ == "_MISSING_TYPE" else field.default,
        }
        for field in fields(cls)
    }


def simulation_request_schema() -> dict[str, Any]:
    """
    Return a lightweight schema description for SimulationRequest.
    """
    return dataclass_schema(SimulationRequest)


__all__ = [
    "dataclass_schema",
    "simulation_request_schema",
]
