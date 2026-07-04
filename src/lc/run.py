"""Top-level run dispatcher for the clean LC package."""

from __future__ import annotations

from .request import TDRequest, StaticRequest, RunSummary
from .workflows.timedependent import run_timedependent
from .workflows.static import run_static

def run(request):
    if isinstance(request, TDRequest):
        return run_timedependent(request)
    if isinstance(request, StaticRequest):
        return run_static(request)
    raise TypeError(f"unsupported request type: {type(request).__name__}")


__all__ = ["run"]
