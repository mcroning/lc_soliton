
"""

Canonical public execution entry points.

"""

from __future__ import annotations

from .static import run_static

from .timedependent import run_td

from .dualgrid import run_dg_td

ENGINE_MODES = {

    "strict_static": run_static,

    "td_predictor_only": run_td,

    "dg_td_predictor": run_dg_td,

}

def run_engine(mode: str, *args, **kwargs):

    """

    Canonical public execution dispatcher.

    """

    if mode not in ENGINE_MODES:

        raise ValueError(

            f"Unknown engine mode: {mode}. "

            f"Available modes: {sorted(ENGINE_MODES)}"

        )

    return ENGINE_MODES[mode](*args, **kwargs)

