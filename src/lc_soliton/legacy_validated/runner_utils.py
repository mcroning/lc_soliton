
# -----------------------------------------------------------------------------
# Runner utilities
# -----------------------------------------------------------------------------


from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

try:
    import cupyx.scipy.fft as _spfft  # type: ignore
except Exception:  # pragma: no cover
    _spfft = None

from ..core.backend import _HAS_CUPY, _cupy
from ..validated_core.runner_core import (
    choose_optics_substeps as core_choose_optics_substeps,
    get_h_for_dz as core_get_h_for_dz,
    prepare_cn_ky_operator,
)
from ..core.context import LCContext, LCParams, LegacyPlans
__all__ = [
    "normalize_info",
    "_prepare_legacy_plans",
    "_prepare_substeps",
]

def normalize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(info or {})
    if "rrms" not in out and "rms_interior" in out:
        out["rrms"] = out["rms_interior"]
    if "rmax" not in out and "max_interior" in out:
        out["rmax"] = out["max_interior"]
    if "converged" not in out:
        out["converged"] = bool(out.get("accepted", False) or out.get("early_accepted", False) or out.get("n_selfcons_passes", 0) > 0)
    return out


def _prepare_legacy_plans(ctx: LCContext) -> LegacyPlans:
    if not (_HAS_CUPY and ctx.xp is _cupy):
        raise RuntimeError("Validated legacy kernels require CuPy/GPU.")
    if _spfft is None:
        raise RuntimeError("cupyx.scipy.fft is unavailable.")

    Ahat = ctx.xp.empty_like(ctx.amp0)
    return LegacyPlans(
        Ahat=Ahat,
        plan_f=_spfft.get_fft_plan(Ahat, axes=(-2, -1)),
        plan_i=_spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C"),
    )


def _prepare_substeps(ctx: LCContext, params: LCParams, *, use_core: bool):
    if use_core:
        nsub, dz_sub, phi = core_choose_optics_substeps(
            float(ctx.dz),
            kout=float(ctx.kout),
            dn_max_est=float(params.dn_max_est),
            dz_opt_max_phi=float(params.dz_opt_max_phi),
            max_substeps=int(params.max_substeps),
        )
        return int(nsub), float(dz_sub), float(phi), core_get_h_for_dz(ctx, dz_sub), core_get_h_for_dz(ctx, 0.5 * dz_sub)

    phi = abs(ctx.kout * ctx.dz * params.dn_max_est)
    nsub = max(1, min(int(params.max_substeps), int(math.ceil(phi / max(params.dz_opt_max_phi, 1e-12)))))
    dz_sub = ctx.dz / nsub
    return int(nsub), float(dz_sub), float(phi), _compute_h_local(ctx, dz_sub), _compute_h_local(ctx, 0.5 * dz_sub)