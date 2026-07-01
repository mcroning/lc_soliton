from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np


def _get_float(z, key, default=np.nan):
    return float(z[key]) if key in z.files else float(default)


def load_ctx_and_profile(profile_path):
    profile_path = Path(profile_path)
    z = np.load(profile_path, allow_pickle=True)

    theta = cp.asarray(z["theta"], dtype=cp.float64)

    if "I" in z.files:
        I = cp.asarray(z["I"], dtype=cp.float64)
    else:
        A = cp.asarray(z["A"])
        I = (cp.abs(A) ** 2).astype(cp.float64)

    mode = {}
    x = z["x_um"]
    y = z["y_um"]
    d_um = x.max() - x.min()
    du = 2.0 * (x[1] - x[0]) / d_um
    dv = 2.0 * (y[1] - y[0]) / d_um

    ctx = SimpleNamespace(
        b=_get_float(z, "b"),
        bi=_get_float(z, "bi"),
        theta_bc=_get_float(z, "theta_bc", 0.0),
        du=du,
        dv=dv,
        mobility=_get_float(z, "mobility", 1.0),
    )

    if not np.isfinite(ctx.du) or not np.isfinite(ctx.dv):
        raise KeyError("Profile must contain du/dv or dx/dy for residual tests.")

    return ctx, mode, theta, I