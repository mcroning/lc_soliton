import numpy as np

from lc_soliton.core.bias import (
    theta_bias_1d_exact_zero_bc,
    theta_bias_2d_from_params,
)
from lc_soliton.legacy_validated.lc_engine_validated import LCParams


def test_theta_bias_exact_zero_bc_has_zero_boundaries():
    th = theta_bias_1d_exact_zero_bc(128, b=2.487025357142857, xp=np)

    assert th.shape == (128,)
    assert th[0] == 0.0
    assert th[-1] == 0.0
    assert th.max() > 0.0


def test_theta_bias_2d_from_params_shape_and_boundaries():
    p = LCParams(Nx=64, Ny=32, b=2.487025357142857, theta_bc=0.0)
    th = theta_bias_2d_from_params(p, np)

    assert th.shape == (64, 32)
    assert np.allclose(th[0, :], 0.0)
    assert np.allclose(th[-1, :], 0.0)
    assert th.max() > 0.0