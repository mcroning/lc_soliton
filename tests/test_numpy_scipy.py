def test_numpy_scipy_available():
    import numpy as np
    from scipy.stats import linregress

    x = np.arange(5.0)
    y = 2.0 * x + 1.0
    fit = linregress(x, y)

    assert abs(fit.slope - 2.0) < 1e-12
    assert abs(fit.intercept - 1.0) < 1e-12
