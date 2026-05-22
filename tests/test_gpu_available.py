def test_cupy_gpu_available():
    import cupy as cp
    x = cp.arange(10)
    assert int(cp.sum(x).get()) == 45
