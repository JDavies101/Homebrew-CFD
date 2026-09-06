import numpy as np
from src.lbm.stream import stream

rng = np.random.default_rng(0)
nx = 4
ny = 3
# test 1: global mass conservation
def test_stream():
    f = rng.uniform(0.5, 1.5, (9, nx, ny))
    f_new = stream(f)
    assert np.allclose(f.sum(), f_new.sum())

# test 2: moving blob
def test_moving_blob():
    f = np.zeros((9, nx, ny))
    f[5, 1, 1] = 1
    f_new = stream(f)
    assert np.allclose(f_new[5, 2, 2], 1)

# test 3: rest doesn't move
def test_rest():
    f = np.zeros((9, nx, ny))
    f_new = stream(f)
    assert np.allclose(f_new[0], f[0])