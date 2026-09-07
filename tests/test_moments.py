# density and velocity from known population states
import numpy as np
from src.lbm import moments

rng = np.random.default_rng(0)
nx = 4
ny = 3

# test 1: uniform populations
def test_uniform_populations():
    f = np.ones((9, nx, ny))
    rho, u = moments.macroscopic(f)
    assert np.allclose(rho, 9)
    assert np.allclose(u, 0)

# test 2: all mass in one direction
def test_mass_direction():
    f = np.zeros((9, nx, ny))
    f[1] = rng.integers(1,10)
    rho, u = moments.macroscopic(f)
    assert np.allclose(u[0], 1)
    assert np.allclose(u[1], 0)

# test 3: shape check
def test_shape_check():
    f = np.ones((9, nx, ny))
    rho, u = moments.macroscopic(f)
    assert rho.shape == (nx, ny)
    assert u.shape == (2, nx, ny)