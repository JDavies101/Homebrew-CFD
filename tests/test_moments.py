# density and velocity from known population states
import numpy as np
from src.lbm.moments import macroscopic

rng = np.random.default_rng(0)
nx = 4
ny = 3
# test 1: uniform populations give density 9 and zero velocity
def test_uniform_populations():

    populations = np.ones((9, nx, ny))
    density, velocity = macroscopic(populations)

    assert np.allclose(density, 9)
    assert np.allclose(velocity, 0)

# test 2: all mass in the east direction gives unit x velocity
def test_all_mass_east():

    populations = np.zeros((9, nx, ny))
    populations[1] = rng.integers(1, 10)
    density, velocity = macroscopic(populations)

    assert np.allclose(velocity[0], 1)
    assert np.allclose(velocity[1], 0)

# test 3: output shapes
def test_output_shapes():

    populations = np.ones((9, nx, ny))
    density, velocity = macroscopic(populations)

    assert density.shape == (nx, ny)
    assert velocity.shape == (2, nx, ny)
