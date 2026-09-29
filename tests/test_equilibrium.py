# equilibrium: moment round-trip and a hand-computed value
import numpy as np
from src.lbm.moments import macroscopic
from src.lbm.equilibrium import equilibrium
from src.lbm import lattice as lt

nx = 4
ny = 3
# test 1: moments of the equilibrium return the input density and velocity
def test_moment_round_trip():

    density = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    velocity = np.full((2, nx, ny), 0.05)
    equilibrium_populations = equilibrium(density, velocity)
    round_trip_density, round_trip_velocity = macroscopic(equilibrium_populations)

    assert np.allclose(round_trip_density, density)
    assert np.allclose(round_trip_velocity, velocity)

# test 2: zero velocity gives weights times density
def test_zero_velocity_gives_weighted_density():

    velocity = np.zeros((2, nx, ny))
    density = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int32)
    equilibrium_populations = equilibrium(density, velocity)

    assert np.allclose(lt.lattice_weights[:, None, None] * density, equilibrium_populations)

# test 3: hand-computed value
def test_hand_computed_value():

    density = np.array([[1]], dtype=np.int32)
    velocity = np.zeros((2, 1, 1))
    velocity[0, 0, 0] = 0.1
    equilibrium_populations = equilibrium(density, velocity)

    # density 1, velocity (0.1, 0), direction 1 (east): (1/9) * (1 + 0.3 + 0.045 - 0.015) = 1.33/9
    assert np.allclose(equilibrium_populations[1, 0, 0], 1.33 / 9)
