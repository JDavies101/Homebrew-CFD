# BGK collision: conservation, relaxation time independence, equilibrium fixed point
import numpy as np
from src.lbm.collision import collide
from src.lbm.moments import macroscopic
from src.lbm.equilibrium import equilibrium

rng = np.random.default_rng(0)
# test 1: mass and momentum conservation
def test_mass_momentum_conservation():

    populations = rng.uniform(0.5, 1.5, (9, 4, 3))

    collided_populations = collide(populations, 1)

    density, velocity = macroscopic(populations)
    collided_density, collided_velocity = macroscopic(collided_populations)

    assert np.allclose(density, collided_density)
    assert np.allclose(velocity, collided_velocity)

# test 2: mass and momentum conservation for any valid relaxation time
def test_conservation_any_relaxation_time():

    for _ in range(5):
        relaxation_time = rng.uniform(0.5, 2)
        populations = rng.uniform(0.5, 1.5, (9, 4, 3))

        collided_populations = collide(populations, relaxation_time)

        density, velocity = macroscopic(populations)
        collided_density, collided_velocity = macroscopic(collided_populations)

        assert np.allclose(density, collided_density)
        assert np.allclose(velocity, collided_velocity)

# test 3: equilibrium is a fixed point of collision
def test_equilibrium_is_fixed_point():

    populations = rng.uniform(0.5, 1.5, (9, 4, 3))
    density, velocity = macroscopic(populations)
    equilibrium_populations = equilibrium(density, velocity)
    collided_populations = collide(equilibrium_populations, 1)

    assert np.allclose(collided_populations, equilibrium_populations)
