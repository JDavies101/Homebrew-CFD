# lid-driven cavity vs Ghia et al. (Re=100)
import numpy as np
import pytest
from src.lbm.advance import initial
from src.lbm.moments import macroscopic
from src.lbm.collision import collide
from src.lbm.stream import stream
from src.lbm.boundary_conditions import bounce_back, moving_wall

pytestmark = pytest.mark.slow
grid_size = 64
lid_velocity = 0.1
reynolds_number = 100
viscosity = lid_velocity * grid_size / reynolds_number  # nu
relaxation_time = 3 * viscosity + 0.5  # tau
steps = 15000
# run the solver once, share the centerline profile across all tests
@pytest.fixture(scope="module")
def profile():

    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((grid_size, grid_size), bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    lid = np.zeros((grid_size, grid_size), bool)
    lid[:, -1] = True
    stationary = solid & ~lid  # the 3 fixed walls

    populations = initial(grid_size, grid_size)
    for _ in range(steps):
        populations = collide(populations, relaxation_time, solid)
        populations = stream(populations)
        populations = bounce_back(populations, stationary)
        populations = moving_wall(populations, lid, lid_velocity)
    density, velocity = macroscopic(populations)

    # centerline x velocity, normalized by lid speed
    velocity_x = velocity[0, grid_size // 2, :] / lid_velocity
    y = (np.arange(grid_size) + 0.5) / grid_size

    return y, velocity_x

# test 1: lid drags the fluid below it
def test_lid_drags_fluid(profile):

    y, velocity_x = profile

    assert velocity_x[-3] > 0

# test 2: primary vortex gives reversed flow near the bottom
def test_vortex_structure(profile):

    y, velocity_x = profile

    assert velocity_x[1:-1].min() < 0  # fluid nodes only: wall rows hold bounced populations, not flow
    assert velocity_x[1:-1].max() > 0
    assert velocity_x[-2] > 0
    assert velocity_x[2] < 0

# test 3: minimum centerline velocity within 5% of Ghia (grid_size 128 passes at 1% in about 70 s)
def test_minimum_velocity_vs_ghia(profile):

    y, velocity_x = profile
    ghia_minimum = -0.2058

    # fluid nodes only (the wall rows' stored populations accumulate the lid term and are not flow); abs(): ghia_minimum < 0
    assert abs(velocity_x[1:-1].min() - ghia_minimum) / abs(ghia_minimum) < 0.05
