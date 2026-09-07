# lid-driven cavity vs Ghia et al. (Re=100)
import numpy as np
import pytest
from src.lbm.advance import initial
from src.lbm.moments import macroscopic
from src.lbm.advance import collide, stream
from src.lbm.boundary_conditions import bounce_back, moving_wall

pytestmark = pytest.mark.slow
N = 64
U = 0.1
Re = 100
nu = U * N / Re
tau = 3 * nu + 0.5
steps = 15000
@pytest.fixture(scope="module")
def profile():
    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((N, N), bool)
    solid[0,:] = solid[-1,:] = solid[:,0] = solid[:,-1] = True
    lid = np.zeros((N, N), bool); lid[:,-1] = True
    stationary = solid & ~lid          # the 3 fixed walls

    f = initial(N, N)
    for _ in range(steps):
        f = collide(f, tau)            # from src.lbm.collision
        f = stream(f)
        f = bounce_back(f, stationary)
        f = moving_wall(f, lid, U)
    rho, u = macroscopic(f)

    ux = u[0, N//2, :] / U             # centerline u, normalized by lid speed
    y  = (np.arange(N) + 0.5) / N
    return y, ux

# test 1: lid drags fluid
def test_drag(profile):
    y, ux = profile
    assert ux[-3] > 0

# test 2: vortex structure
def test_vortex(profile):
    y, ux = profile
    assert ux.min() < 0
    assert ux.max() > 0
    assert ux[-2] > 0 
    assert ux[2] < 0

# test 3: interior magnitude check
def test_interior(profile):
    y, ux = profile
    ghia_min = -0.2058
    assert abs(ux.min() - (ghia_min)) / ghia_min < 0.05
    # n = 128 and < 0.01 passes in 70s