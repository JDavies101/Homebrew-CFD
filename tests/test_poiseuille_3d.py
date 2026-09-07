# 3D Poiseuille on the taichi engine, same analytic parabola
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D as S
from src.engine import lattice3d as L

# validation gate: force-driven channel flow should be an exact parabola
pytestmark = pytest.mark.slow

nx = 8
ny = 32
nz = 8
tau = 0.8
g = 1e-6
steps = 40000
nu = (tau - 0.5) / 3

# run the solver once, share the profile across all tests
@pytest.fixture(scope="module")
def profile():
    sim = S(nx, ny, nz, "cpu")

    # walls at y=0 and y=ny-1
    # periodic in x and z
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)

    # init and rest equilibrium
    # drive with body force g in +x direction
    sim.f.from_numpy(np.tile(L.W[:, None, None, None], (1, nx, ny, nz)).astype(np.float64))
    for _ in range(steps):
        sim.collide_forced(tau, g)
        sim.stream()
        sim.bounce_back()

    sim.macroscopic()
    u = sim.u.to_numpy()
    rho = sim.rho.to_numpy()

    col, kmid = nx // 2, nz // 2
    ux = u[0, col, :, kmid] + g / (2 * rho[col, :, kmid]) # guo half-force correction
    fluid = ~solid[col, :, kmid].astype(bool)
    y = np.arange(ny)[fluid]

    return y, ux[fluid]

# test 1: profile is parabolic (R^2 ~ 1)
def test_parabolic(profile):
    y, ux = profile
    coeffs = np.polyfit(y, ux, 2)
    fit = np.polyval(coeffs, y)
    r2 = 1 - np.sum((ux - fit) ** 2) / np.sum((ux - ux.mean()) ** 2)
    assert r2 > 0.9999

# test 2: profile is symmetric about the channel center
def test_symmetric(profile):
    y, ux = profile
    assert np.allclose(ux, ux[::-1], atol=1e-6) # looser tolerance for fp32 instead of fp64

# test 3: peak matches g*L^2/(8 nu), walls taken from the fit
def test_peak(profile):
    
    y, ux = profile
    coeffs = np.polyfit(y, ux, 2)
    roots = np.sort(np.roots(coeffs))
    L = roots[1] - roots[0]
    u_max = g * L ** 2 / (8 * nu)
    assert abs(ux.max() - u_max) / u_max < 0.01
