# 2D Poiseuille: parabolic profile, symmetry, peak vs analytic
import numpy as np
import pytest
from src.lbm.advance import initial, run
from src.lbm.moments import macroscopic

# validation gate: force-driven channel flow should be an exact parabola
pytestmark = pytest.mark.slow

nx = 8
ny = 32
tau = 0.8
g = 1e-6
steps = 40000
nu = (tau - 0.5) / 3

# run the solver once, share the profile across all tests
@pytest.fixture(scope="module")
def profile():
    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True

    f = run(initial(nx, ny), tau, solid, steps, g)
    rho, u_raw = macroscopic(f)

    col = nx // 2
    ux = u_raw[0, col, :] + g / (2 * rho[col, :])   # Guo half-force correction

    fluid = ~solid[col]
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
    assert np.allclose(ux, ux[::-1], atol=1e-9)

# test 3: peak matches g*L^2/(8 nu), walls taken from the fit
def test_peak(profile):
    y, ux = profile
    coeffs = np.polyfit(y, ux, 2)
    roots = np.sort(np.roots(coeffs))
    L = roots[1] - roots[0]
    u_max = g * L ** 2 / (8 * nu)
    assert abs(ux.max() - u_max) / u_max < 0.01
