# 2D Poiseuille: parabolic profile, symmetry, peak vs analytic
import numpy as np
import pytest
from src.lbm.advance import initial, run
from src.lbm.moments import macroscopic

# validation gate: force-driven channel flow should be an exact parabola
pytestmark = pytest.mark.slow
nx = 8
ny = 32
relaxation_time = 0.5 + np.sqrt(3) / 4  # tau, BGK magic value: halfway bounce-back exact for Poiseuille
body_force_x = 1e-6  # g
steps = 40000
viscosity = (relaxation_time - 0.5) / 3  # nu
# run the solver once, share the profile across all tests
@pytest.fixture(scope="module")
def profile():

    solid = np.zeros((nx, ny), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True

    populations = run(initial(nx, ny), relaxation_time, solid, steps, body_force_x)
    density, raw_velocity = macroscopic(populations)

    column = nx // 2
    velocity_x = raw_velocity[0, column, :] + body_force_x / (2 * density[column, :])  # Guo half-force correction

    fluid = ~solid[column]
    y = np.arange(ny)[fluid]

    return y, velocity_x[fluid]

# test 1: profile is parabolic (R^2 ~ 1)
def test_profile_is_parabolic(profile):

    y, velocity_x = profile
    coefficients = np.polyfit(y, velocity_x, 2)
    fit = np.polyval(coefficients, y)
    residual = velocity_x - fit
    deviation = velocity_x - velocity_x.mean()
    r_squared = 1 - np.sum(residual * residual) / np.sum(deviation * deviation)

    assert r_squared > 0.9999

# test 2: profile is symmetric about the channel center
def test_profile_is_symmetric(profile):

    y, velocity_x = profile

    assert np.allclose(velocity_x, velocity_x[::-1], atol=1e-9)

# test 3: peak matches g L^2 / (8 nu), walls taken from the fit
def test_peak_velocity_vs_analytic(profile):

    y, velocity_x = profile
    coefficients = np.polyfit(y, velocity_x, 2)
    roots = np.sort(np.roots(coefficients))
    channel_width = roots[1] - roots[0]
    analytic_peak_velocity = body_force_x * channel_width * channel_width / (8 * viscosity)

    assert abs(velocity_x.max() - analytic_peak_velocity) / analytic_peak_velocity < 0.01

# test 4: walls sit exactly at the halfway points (not taken from the fit)
def test_wall_location(profile):

    y, velocity_x = profile
    roots = np.sort(np.roots(np.polyfit(y, velocity_x, 2)))

    assert abs(roots[0] - 0.5) < 0.01
    assert abs(roots[1] - (ny - 1.5)) < 0.01
