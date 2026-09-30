# 3D Poiseuille on the taichi engine, same analytic parabola
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D
from src.engine import lattice_d3q19 as d3q19

# validation gate: force-driven channel flow should be an exact parabola
pytestmark = pytest.mark.slow
nx = 8
ny = 32
nz = 8
relaxation_time = 0.8
body_force_x = 1e-6  # g
steps = 40000
viscosity = (relaxation_time - 0.5) / 3

def _channel_profile(collide):
    """
    Run the forced channel (walls at y = 0 and y = ny - 1, periodic in x and z) with the given collision call.

    Returns (y, u_x) on the fluid nodes of the mid column, Guo half-force corrected.
    """

    sim = Simulation3D(nx, ny, nz, "cpu")
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)

    # start at rest equilibrium, drive with the body force in +x
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, nx, ny, nz)).astype(np.float32))
    for _ in range(steps):
        collide(sim)
        sim.stream()
        sim.bounce_back()
    sim.macroscopic()

    velocity = sim.u.to_numpy()
    density = sim.rho.to_numpy()
    column, k_mid = nx // 2, nz // 2
    velocity_x = velocity[0, column, :, k_mid] + body_force_x / (2 * density[column, :, k_mid])  # Guo half-force correction
    fluid = ~solid[column, :, k_mid].astype(bool)
    y = np.arange(ny)[fluid]

    return y, velocity_x[fluid]

# run the solver once, share the profile across tests 1-3
@pytest.fixture(scope="module")
def profile():

    return _channel_profile(lambda sim: sim.collide_forced(relaxation_time, body_force_x))

# test 1: profile is parabolic (R^2 ~ 1)
def test_profile_is_parabolic(profile):

    y, velocity_x = profile
    fit = np.polyval(np.polyfit(y, velocity_x, 2), y)
    residual = velocity_x - fit
    deviation = velocity_x - velocity_x.mean()
    r_squared = 1 - np.sum(residual * residual) / np.sum(deviation * deviation)

    assert r_squared > 0.9999

# test 2: profile is symmetric about the channel center
def test_profile_is_symmetric(profile):

    y, velocity_x = profile

    assert np.allclose(velocity_x, velocity_x[::-1], atol=1e-6)  # looser tolerance for fp32 instead of fp64

# test 3: peak matches g L^2 / (8 nu), walls taken from the fit
def test_peak_velocity_vs_analytic(profile):

    y, velocity_x = profile
    roots = np.sort(np.roots(np.polyfit(y, velocity_x, 2)))
    channel_width = roots[1] - roots[0]
    analytic_peak = body_force_x * channel_width * channel_width / (8 * viscosity)

    assert abs(velocity_x.max() - analytic_peak) / analytic_peak < 0.01

# test 4: forced Poiseuille under the regularized operator certifies the Guo force-Pi correction
def test_regularized_poiseuille_peak():

    y, velocity_x = _channel_profile(lambda sim: sim.collide_reg(relaxation_time, 0.0, body_force_x))
    roots = np.sort(np.roots(np.polyfit(y, velocity_x, 2)))
    channel_width = roots[1] - roots[0]
    analytic_peak = body_force_x * channel_width * channel_width / (8 * viscosity)

    assert abs(velocity_x.max() - analytic_peak) / analytic_peak < 0.015
