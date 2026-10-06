# 3D Poiseuille on the taichi engine, same analytic parabola
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D
from src.engine import lattice_d3q19 as d3q19
from src.geometry.sdf import node_values, solid_from_sdf, q_from_sdf

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

def _bouzidi_channel(relaxation_time, wall_fraction):
    """
    Forced channel between two Bouzidi plates at y = 1 - q and y = ny - 2 + q, regularized collision and the
    runner's step order (collide, snapshot, stream, bounce back, Bouzidi).

    Returns (bottom wall error, top wall error) in cells from the fitted parabola's roots, the fitted peak's relative
    error against g W^2 / (8 nu), and the measured wall force's relative error against the total body force.
    """

    channel_nx = 4
    channel_ny = 12
    channel_nz = 4
    channel_viscosity = (relaxation_time - 0.5) / 3
    bottom_wall = 1.0 - wall_fraction
    top_wall = channel_ny - 2 + wall_fraction
    width = top_wall - bottom_wall
    force = 0.08 * channel_viscosity / (width * width)  # peak u = g W^2 / (8 nu) = 0.01
    step_count = int(8 * width * width / (np.pi * np.pi * channel_viscosity))

    sim = Simulation3D(channel_nx, channel_ny, channel_nz, "cpu", interp=True, periodic=(True, False, True))
    solid = np.zeros((channel_nx, channel_ny, channel_nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(solid)  # plates are part 1, so drag_interp sums their links

    # every wall-ward link crosses the flat plate at the same fraction q
    wall_fractions = np.zeros((d3q19.direction_count, channel_nx, channel_ny, channel_nz), np.float32)
    for d in range(d3q19.direction_count):
        if d3q19.lattice_velocities[d, 1] == -1:
            wall_fractions[d, :, 1, :] = wall_fraction
        if d3q19.lattice_velocities[d, 1] == 1:
            wall_fractions[d, :, -2, :] = wall_fraction
    sim.set_wall_fractions(wall_fractions)

    # start at rest, then the production step order
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, channel_nx, channel_ny, channel_nz)).astype(np.float32))
    for _ in range(step_count):
        sim.collide_reg(relaxation_time, 0.0, force)
        sim.fc.copy_from(sim.f)
        sim.stream()
        sim.bounce_back()
        sim.bounce_back_interp()
    sim.drag_interp()
    sim.macroscopic()

    # mid column, Guo half-force corrected, fluid rows only
    column = channel_nx // 2
    layer = channel_nz // 2
    density = sim.rho.to_numpy()[column, 1:-1, layer]
    velocity_x = sim.u.to_numpy()[0, column, 1:-1, layer] + force / (2 * density)
    y = np.arange(1, channel_ny - 1, dtype=np.float64)
    coefficients = np.polyfit(y, velocity_x, 2)
    roots = np.sort(np.roots(coefficients).real)
    fitted_peak = coefficients[2] - coefficients[1] * coefficients[1] / (4 * coefficients[0])
    analytic_peak = force * width * width / (8 * channel_viscosity)

    # steady state: the walls carry all the body force put into the fluid
    body_force_total = force * channel_nx * (channel_ny - 2) * channel_nz
    wall_force_x = float(sim.force.to_numpy()[0])

    return roots[0] - bottom_wall, roots[1] - top_wall, fitted_peak / analytic_peak - 1, wall_force_x / body_force_total - 1

# test 5: Bouzidi plates sit where q puts them under the regularized operator, at a reference and the production tau
@pytest.mark.parametrize("relaxation_time", [0.6, 0.5024])
@pytest.mark.parametrize("wall_fraction", [0.5, 0.25, 0.1, 0.75])
def test_bouzidi_channel_wall_location(relaxation_time, wall_fraction):

    bottom_error, top_error, peak_error, force_error = _bouzidi_channel(relaxation_time, wall_fraction)
    print(f"tau {relaxation_time} q {wall_fraction}: wall error {bottom_error:+.4f} / {top_error:+.4f} cells, "
          f"peak {peak_error:+.2%}, wall force {force_error:+.3%}")

    assert abs(bottom_error) < 0.06
    assert abs(top_error) < 0.06
    assert abs(peak_error) < 0.03
    assert abs(force_error) < 0.01

def _tilted_band(relaxation_time, offset):
    """
    Force-driven flow in a band at slope 1/16 between Bouzidi walls, periodic in x and y so the walls cross every
    sub-cell position along x; regularized collision and the runner's step order.

    Returns (lower wall error, upper wall error) in cells along the wall normal from the fitted along-band profile,
    the RMS scatter of the nodes about that fit relative to the peak, and the wall force's relative error against
    the total body force.
    """

    band_nx = 256
    band_ny = 16
    band_nz = 4
    slope = band_ny / band_nx
    cosine = 1.0 / np.sqrt(1.0 + slope * slope)
    sine = slope * cosine
    band_height = 10.0  # vertical width of the fluid band, cells
    normal_width = band_height * cosine
    band_viscosity = (relaxation_time - 0.5) / 3
    force = 0.4 * band_viscosity / (normal_width * normal_width * cosine)  # along-band peak about 0.05: signal well above float32 rounding in the wall-force sum
    step_count = int(8 * normal_width * normal_width / (np.pi * np.pi * band_viscosity))

    # signed distance to the band walls: > 0 inside the band; periodic because the band rises ny over nx
    def phi(x, y, z):

        height = np.mod(y - slope * x - offset, band_ny)
        inside = np.minimum(height, band_height - height)
        outside = -np.minimum(height - band_height, band_ny - height)

        return cosine * np.where(height < band_height, inside, outside)

    node_phi = node_values(phi, band_nx, band_ny, band_nz)
    solid = solid_from_sdf(phi, band_nx, band_ny, band_nz, node_phi)
    sim = Simulation3D(band_nx, band_ny, band_nz, "cpu", interp=True, periodic=(True, True, True))
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(solid)  # walls are part 1, so drag_interp sums their links
    sim.set_wall_fractions(q_from_sdf(phi, band_nx, band_ny, band_nz, node_phi=node_phi, periodic=(True, True, True)))

    # start at rest, then the production step order
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, band_nx, band_ny, band_nz)).astype(np.float32))
    for _ in range(step_count):
        sim.collide_reg(relaxation_time, 0.0, force)
        sim.fc.copy_from(sim.f)
        sim.stream()
        sim.bounce_back()
        sim.bounce_back_interp()
    sim.drag_interp()
    sim.macroscopic()

    # along-band speed against the wall-normal coordinate, every fluid node of one layer, Guo half-force corrected
    velocity = sim.u.to_numpy()[:, :, :, 0]
    density = sim.rho.to_numpy()[:, :, 0]
    fluid = solid[:, :, 0] == 0
    x, y = np.nonzero(fluid)
    velocity_x = velocity[0][fluid] + force / (2 * density[fluid])
    along = velocity_x * cosine + velocity[1][fluid] * sine
    normal = cosine * np.mod(y - slope * x - offset, band_ny)
    coefficients = np.polyfit(normal, along, 2)
    roots = np.sort(np.roots(coefficients).real)
    scatter = along - np.polyval(coefficients, normal)
    peak = coefficients[2] - coefficients[1] * coefficients[1] / (4 * coefficients[0])

    # steady state: the walls carry all the body force put into the fluid
    body_force_total = force * int(fluid.sum()) * band_nz
    wall_force_x = float(sim.force.to_numpy()[0])

    return roots[0], roots[1] - normal_width, np.sqrt(np.mean(scatter * scatter)) / peak, wall_force_x / body_force_total - 1

# test 6: shallow-angle Bouzidi walls (slope 1/16) keep the analytic band profile at every sub-cell position
@pytest.mark.parametrize("relaxation_time", [0.6, 0.5024])
@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5])
def test_tilted_band_profile(relaxation_time, offset):

    lower_error, upper_error, scatter, force_error = _tilted_band(relaxation_time, offset)
    print(f"tau {relaxation_time} offset {offset}: wall error {lower_error:+.4f} / {upper_error:+.4f} cells, "
          f"scatter {scatter:.3%}, wall force {force_error:+.3%}")

    assert abs(lower_error) < 0.06
    assert abs(upper_error) < 0.06
    assert scatter < 0.01
    assert abs(force_error) < 0.03  # near tau = 1/2 the shallow-wall force balance converges to about -2% (measured)

# test 7: a channel between two plates thinner than a cell: walls where the plates are, no leak, force on the part
def test_thin_plate_channel():

    nx, ny, nz = 4, 16, 4
    plate_relaxation_time = 0.6
    plate_viscosity = (plate_relaxation_time - 0.5) / 3
    bottom_wall = 2.55
    top_wall = 13.45
    width = top_wall - bottom_wall
    force = 0.08 * plate_viscosity / (width * width)  # peak u = g W^2 / (8 nu) = 0.01
    step_count = int(8 * width * width / (np.pi * np.pi * plate_viscosity))

    # two plates 0.1 cells thick at y = 2.5 and 13.5: no node falls inside either
    def phi(x, y, z):

        return np.minimum(np.abs(y - 2.5), np.abs(y - 13.5)) - 0.05

    wall_fractions = q_from_sdf(phi, nx, ny, nz, periodic=(True, True, True))
    sim = Simulation3D(nx, ny, nz, "cpu", interp=True, periodic=(True, True, True))
    sim.solid.from_numpy(np.zeros((nx, ny, nz), np.int32))
    sim.set_wall_fractions(wall_fractions, (wall_fractions > 0.0).astype(np.int8))

    # start at rest, then the production step order
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, nx, ny, nz)).astype(np.float32))
    for _ in range(step_count):
        sim.collide_reg(plate_relaxation_time, 0.0, force)
        sim.fc.copy_from(sim.f)
        sim.stream()
        sim.bounce_back()
        sim.bounce_back_interp()
    sim.drag_interp()
    sim.macroscopic()

    # inner band (rows 3-13), Guo half-force corrected; every node is fluid, so all of them are driven
    column = nx // 2
    layer = nz // 2
    density = sim.rho.to_numpy()[column, 3:14, layer]
    velocity_x = sim.u.to_numpy()[0, column, 3:14, layer] + force / (2 * density)
    y = np.arange(3, 14, dtype=np.float64)
    roots = np.sort(np.roots(np.polyfit(y, velocity_x, 2)).real)
    body_force_total = force * nx * ny * nz
    force_error = float(sim.part_force.to_numpy()[1, 0]) / body_force_total - 1

    assert abs(roots[0] - bottom_wall) < 0.06
    assert abs(roots[1] - top_wall) < 0.06
    assert abs(force_error) < 0.01