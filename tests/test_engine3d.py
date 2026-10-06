# 3D engine invariants: moments, collision, streaming, walls, inlet, outlet, wall models, layers, forces
import numpy as np
import pytest
from src.engine import lattice_d3q19 as d3q19
from src.engine import runtime
from src.engine.simulation3d import Simulation3D
from src.geometry.sdf import sdf_sphere, solid_from_sdf, q_from_sdf, node_grid, node_values
from src.turbulence.wall_function import friction_velocity

rng = np.random.default_rng(0)
grid_size = 16
east = int(np.where((d3q19.lattice_velocities == [1, 0, 0]).all(1))[0][0])
west = int(d3q19.opposite_direction[east])

@pytest.fixture(scope="module")
def sim():

    return Simulation3D(grid_size, grid_size, grid_size, "cpu", interp=True)

def _moments(populations):
    """
    Per-cell mass and momentum, so conservation errors can't cancel across the field.

    Returns density (N, N, N), momentum (3, N, N, N).
    """

    return populations.sum(axis=0), np.einsum("qc, qxyz -> cxyz", d3q19.lattice_velocities, populations)

def _load(sim, populations):
    """
    Load populations on a clean slate: no walls, no lid, no wall model, so every cell collides.
    """

    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.lid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.nut_wall.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.float32))
    sim.f.from_numpy(populations)

def _random_populations(low=0.5, high=1.5):
    """
    Random float32 populations on the fixture grid.

    Returns (19, N, N, N).
    """

    return rng.uniform(low, high, (d3q19.direction_count, grid_size, grid_size, grid_size)).astype(np.float32)

def _rest_equilibrium(nx=grid_size, ny=grid_size, nz=grid_size):
    """
    Rest equilibrium (density 1, zero velocity): the lattice weights in every cell.

    Returns float32 (19, nx, ny, nz).
    """

    return np.tile(d3q19.lattice_weights[:, None, None, None], (1, nx, ny, nz)).astype(np.float32)

def _equilibrium(density, velocity):
    """
    NumPy D3Q19 equilibrium, independent of the engine's feq. velocity is (3, ...) with any trailing shape.

    Returns float32 (19, ...).
    """

    velocity_dot_direction = np.einsum("qc,c...->q...", d3q19.lattice_velocities, velocity)
    velocity_squared = (velocity * velocity).sum(0)
    weights = d3q19.lattice_weights.reshape((-1,) + (1,) * (velocity.ndim - 1))

    return (weights * density * (1 + 3 * velocity_dot_direction + 4.5 * velocity_dot_direction * velocity_dot_direction - 1.5 * velocity_squared)).astype(np.float32)

def _nonequilibrium_stress(populations):
    """
    Pi_ab = sum_q c_qa c_qb (f_q - feq_q) from the populations' own density and velocity; populations are (19, ...).

    Returns (3, 3, ...) in float64.
    """

    populations = populations.astype(np.float64)
    density = populations.sum(axis=0)
    velocity = np.einsum("qc,q...->c...", d3q19.lattice_velocities, populations) / density
    nonequilibrium = populations - _equilibrium(density, velocity).astype(np.float64)

    return np.einsum("qa,qb,q...->ab...", d3q19.lattice_velocities, d3q19.lattice_velocities, nonequilibrium)

def _two_part_sphere():
    """
    Off-lattice sphere split at x = 8 into part 1 (front) and part 2 (back).

    Returns (phi, solid, parts).
    """

    phi = sdf_sphere(7.3, 8.1, 7.7, 4.2)
    solid = solid_from_sdf(phi, grid_size, grid_size, grid_size)
    parts = solid.copy()
    parts[8:] *= 2  # x >= 8 -> part 2, front stays part 1

    return phi, solid, parts

# test 1: uniform populations give density 19 and zero velocity
def test_macroscopic_density_and_velocity(sim):

    sim.f.from_numpy(np.ones((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32))
    sim.macroscopic()

    assert np.allclose(sim.rho.to_numpy(), 19)
    assert np.allclose(sim.u.to_numpy(), 0)

# test 2: collision leaves the rest equilibrium unchanged
def test_collide_equilibrium_is_unchanged(sim):

    populations = _rest_equilibrium()
    sim.f.from_numpy(populations)
    sim.collide(0.8)

    assert np.allclose(sim.f.to_numpy(), populations, atol=1e-5)

# test 3: BGK conserves mass and momentum per cell; periodic stream is an exact shift
def test_bgk_conserves_mass_momentum_and_stream_shifts(sim):

    populations = _random_populations()
    _load(sim, populations)
    sim.collide(rng.uniform(0.5, 1.5))
    collided = sim.f.to_numpy()
    density, momentum = _moments(populations)
    collided_density, collided_momentum = _moments(collided)
    sim.stream()
    streamed = sim.f.to_numpy()

    assert np.allclose(collided_density, density, atol=1e-4)
    assert np.allclose(collided_momentum, momentum, atol=1e-4)
    # f_new[q](x) = f[q](x - c_q)
    for q in range(d3q19.direction_count):
        assert np.array_equal(streamed[q], np.roll(collided[q], shift=tuple(d3q19.lattice_velocities[q]), axis=(0, 1, 2)))

# test 4: bounce-back swaps opposite populations on a solid node
def test_bounce_back_reverses_populations(sim):

    center = (5, 5, 5)
    populations = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    populations[1][center] = 5  # east at the cell
    populations[3][center] = 2  # west at the same cell
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[center] = 1
    sim.solid.from_numpy(solid)
    sim.f.from_numpy(populations)
    sim.bounce_back()
    bounced = sim.f.to_numpy()

    assert bounced[1][center] == 2  # east now holds west's old value
    assert bounced[3][center] == 5  # west now holds east's old value

# test 5: free-slip y changes the wall populations once and restores them when applied twice
def test_free_slip_is_involution(sim):

    populations = _random_populations()
    sim.f.from_numpy(populations)
    sim.free_slip_y()
    once = sim.f.to_numpy()
    sim.free_slip_y()
    twice = sim.f.to_numpy()

    assert not np.allclose(once, populations, atol=1e-6)  # a no-op kernel would fail here
    assert np.allclose(twice, populations, atol=1e-6)

# test 6: TRT conserves mass and momentum per cell
def test_trt_conserves_mass_and_momentum(sim):

    populations = _random_populations()
    _load(sim, populations)
    sim.collide_trt(rng.uniform(0.5, 1.5))
    density, momentum = _moments(populations)
    collided_density, collided_momentum = _moments(sim.f.to_numpy())

    assert np.allclose(collided_density, density, atol=1e-4)
    assert np.allclose(collided_momentum, momentum, atol=1e-4)

# test 7: LES is live (changes the result vs plain BGK) and still conserves per cell
def test_les_changes_result_and_conserves(sim):

    populations = _random_populations()
    relaxation_time = rng.uniform(0.6, 1.5)
    _load(sim, populations)
    sim.collide(relaxation_time)
    bgk = sim.f.to_numpy()
    _load(sim, populations)
    sim.collide_les(relaxation_time, 0.16)
    les = sim.f.to_numpy()
    density, momentum = _moments(populations)
    les_density, les_momentum = _moments(les)

    assert not np.allclose(les, bgk, atol=1e-6)  # the LES branch actually did something
    assert np.allclose(les_density, density, atol=1e-4)
    assert np.allclose(les_momentum, momentum, atol=1e-4)

# test 8: LES with a vanishing Smagorinsky constant matches BGK
def test_les_tiny_smagorinsky_constant_matches_bgk(sim):

    populations = _random_populations()
    relaxation_time = rng.uniform(0.5, 1.5)
    sim.f.from_numpy(populations)
    sim.collide(relaxation_time)
    bgk = sim.f.to_numpy()
    sim.f.from_numpy(populations)
    sim.collide_les(relaxation_time, 1e-9)
    les = sim.f.to_numpy()

    assert np.allclose(les, bgk, atol=1e-6)

# test 9: free-slip y keeps the tangential velocity and flips the y component
def test_free_slip_y_reflects_y_velocity(sim):

    velocity = np.zeros((3, grid_size, grid_size, grid_size))
    velocity[0] = 0.08
    velocity[1] = 0.05
    sim.f.from_numpy(_equilibrium(np.ones((grid_size, grid_size, grid_size)), velocity))
    sim.free_slip_y()
    sim.macroscopic()
    reflected = sim.u.to_numpy()
    wall_cell = (grid_size // 2, 0, grid_size // 2)  # a cell on the y = 0 wall

    assert np.isclose(reflected[0][wall_cell], 0.08, atol=1e-5)  # tangential kept
    assert np.isclose(reflected[1][wall_cell], -0.05, atol=1e-5)  # normal flipped

# test 10: free-slip z keeps the tangential velocity and flips the z component
def test_free_slip_z_reflects_z_velocity(sim):

    velocity = np.zeros((3, grid_size, grid_size, grid_size))
    velocity[0] = 0.08
    velocity[2] = 0.05
    sim.f.from_numpy(_equilibrium(np.ones((grid_size, grid_size, grid_size)), velocity))
    sim.free_slip_z()
    sim.macroscopic()
    reflected = sim.u.to_numpy()
    wall_cell = (grid_size // 2, grid_size // 2, 0)  # a cell on the z = 0 wall

    assert np.isclose(reflected[0][wall_cell], 0.08, atol=1e-5)  # tangential kept
    assert np.isclose(reflected[2][wall_cell], -0.05, atol=1e-5)  # normal flipped

# test 11: Bouzidi q < 1/2 with fluid upstream: f_west(x_f) = 2q f_east(x_f) + (1 - 2q) f_east(x_f - c_east)
# hand value at q = 0.25: 2 (0.25) (0.3) + (1 - 0.5) (0.9) = 0.15 + 0.45 = 0.6 (halfway would give 0.3)
def test_bouzidi_quarter_fraction_interpolates_upstream(sim):

    center = (5, 5, 5)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[6, 5, 5] = 1  # solid cell one step east of the center; (4, 5, 5) upstream stays fluid
    fractions = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    fractions[east][center] = 0.25
    populations = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    populations[east][center] = 0.3  # incoming population toward the wall
    populations[east][4, 5, 5] = 0.9  # same direction, one node upstream
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.body.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.set_wall_fractions(fractions)
    sim.f.from_numpy(populations)
    sim.fc.from_numpy(populations)  # kernel reads fc as post-collision
    sim.bounce_back_interp()
    reflected = sim.f.to_numpy()

    assert np.isclose(reflected[west][center], 0.6, atol=1e-6)

# test 12: drag_interp sums c_q (f_in + f_out) over boundary links
def test_drag_interp_sums_momentum_exchange(sim):

    center = (5, 5, 5)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[6, 5, 5] = 1
    fractions = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    fractions[east][center] = 0.5
    incoming = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    incoming[east][center] = 0.3
    reflected = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    reflected[west][center] = 0.2
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.set_wall_fractions(fractions)
    sim.fc.from_numpy(incoming)
    sim.f.from_numpy(reflected)
    sim.drag_interp()
    force = sim.force.to_numpy()

    # only link is east: F = c_q (f_in + f_out) = (1, 0, 0) (0.3 + 0.2)
    assert np.isclose(force[0], 0.5, atol=1e-6)
    assert np.isclose(force[1], 0.0, atol=1e-6)
    assert np.isclose(force[2], 0.0, atol=1e-6)

# test 13: inlet_neem imposes u = (U, 0, 0) with the x = 1 density, and copies the x = 1 non-equilibrium stress
# (second-order Hermite reconstruction reproduces Pi exactly and adds no mass or momentum)
def test_inlet_neem(sim):

    inlet_velocity = 0.1
    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    populations = _random_populations(0.9, 1.1)  # strongly non-equilibrium neighbour plane
    sim.f.from_numpy(populations)
    sim.inlet_neem(inlet_velocity)
    inlet_plane = sim.f.to_numpy()[:, 0].astype(np.float64)
    neighbour_plane = populations[:, 1].astype(np.float64)
    inlet_density = inlet_plane.sum(axis=0)
    inlet_velocity_field = np.einsum("qc,qjk->cjk", d3q19.lattice_velocities, inlet_plane) / inlet_density

    assert np.allclose(inlet_density, neighbour_plane.sum(axis=0), atol=1e-5)
    assert np.allclose(inlet_velocity_field[0], inlet_velocity, atol=1e-5)
    assert np.allclose(inlet_velocity_field[1:], 0.0, atol=1e-5)
    assert np.allclose(_nonequilibrium_stress(inlet_plane), _nonequilibrium_stress(neighbour_plane), atol=1e-5)

# test 14: inlet_neem_open drives open rows, leaves solid inlet columns alone
def test_inlet_neem_open(sim):

    inlet_velocity = 0.1
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[0:2, 0:4, :] = 1  # a solid block spanning the inlet columns
    sim.solid.from_numpy(solid)
    sim.f.from_numpy(_rest_equilibrium())
    solid_inlet_before = sim.f.to_numpy()[:, 0, 2, 0].copy()  # a solid inlet cell, pre-call
    sim.inlet_neem_open(inlet_velocity)
    populations = sim.f.to_numpy()
    sim.macroscopic()
    velocity = sim.u.to_numpy()

    assert np.allclose(velocity[0, 0, 6, :], inlet_velocity, atol=1e-3)  # open row (j = 6, above the block)
    assert np.allclose(populations[:, 0, 2, 0], solid_inlet_before, atol=1e-6)  # solid inlet column skipped

# test 15: outlet copies the second-to-last plane onto the last
def test_outlet(sim):

    populations = _random_populations()
    sim.f.from_numpy(populations)
    sim.outlet()
    after = sim.f.to_numpy()

    assert np.allclose(after[:, -1, :, :], after[:, -2, :, :])  # zero gradient at the exit
    assert np.allclose(after[:, :-1, :, :], populations[:, :-1, :, :])  # interior untouched

# test 16: TRT + LES + forcing together: mass exact per cell, and each cell's x-momentum
# goes up by exactly gx per step (Guo); the old antisymmetric-rate bug breaks this
def test_collide_full_combined(sim):

    body_force_x = 1e-2
    populations = _random_populations()
    _load(sim, populations)
    sim.collide_full(0.8, 0.1, body_force_x, 1)
    collided = sim.f.to_numpy()
    density, momentum = _moments(populations)
    collided_density, collided_momentum = _moments(collided)

    assert np.isfinite(collided).all()
    assert np.allclose(collided_density, density, atol=1e-4)
    assert np.allclose(collided_momentum[0] - momentum[0], body_force_x, atol=1e-4)
    assert np.allclose(collided_momentum[1:], momentum[1:], atol=1e-4)

# test 17: init_equilibrium sets the right moments (density and velocity recovered)
def test_init_equilibrium_moments(sim):

    velocity_x = np.full((grid_size, grid_size, grid_size), 0.05, np.float32)
    velocity_y = np.full((grid_size, grid_size, grid_size), -0.02, np.float32)
    velocity_z = np.full((grid_size, grid_size, grid_size), 0.01, np.float32)
    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.init_equilibrium(velocity_x, velocity_y, velocity_z)
    sim.macroscopic()
    density, velocity = sim.rho.to_numpy(), sim.u.to_numpy()

    assert np.allclose(density, 1, atol=1e-5)
    assert np.allclose(velocity[0], 0.05, atol=1e-5)
    assert np.allclose(velocity[1], -0.02, atol=1e-5)
    assert np.allclose(velocity[2], 0.01, atol=1e-5)

# test 18: an equilibrium state is a collision fixed point (no force); fails if
# init_equilibrium's feq and collide_full's feq ever drift apart
def test_equilibrium_is_collision_fixed_point(sim):

    velocity_x = (0.05 * (2 * rng.random((grid_size, grid_size, grid_size)) - 1)).astype(np.float32)
    velocity_y = (0.05 * (2 * rng.random((grid_size, grid_size, grid_size)) - 1)).astype(np.float32)
    velocity_z = (0.05 * (2 * rng.random((grid_size, grid_size, grid_size)) - 1)).astype(np.float32)
    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.init_equilibrium(velocity_x, velocity_y, velocity_z)
    equilibrium = sim.f.to_numpy()
    sim.collide(0.8)  # BGK, no force -> feq is the fixed point

    assert np.allclose(sim.f.to_numpy(), equilibrium, atol=1e-5)

# test 19: collision skips wall nodes, so bounce-back hands populations back unchanged
@pytest.mark.parametrize("kernel", ["full", "reg"])
def test_collide_skips_walls(sim, kernel):

    populations = _random_populations()
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[8, 8, 8] = 1
    sim.solid.from_numpy(solid)
    sim.lid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.nut_wall.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.float32))
    sim.f.from_numpy(populations)
    if kernel == "full":
        sim.collide_full(0.8, 0.1, 1e-4, 1)
    else:
        sim.collide_reg(0.8, 0.1, 0.0)
    collided = sim.f.to_numpy()

    assert np.array_equal(collided[:, 8, 8, 8], populations[:, 8, 8, 8])
    assert not np.allclose(collided[:, 0, 0, 0], populations[:, 0, 0, 0])

# test 20: nut_wall raises the local tau by 3 nu_t, at that node only, independent of LES
# (run at cs = 0 so it fails if the augmentation is gated inside the LES branch)
def test_nut_wall_augments_relaxation_time(sim):

    base_relaxation_time, wall_eddy_viscosity = 0.8, 0.03
    center = (8, 8, 8)
    perturbed = _rest_equilibrium()
    perturbed[5][center] += 0.05  # a non-equilibrium perturbation at one node
    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))

    sim.nut_wall.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.float32))
    sim.f.from_numpy(perturbed)
    sim.collide_full(base_relaxation_time, 0.0, 0.0, 0)
    base = sim.f.to_numpy()

    # raise tau by hand
    sim.f.from_numpy(perturbed)
    sim.collide_full(base_relaxation_time + 3 * wall_eddy_viscosity, 0.0, 0.0, 0)
    manual = sim.f.to_numpy()

    # raise tau through the wall field
    wall_field = np.zeros((grid_size, grid_size, grid_size), np.float32)
    wall_field[center] = wall_eddy_viscosity
    sim.nut_wall.from_numpy(wall_field)
    sim.f.from_numpy(perturbed)
    sim.collide_full(base_relaxation_time, 0.0, 0.0, 0)
    via_wall_field = sim.f.to_numpy()

    assert np.allclose(via_wall_field[:, 8, 8, 8], manual[:, 8, 8, 8], atol=1e-6)  # nut_wall == raising tau
    assert np.allclose(via_wall_field[:, 0, 0, 0], base[:, 0, 0, 0], atol=1e-6)  # untouched elsewhere

# test 21: wall_model sets nut_wall from the log-law u_tau at wall-adjacent nodes, gated on y+ > 30
def test_wall_model(sim):

    viscosity, y1 = 0.01, 10.0
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[:, 0, :] = 1  # bottom wall row
    sim.solid.from_numpy(solid)
    velocity = np.zeros((3, grid_size, grid_size, grid_size), np.float32)
    velocity[0, 4, 1, 4] = 0.737  # y+ ~ 50 (> 30): engages
    velocity[0, 6, 1, 6] = 0.02  # y+ < 30: stays off
    sim.u.from_numpy(velocity)
    sim.nut_wall.from_numpy(np.ones((grid_size, grid_size, grid_size), np.float32))  # nonzero -> check it resets
    sim.wall_model(viscosity, y1)
    wall_eddy_viscosity = sim.nut_wall.to_numpy()
    u_tau = friction_velocity(0.737, y1, viscosity)
    expected = u_tau * u_tau * y1 / 0.737 - viscosity

    assert np.isclose(wall_eddy_viscosity[4, 1, 4], expected, rtol=1e-4)  # engaged node
    assert wall_eddy_viscosity[6, 1, 6] == 0.0  # y+ < 30 -> off
    assert wall_eddy_viscosity[8, 8, 8] == 0.0  # interior (not adjacent) -> reset

# test 22: regularized + LES conserves mass and momentum per cell
def test_regularized_conserves_mass_and_momentum(sim):

    populations = _random_populations()
    _load(sim, populations)
    sim.collide_reg(rng.uniform(0.6, 1.5), 0.1, 0.0)
    density, momentum = _moments(populations)
    collided_density, collided_momentum = _moments(sim.f.to_numpy())

    assert np.allclose(collided_density, density, atol=1e-4)
    assert np.allclose(collided_momentum, momentum, atol=1e-4)

# test 23: regularized collision has the correct shear viscosity. A resolved, unforced shear wave
# u_x = U0 sin(k y) decays as exp(-nu k^2 t) with nu = c_s^2 (tau - 1/2). No forcing, fully periodic,
# well resolved -> regularized must match analytic and BGK
def test_regularized_shear_viscosity():

    nx, ny, nz = 4, 64, 4
    relaxation_time = 0.6
    analytic_viscosity = (relaxation_time - 0.5) / 3.0
    wavenumber = 2 * np.pi / ny
    amplitude = 0.01
    steps = 1500
    y = np.arange(ny)

    def measured_viscosity(collide):
        sim = Simulation3D(nx, ny, nz, "cpu")
        sim.solid.from_numpy(np.zeros((nx, ny, nz), np.int32))  # no walls: fully periodic
        velocity_x = np.broadcast_to((amplitude * np.sin(wavenumber * y))[None, :, None], (nx, ny, nz)).astype(np.float32).copy()
        zero = np.zeros((nx, ny, nz), np.float32)
        sim.init_equilibrium(velocity_x, zero.copy(), zero.copy())

        def mode_amplitude():
            sim.macroscopic()
            profile = sim.u.to_numpy()[0].mean(axis=(0, 2))  # u_x(y)

            return 2.0 / ny * np.sum(profile * np.sin(wavenumber * y))  # sin-mode amplitude

        start_amplitude = mode_amplitude()
        for _ in range(steps):
            collide(sim)
            sim.stream()
        end_amplitude = mode_amplitude()

        return -np.log(end_amplitude / start_amplitude) / (wavenumber * wavenumber * steps)

    regularized_viscosity = measured_viscosity(lambda sim: sim.collide_reg(relaxation_time, 0.0, 0.0))
    bgk_viscosity = measured_viscosity(lambda sim: sim.collide(relaxation_time))

    assert abs(regularized_viscosity / analytic_viscosity - 1) < 0.03  # regularized viscosity is correct
    assert abs(regularized_viscosity / bgk_viscosity - 1) < 0.02  # and matches BGK on a resolved wave

# test 24: wall_model detects a non-y wall normal and uses the wall-parallel speed: on an x-normal wall
# the model must drive off sqrt(u_y^2 + u_z^2) and ignore the normal u_x
def test_wall_model_x_wall():

    sim = Simulation3D(grid_size, grid_size, grid_size, "cpu")  # own sim: needs fresh zeroed fields
    viscosity, y1 = 0.01, 10.0
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[0, :, :] = 1  # wall at i = 0 -> +x normal
    sim.solid.from_numpy(solid)
    velocity = np.zeros((3, grid_size, grid_size, grid_size), np.float32)
    velocity[0, 1, 5, 5] = 0.3  # normal component -> must be excluded
    velocity[1, 1, 5, 5] = 0.737  # tangential (y): drives the model
    sim.u.from_numpy(velocity)
    sim.nut_wall.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.float32))
    sim.wall_model(viscosity, y1)
    wall_eddy_viscosity = sim.nut_wall.to_numpy()
    u_tau = friction_velocity(0.737, y1, viscosity)  # from the tangential speed only
    expected = u_tau * u_tau * y1 / 0.737 - viscosity

    assert np.isclose(wall_eddy_viscosity[1, 5, 5], expected, rtol=1e-4)  # normal component correctly excluded

# test 25: a second instance without interp has no Bouzidi fields and leaves the fixture untouched
def test_second_instance_keeps_fixture(sim):

    sim.f.fill(1.0)
    second = Simulation3D(4, 4, 4, "cpu")

    assert not hasattr(second, "q")
    assert not hasattr(second, "fc")
    assert np.all(sim.f.to_numpy() == 1.0)

# test 26: the runtime rejects switching backend
def test_runtime_rejects_backend_switch(sim):

    with pytest.raises(RuntimeError):
        runtime.init("cuda")

# test 27: fast (wall-list) wall model matches the full-grid wall model, with at least one engaged node
def test_wall_model_fast_matches_full_grid(sim):

    viscosity, y1 = 0.01, 10.0
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[:, 0, :] = 1
    velocity_x = np.zeros((grid_size, grid_size, grid_size), np.float32)
    velocity_x[4, 1, 4] = 0.737  # y+ ~ 50: engages
    velocity_x[6, 1, 6] = 0.02  # y+ < 30: stays off
    zero = np.zeros((grid_size, grid_size, grid_size), np.float32)
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.init_equilibrium(velocity_x, zero, zero)  # equilibrium moments are exact, so f carries this velocity
    sim.macroscopic()
    sim.nut_wall.from_numpy(zero)
    sim.wall_model(viscosity, y1)
    full_grid = sim.nut_wall.to_numpy().copy()
    sim.nut_wall.from_numpy(zero)
    sim.build_wall_list()
    sim.wall_y1.fill(y1)  # same first-node distance as the full-grid call
    sim.wall_model_fast(viscosity)
    fast = sim.nut_wall.to_numpy()

    assert full_grid[4, 1, 4] > 0.0  # the comparison below is not between two all-zero fields
    assert np.count_nonzero(full_grid) == 1
    assert np.allclose(fast, full_grid, rtol=1e-5, atol=1e-9)

# test 28: drag_body matches drag when the body is all the solid
def test_drag_body_matches_drag(sim):

    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    body = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[8, 8, 8] = 1
    body[8, 8, 8] = 1  # one body voxel = one solid voxel
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(body)
    sim.f.from_numpy(_random_populations(0.9, 1.1))
    sim.drag()
    all_solid_force = sim.force.to_numpy().copy()
    sim.drag_body()
    body_force = sim.force.to_numpy()

    assert np.allclose(all_solid_force, body_force)  # body == solid here -> identical force

# test 29: WALE matches an independent NumPy evaluation (central inside, one-sided at the domain edge)
def test_les_wale_matches_numpy(sim):

    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.lid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    velocity = rng.uniform(-0.05, 0.05, (3, grid_size, grid_size, grid_size)).astype(np.float32)
    sim.u.from_numpy(velocity)
    wale_constant = 0.5
    filter_width = 1.0
    epsilon = 1e-12
    sim.les_wale(wale_constant)
    engine_eddy_viscosity = sim.nut_les.to_numpy()

    # numpy reference: same WALE formula on the same velocity
    velocity64 = velocity.astype(np.float64)
    velocity_gradient = np.zeros((3, 3, grid_size, grid_size, grid_size))
    for a in range(3):
        for b in range(3):
            velocity_gradient[a, b] = np.gradient(velocity64[a], axis=b)  # d u_a / d x_b, unit spacing
    strain_rate = 0.5 * (velocity_gradient + velocity_gradient.transpose(1, 0, 2, 3, 4))
    gradient_squared = np.einsum("ac...,cb...->ab...", velocity_gradient, velocity_gradient)
    gradient_squared_trace = gradient_squared[0, 0] + gradient_squared[1, 1] + gradient_squared[2, 2]
    traceless_symmetric = 0.5 * (gradient_squared + gradient_squared.transpose(1, 0, 2, 3, 4))
    for a in range(3):
        traceless_symmetric[a, a] -= gradient_squared_trace / 3.0
    strain_contraction = np.einsum("ab...,ab...->...", strain_rate, strain_rate)
    traceless_contraction = np.einsum("ab...,ab...->...", traceless_symmetric, traceless_symmetric)
    reference_eddy_viscosity = (wale_constant * filter_width) ** 2 * traceless_contraction ** 1.5 / (strain_contraction ** 2.5 + traceless_contraction ** 1.25 + epsilon)

    # whole field: at the domain edge the engine clamps to the cell and halves the divisor, which is np.gradient's first-order one-sided edge
    assert np.allclose(engine_eddy_viscosity, reference_eddy_viscosity, atol=1e-6)

# test 30: WALE is zero on a quiescent field
def test_les_wale_zero_on_quiescent(sim):

    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.lid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.u.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.les_wale(0.5)

    assert np.allclose(sim.nut_les.to_numpy(), 0.0)

# test 31: pressure outlet sets rho = rho_out on the last plane, keeps the neighbour's velocity,
# and leaves every other plane untouched
def test_outlet_pressure_pins_density(sim):

    sim.solid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    populations = _random_populations(0.9, 1.1)
    sim.f.from_numpy(populations)
    sim.outlet_pressure(1.0)
    after = sim.f.to_numpy()
    outlet_density = after[:, -1].sum(axis=0)
    outlet_velocity = np.einsum("qc,qjk->cjk", d3q19.lattice_velocities, after[:, -1]) / outlet_density
    neighbour_velocity = np.einsum("qc,qjk->cjk", d3q19.lattice_velocities, after[:, -2]) / after[:, -2].sum(axis=0)

    assert np.allclose(outlet_density, 1.0, atol=1e-5)
    assert np.allclose(outlet_velocity, neighbour_velocity, atol=1e-5)
    assert np.array_equal(after[:, :-1], populations[:, :-1])

# test 32: fused layers in collide_reg == the separate kernels (collide_reg -> sponge_relax -> sponge_relax_mean)
def test_collide_reg_fused_layers():

    size = 12
    populations = rng.uniform(0.5, 1.5, (d3q19.direction_count, size, size, size)).astype(np.float32)
    layer_x = np.zeros((size, size), np.float32)
    layer_x[:3, :] = 0.1
    layer_z = np.zeros((size, size), np.float32)
    layer_z[:, :3] = 0.07
    inlet_velocity = 0.05
    mean_update_rate = 0.01

    separate = Simulation3D(size, size, size, "cpu")
    separate.f.from_numpy(populations)
    separate.rho_bar.fill(1.0)
    separate.collide_reg(0.8, 0.0, 0.0)  # layers off during the collide
    separate.sigma.from_numpy(layer_x)
    separate.sigma_z.from_numpy(layer_z)
    separate.sponge_relax(inlet_velocity)
    separate.sponge_relax_mean(mean_update_rate)

    fused = Simulation3D(size, size, size, "cpu")
    fused.f.from_numpy(populations)
    fused.rho_bar.fill(1.0)
    fused.sigma.from_numpy(layer_x)
    fused.sigma_z.from_numpy(layer_z)
    fused.collide_reg(0.8, 0.0, 0.0, inlet_velocity, mean_update_rate, 1)

    assert np.allclose(fused.f.to_numpy(), separate.f.to_numpy(), atol=1e-6)
    assert np.allclose(fused.u_bar.to_numpy(), separate.u_bar.to_numpy(), atol=1e-7)
    assert np.allclose(fused.rho_bar.to_numpy(), separate.rho_bar.to_numpy(), atol=1e-7)

# test 33: moving bottom wall (uw) + static top wall gives the exact linear Couette profile
def test_bounce_back_moving_wall_couette():

    nx, ny, nz = 6, 18, 6
    wall_velocity_x = 0.05
    relaxation_time = 0.8
    sim = Simulation3D(nx, ny, nz, "cpu")
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)
    wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
    wall_velocity[0, :, 0, :] = wall_velocity_x  # bottom wall moves in +x
    sim.uw.from_numpy(wall_velocity)
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(zero, zero, zero)

    # ~3 diffusion times H^2 / nu
    for _ in range(8000):
        sim.collide(relaxation_time)
        sim.stream()
        sim.bounce_back()
    sim.macroscopic()

    velocity_x = sim.u.to_numpy()[0, nx // 2, 1:-1, nz // 2]
    j = np.arange(1, ny - 1)
    channel_height = ny - 2  # walls sit halfway: y = 0.5 and y = ny - 1.5
    exact = wall_velocity_x * (ny - 1.5 - j) / channel_height
    cross_velocity = sim.u.to_numpy()[1:, :, 1:-1, :]  # y, z components, fluid rows only

    assert np.allclose(velocity_x, exact, atol=5e-5)  # 0.1% of U
    assert np.abs(cross_velocity).max() < 1e-4 * wall_velocity_x  # no wall-normal or spanwise flow (0.01% of U)

# test 34: Bouzidi moving wall reproduces Taylor-Couette flow (rotating inner, static outer cylinder)
def test_bouzidi_moving_wall_taylor_couette():

    size = 48
    depth = 4
    center_x = 23.4
    center_y = 23.7
    inner_radius = 8.3
    outer_radius = 20.6
    angular_velocity = 0.004  # inner wall speed omega R1 ~ 0.033

    def phi(x, y, z):
        offset_x = x - center_x
        offset_y = y - center_y
        radius = np.sqrt(offset_x * offset_x + offset_y * offset_y)

        return np.minimum(radius - inner_radius, outer_radius - radius)  # fluid only in the annulus

    sim = Simulation3D(size, size, depth, "cpu", interp=True)
    sim.solid.from_numpy(solid_from_sdf(phi, size, size, depth))
    sim.set_wall_fractions(q_from_sdf(phi, size, size, depth))

    # rigid rotation on both sides of the inner wall
    X, Y, Z = node_grid(size, size, depth)
    offset_x = X - center_x
    offset_y = Y - center_y
    radius = np.sqrt(offset_x * offset_x + offset_y * offset_y)
    wall_velocity = np.zeros((3, size, size, depth), np.float32)
    band = radius < inner_radius + 2.0
    wall_velocity[0][band] = -angular_velocity * offset_y[band]
    wall_velocity[1][band] = angular_velocity * offset_x[band]
    sim.uw.from_numpy(wall_velocity)
    zero = np.zeros((size, size, depth), np.float32)
    sim.init_equilibrium(zero, zero, zero)

    # ~8 diffusion times (R2 - R1)^2 / nu
    for _ in range(12000):
        sim.collide(0.8)
        sim.fc.copy_from(sim.f)
        sim.stream()
        sim.bounce_back_interp()
    sim.macroscopic()

    velocity = sim.u.to_numpy()
    azimuthal = (-offset_y * velocity[0] + offset_x * velocity[1]) / radius
    radial = (offset_x * velocity[0] + offset_y * velocity[1]) / radius
    coefficient_a = -angular_velocity * inner_radius * inner_radius / (outer_radius * outer_radius - inner_radius * inner_radius)
    coefficient_b = angular_velocity * inner_radius * inner_radius * outer_radius * outer_radius / (outer_radius * outer_radius - inner_radius * inner_radius)
    exact = coefficient_a * radius + coefficient_b / radius
    fluid = (radius > inner_radius + 0.5) & (radius < outer_radius - 0.5)
    azimuthal_error = azimuthal[fluid] - exact[fluid]
    rms_error = np.sqrt(np.mean(azimuthal_error * azimuthal_error)) / (angular_velocity * inner_radius)

    assert fluid.sum() > 4000
    assert rms_error < 0.01  # 1% of the wall speed, RMS
    assert np.abs(radial[fluid]).max() < 0.01 * angular_velocity * inner_radius  # no spurious radial flow

# test 35: Bouzidi falls back to halfway bounce-back when the upstream node is solid (thin gap)
def test_bouzidi_falls_back_to_halfway_when_upstream_is_solid(sim):

    center = (5, 5, 5)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[6, 5, 5] = 1
    solid[4, 5, 5] = 1
    fractions = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    fractions[east][center] = 0.25
    populations = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    populations[east][center] = 0.3
    populations[east][4, 5, 5] = 0.9
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.body.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.set_wall_fractions(fractions)
    sim.f.from_numpy(populations)
    sim.fc.from_numpy(populations)
    sim.bounce_back_interp()
    reflected = sim.f.to_numpy()[west][center]

    assert np.isclose(reflected, 0.3, atol=1e-6)

# test 36: drag_interp on a moving wall adds the Galilean term -u_w (f_in - f_out)
def test_drag_interp_moving_wall_galilean_term(sim):

    center = (5, 5, 5)
    wall_velocity = np.array([0.02, 0.01, 0.0], np.float32)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[6, 5, 5] = 1
    fractions = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    fractions[east][center] = 0.5
    velocity_field = np.zeros((3, grid_size, grid_size, grid_size), np.float32)
    velocity_field[:, 5, 5, 5] = wall_velocity
    velocity_field[:, 6, 5, 5] = wall_velocity
    incoming = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    incoming[east][center] = 0.3
    reflected = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    reflected[west][center] = 0.2
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(velocity_field)
    sim.body.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.set_wall_fractions(fractions)
    sim.fc.from_numpy(incoming)
    sim.f.from_numpy(reflected)
    sim.drag_interp()
    force = sim.force.to_numpy()
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))

    assert np.isclose(force[0], 0.5 - 0.02 * 0.1, atol=1e-6)
    assert np.isclose(force[1], -0.01 * 0.1, atol=1e-6)
    assert np.isclose(force[2], 0.0, atol=1e-6)

# test 37: WALE sees no velocity gradient when the fluid moves with a moving wall
def test_les_wale_uses_moving_wall_velocity(sim):

    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[:, 0, :] = 1
    velocity_field = np.zeros((3, grid_size, grid_size, grid_size), np.float32)
    velocity_field[0, :, 0, :] = 0.05
    zero = np.zeros((grid_size, grid_size, grid_size), np.float32)
    sim.solid.from_numpy(solid)
    sim.lid.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.uw.from_numpy(velocity_field)
    sim.init_equilibrium(np.full((grid_size, grid_size, grid_size), 0.05, np.float32), zero, zero)
    sim.macroscopic()
    sim.les_wale(0.5)
    subgrid_viscosity = sim.nut_les.to_numpy()
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))

    assert np.abs(subgrid_viscosity[:, 1, :]).max() < 1e-9

# test 38: staircase per-part forces sum to the drag_body total (two-part sphere)
def test_part_forces_sum_to_drag_body(sim):

    phi, solid, parts = _two_part_sphere()
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(parts)
    sim.f.from_numpy(_random_populations(0.9, 1.1))
    sim.drag_body()
    total = sim.force.to_numpy()
    part_force = sim.part_force.to_numpy()
    # each part carries ~480 of pressure force that cancels to a total of ~5, so float32 rounding in the
    # separately ordered atomic sums scales with the part magnitude, not the total
    rounding_tolerance = 1e-5 * (np.abs(part_force[1]) + np.abs(part_force[2])).max()

    assert np.allclose(part_force[1] + part_force[2], total, rtol=0.0, atol=rounding_tolerance)
    assert np.abs(part_force[1]).max() > 1e-3 and np.abs(part_force[2]).max() > 1e-3
    assert np.allclose(part_force[0], 0.0)

# test 39: Bouzidi per-part forces sum to the drag_interp total (two-part sphere)
def test_part_forces_sum_to_drag_interp(sim):

    phi, solid, parts = _two_part_sphere()
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(parts)
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.set_wall_fractions(q_from_sdf(phi, grid_size, grid_size, grid_size))  # after body: link_part is read from it
    populations = _random_populations(0.9, 1.1)
    sim.f.from_numpy(populations)
    sim.fc.from_numpy(populations)
    sim.drag_interp()
    total = sim.force.to_numpy()
    part_force = sim.part_force.to_numpy()
    # each part carries ~480 of pressure force that cancels to a total of ~5, so float32 rounding in the
    # separately ordered atomic sums scales with the part magnitude, not the total
    rounding_tolerance = 1e-5 * (np.abs(part_force[1]) + np.abs(part_force[2])).max()

    assert np.allclose(part_force[1] + part_force[2], total, rtol=0.0, atol=rounding_tolerance)
    assert np.abs(part_force[1]).max() > 1e-3 and np.abs(part_force[2]).max() > 1e-3
    assert np.allclose(part_force[0], 0.0)

# test 40: Bouzidi q > 1/2: f_west(x_f) = f_east / (2q) + (2q - 1) / (2q) f_west, both post-collision at x_f
# hand value at q = 0.75: 0.3 / 1.5 + (0.5 / 1.5) (0.2) = 0.2 + 0.0667 = 0.26667 (halfway would give 0.3)
def test_bouzidi_three_quarter_fraction(sim):

    center = (5, 5, 5)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[6, 5, 5] = 1
    fractions = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    fractions[east][center] = 0.75
    populations = np.zeros((d3q19.direction_count, grid_size, grid_size, grid_size), np.float32)
    populations[east][center] = 0.3
    populations[west][center] = 0.2
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(np.zeros((3, grid_size, grid_size, grid_size), np.float32))
    sim.body.from_numpy(np.zeros((grid_size, grid_size, grid_size), np.int32))
    sim.set_wall_fractions(fractions)
    sim.f.from_numpy(populations)
    sim.fc.from_numpy(populations)
    sim.bounce_back_interp()
    reflected = sim.f.to_numpy()

    assert np.isclose(reflected[west][center], 0.3 / 1.5 + 0.5 / 1.5 * 0.2, atol=1e-6)

# test 41: sponge_relax on its own: f -> f - sigma (f - feq(1, U, 0, 0)) where sigma > 0, fluid only; untouched elsewhere
def test_sponge_relax_matches_formula(sim):

    inlet_velocity = 0.05
    layer_strength = 0.2
    populations = _random_populations(0.9, 1.1)
    solid = np.zeros((grid_size, grid_size, grid_size), np.int32)
    solid[2, 7, 3] = 1  # a solid node inside the layer column stays untouched
    layer = np.zeros((grid_size, grid_size), np.float32)
    layer[2, 3] = layer_strength  # one (x, z) column
    sim.solid.from_numpy(solid)
    sim.sigma.from_numpy(layer)
    sim.f.from_numpy(populations)
    sim.sponge_relax(inlet_velocity)
    relaxed = sim.f.to_numpy()
    sim.sigma.from_numpy(np.zeros((grid_size, grid_size), np.float32))  # shared fixture: layers back off
    target = _equilibrium(np.ones(()), np.array([inlet_velocity, 0.0, 0.0]))  # (19,)
    column = populations[:, 2, :, 3]
    expected_column = column - layer_strength * (column - target[:, None])
    expected_column[:, 7] = column[:, 7]  # solid node skipped
    outside = np.ones((grid_size, grid_size, grid_size), bool)
    outside[2, :, 3] = False

    assert np.allclose(relaxed[:, 2, :, 3], expected_column, atol=1e-6)
    assert np.array_equal(relaxed[:, outside], populations[:, outside])

# test 42: f_absmax returns max |f|, and 1e30 when any population is NaN
def test_f_absmax(sim):

    populations = _random_populations(0.9, 1.1)
    populations[7, 3, 4, 5] = -5.0
    sim.f.from_numpy(populations)
    finite_max = sim.f_absmax()
    populations[2, 1, 1, 1] = np.nan
    sim.f.from_numpy(populations)
    nan_max = sim.f_absmax()

    assert np.isclose(finite_max, 5.0)
    assert nan_max >= 1e29

# test 43: a z-periodic Bouzidi wall is invariant under a one-layer shift in z (links and upstream nodes wrap)
def test_bouzidi_periodic_shift_invariance():

    nx, ny, nz = 24, 16, 8
    populations = rng.uniform(0.01, 0.1, (d3q19.direction_count, nx, ny, nz)).astype(np.float32)
    results = []

    # sphere through the z face (center z = 0.4), then the same sphere and populations one layer up
    for shift, center_z in ((0, 0.4), (1, 1.4)):

        def phi(x, y, z, center_z=center_z):

            dz = np.mod(z - center_z + nz / 2, nz) - nz / 2

            return np.sqrt((x - 12.0) * (x - 12.0) + (y - 8.0) * (y - 8.0) + dz * dz) - 3.0

        node_phi = node_values(phi, nx, ny, nz)
        sim = Simulation3D(nx, ny, nz, "cpu", interp=True, periodic=(True, True, True))
        sim.solid.from_numpy(solid_from_sdf(phi, nx, ny, nz, node_phi))
        sim.set_wall_fractions(q_from_sdf(phi, nx, ny, nz, node_phi=node_phi, periodic=(True, True, True)))
        shifted = np.roll(populations, shift, axis=3)
        sim.f.from_numpy(shifted)
        sim.fc.from_numpy(shifted)
        sim.bounce_back_interp()
        results.append(sim.f.to_numpy())

    assert np.array_equal(np.roll(results[0], 1, axis=3), results[1])

# test 44: wall-node normals of a z-invariant slab are the same on every layer with periodic z
def test_wall_list_periodic_layers_match():

    nx, ny, nz = 12, 12, 4
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, :5, :] = 1
    sim = Simulation3D(nx, ny, nz, "cpu", periodic=(False, False, True))
    sim.solid.from_numpy(solid)
    sim.build_wall_list()
    nodes = sim.wall_ijk.to_numpy()[:sim.n_wall]
    normals = sim.wall_n.to_numpy()[:sim.n_wall]
    face_layer = normals[(nodes[:, 2] == 0) & (nodes[:, 0] == 6)]
    middle_layer = normals[(nodes[:, 2] == 1) & (nodes[:, 0] == 6)]

    assert len(face_layer) == len(middle_layer) == 1
    assert np.allclose(face_layer, middle_layer)