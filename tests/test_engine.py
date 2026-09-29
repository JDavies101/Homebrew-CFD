# 2D Taichi engine parity against the NumPy reference solver (src/lbm)
import numpy as np
import pytest
from src.engine.simulation import Simulation
from src.engine.lattice_d2q9 import lattice_weights
from src.lbm.moments import macroscopic as numpy_macroscopic
from src.lbm.collision import collide as numpy_collide
from src.lbm.stream import stream as numpy_stream
from src.lbm.boundary_conditions import bounce_back as numpy_bounce_back
from src.lbm.boundary_conditions import moving_wall as numpy_moving_wall
from src.lbm.advance import initial as numpy_initial

rng = np.random.default_rng(0)
nx, ny = 64, 64
wall_mask = np.zeros((nx, ny), np.int32)
wall_mask[10, 10] = 1

@pytest.fixture(scope="module")
def sim():

    return Simulation(nx, ny, "cpu")

# test 1: density and velocity match between taichi and numpy
def test_macroscopic_parity(sim):

    populations = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.f.from_numpy(populations)
    sim.macroscopic()
    taichi_density, taichi_velocity = sim.rho.to_numpy(), sim.u.to_numpy()

    # numpy oracle
    numpy_density, numpy_velocity = numpy_macroscopic(populations.astype(np.float64))

    assert np.allclose(taichi_density, numpy_density, atol=1e-4)
    assert np.allclose(taichi_velocity, numpy_velocity, atol=1e-4)

# test 2: collision matches between taichi and numpy
def test_collide_parity(sim):

    populations = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)
    relaxation_time = rng.uniform(0.5, 1.5)

    # taichi
    sim.f.from_numpy(populations)
    sim.collide(relaxation_time)
    taichi_populations = sim.f.to_numpy()

    # numpy oracle
    numpy_populations = numpy_collide(populations.astype(np.float64), relaxation_time)

    assert np.allclose(taichi_populations, numpy_populations, atol=1e-4)

# test 3: streaming matches between taichi and numpy
def test_stream_parity(sim):

    populations = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.f.from_numpy(populations)
    sim.stream()
    taichi_populations = sim.f.to_numpy()

    # numpy oracle
    numpy_populations = numpy_stream(populations.astype(np.float64))

    assert np.allclose(taichi_populations, numpy_populations, atol=1e-4)

# test 4: bounce-back matches between taichi and numpy
def test_bounce_back_parity(sim):

    populations = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.solid.from_numpy(wall_mask)  # load the wall into the Taichi field
    sim.f.from_numpy(populations)
    sim.bounce_back()
    taichi_populations = sim.f.to_numpy()

    # numpy oracle
    numpy_populations = numpy_bounce_back(populations.astype(np.float64), wall_mask.astype(bool))

    assert np.allclose(taichi_populations, numpy_populations, atol=1e-4)

# test 5: lid-driven cavity centerline matches between taichi and numpy
@pytest.mark.slow
def test_cavity_parity(sim):

    grid_size = 64
    lid_velocity = 0.1
    reynolds_number = 100
    viscosity = lid_velocity * grid_size / reynolds_number
    relaxation_time = 3 * viscosity + 0.5
    steps = 2000

    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((grid_size, grid_size), bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    lid = np.zeros((grid_size, grid_size), bool)
    lid[:, -1] = True
    stationary = solid & ~lid  # the 3 fixed walls

    # taichi
    sim.f.from_numpy(np.tile(lattice_weights[:, None, None], (1, grid_size, grid_size)).astype(np.float32))
    sim.solid.from_numpy(stationary.astype(np.int32))
    sim.lid.from_numpy(lid.astype(np.int32))
    for _ in range(steps):
        sim.collide(relaxation_time)
        sim.stream()
        sim.bounce_back()
        sim.moving_wall(lid_velocity)
    sim.macroscopic()
    taichi_centerline = sim.u.to_numpy()[0, grid_size // 2, :]

    # numpy oracle
    populations = numpy_initial(grid_size, grid_size)
    for _ in range(steps):
        populations = numpy_collide(populations, relaxation_time, solid)
        populations = numpy_stream(populations)
        populations = numpy_bounce_back(populations, stationary)
        populations = numpy_moving_wall(populations, lid, lid_velocity)
    numpy_density, numpy_velocity = numpy_macroscopic(populations)
    numpy_centerline = numpy_velocity[0, grid_size // 2, :]

    assert np.allclose(taichi_centerline, numpy_centerline, atol=1e-4)
