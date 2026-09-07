import numpy as np
from src.engine.simulation import Simulation
from src.lbm.moments import macroscopic as np_macroscopic
from src.lbm.collision import collide as np_collide
from src.lbm.stream import stream as np_stream
from src.lbm.boundary_conditions import bounce_back as np_bounce_back
from src.lbm.boundary_conditions import moving_wall as np_moving_wall
from src.lbm.advance import initial as np_initial
from src.engine.lattice import W
import pytest

rng = np.random.default_rng(0)
nx, ny = 64, 64
mask = np.zeros((nx, ny), np.int32)
mask[10, 10] = 1

# test 1: compare rho and u between taichi and numpy
def test_macroscopic_parity(sim):

    f = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.f.from_numpy(f)
    sim.macroscopic()
    rho_ti, u_ti = sim.rho.to_numpy(), sim.u.to_numpy()

    # numpy oracle
    rho_np, u_np = np_macroscopic(f.astype(np.float64))

    assert np.allclose(rho_ti, rho_np, atol=1e-4)
    assert np.allclose(u_ti, u_np, atol=1e-4)

# test 2: compare collision between taichi and numpy
def test_collide_parity(sim):

    f = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)
    tau = rng.uniform(0.5, 1.5)

    # taichi
    sim.f.from_numpy(f)
    sim.collide(tau)
    f_ti = sim.f.to_numpy()

    # numpy oracle
    f_np = np_collide(f.astype(np.float64), tau)
    
    assert np.allclose(f_ti, f_np, atol=1e-4)

# test 3: compare stream between taichi and numpy
def test_stream_parity(sim):

    f = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.f.from_numpy(f)
    sim.stream()
    f_ti = sim.f.to_numpy()

    # numpy oracle
    f_np = np_stream(f.astype(np.float64))

    assert np.allclose(f_ti, f_np, atol=1e-4)

# test 4: compare bounce_back between taichi and numpy
def test_bounce_back_parity(sim):

    f = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)

    # taichi
    sim.solid.from_numpy(mask)  #load the wall into the Taichi field
    sim.f.from_numpy(f)
    sim.bounce_back()
    f_ti = sim.f.to_numpy()

    # numpy oracle
    f_np = np_bounce_back(f.astype(np.float64), mask.astype(bool))

    assert np.allclose(f_ti, f_np, atol=1e-4)

@pytest.mark.slow
# test 5: compare cavity solving between taichi and numpy
def test_cavity_parity(sim):

    f = rng.uniform(0.5, 1.5, (9, nx, ny)).astype(np.float32)
    N = 64
    U = 0.1
    Re = 100
    nu = U * N / Re
    tau = 3 * nu + 0.5
    steps = 2000

    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((N, N), bool)
    solid[0,:] = solid[-1,:] = solid[:,0] = solid[:,-1] = True
    lid = np.zeros((N, N), bool); lid[:,-1] = True
    stationary = solid & ~lid          # the 3 fixed walls

    # taichi
    sim.f.from_numpy(np.tile(W[:, None, None], (1, N, N)).astype(np.float32))
    sim.solid.from_numpy(stationary.astype(np.int32))
    sim.lid.from_numpy(lid.astype(np.int32))
    for _ in range(steps):
        sim.collide(tau)
        sim.stream()
        sim.bounce_back()
        sim.moving_wall(U)
    sim.macroscopic()
    u_ti = sim.u.to_numpy()[0, nx//2, :]     # read the centerline out of the field

    # numpy oracle
    f_np = np_initial(N, N)
    for _ in range(steps):
        f_np = np_collide(f_np, tau)            # from src.lbm.collision
        f_np = np_stream(f_np)
        f_np = np_bounce_back(f_np, stationary)
        f_np = np_moving_wall(f_np, lid, U)
    rho_np, u_np = np_macroscopic(f_np)
    u_np = u_np[0, nx//2, :]

    assert np.allclose(u_ti, u_np, atol=1e-4)

@pytest.fixture(scope="module")
def sim():
    return Simulation(64, 64, "cpu")