# 3D engine invariants: moments, collision, streaming, walls, inlet
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L3

rng = np.random.default_rng(0)
N = 16
mask = np.zeros((N, N, N), np.int32)
mask[10, 10, 10] = 1

@pytest.fixture(scope="module")
def sim():
    return Simulation3D(N, N, N, "cpu")

# test 1: macroscopic rho and u
def test_macroscopic3d(sim):
    sim.f.from_numpy(np.ones((L3.Q, N, N, N), np.float32))
    sim.macroscopic()
    
    assert np.allclose(sim.rho.to_numpy(), 19)
    assert np.allclose(sim.u.to_numpy(), 0)

# test 2: collide idempotent
def test_collide3d(sim):
    f = np.tile(L3.W[:,None,None,None], (1,N,N,N)).astype(np.float32)  # rest equilibrium
    sim.f.from_numpy(f); sim.collide(0.8)
    
    assert np.allclose(sim.f.to_numpy(), f, atol=1e-5)

# test 3: mass conservation
def test_conservation3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    tau = rng.uniform(0.5, 1.5)
    sim.f.from_numpy(f)
    sim.collide(tau)
    sim.stream()
    f_new = sim.f.to_numpy()

    assert np.allclose(f.sum(), f_new.sum())

# test 4: bounce back
def test_bounce3d(sim):
    c = (5, 5, 5)
    f = np.zeros((L3.Q, N, N, N), np.float32)
    f[1][c] = 5        # East at the cell
    f[3][c] = 2        # West at the same cell
    solid = np.zeros((N, N, N), np.int32); solid[c] = 1
    sim.solid.from_numpy(solid)
    sim.f.from_numpy(f)
    sim.bounce_back()
    g = sim.f.to_numpy()
    
    assert g[1][c] == 2   # East now holds West's old value
    assert g[3][c] == 5   # West now holds East's old value

# test 5: free slip walls
def test_slip3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    sim.f.from_numpy(f)
    sim.free_slip_y()
    sim.free_slip_y()
    f_new = sim.f.to_numpy()

    assert np.allclose(f, f_new, atol=1e-6)

# test 6: inlet
def test_inlet3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    U = rng.uniform(0.01, 0.1)
    sim.f.from_numpy(f)
    sim.inlet(U)
    sim.macroscopic()
    rho, u = sim.rho.to_numpy(), sim.u.to_numpy()
    f_new = sim.f.to_numpy()

    assert np.allclose(u[0, 0], U, atol=1e-5)
    assert np.allclose(u[1, 0], 0, atol=1e-5)
    assert np.allclose(u[2, 0], 0, atol=1e-5)
    assert np.allclose(rho[0], 1, atol=1e-5)

# test 7: TRT collision
def test_collideTRT3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    tau = rng.uniform(0.5, 1.5)
    sim.f.from_numpy(f)
    sim.collide_trt(tau)
    f_new = sim.f.to_numpy()

    assert np.allclose(f.sum(), f_new.sum(), atol=1e-4)

# test 8: LES collisions
def test_LES3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    tau = rng.uniform(0.5, 1.5)
    sim.f.from_numpy(f)
    sim.collide(tau)
    f_new = sim.f.to_numpy()

    sim.f.from_numpy(f)
    sim.collide_les(tau, 0.0)
    f_check = sim.f.to_numpy()

    sim.f.from_numpy(f)
    sim.collide_les(tau, 0.16)
    f_check2 = sim.f.to_numpy()

    assert np.allclose(f_check.sum(), f_new.sum(), atol=1e-4)
    assert np.allclose(f_check2.sum(), f.sum(), atol=1e-4)

# test 9: LES BGK match
def test_LESTinyCSMatchesBGK(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    tau = rng.uniform(0.5, 1.5)
    sim.f.from_numpy(f)
    sim.collide(tau)
    f_new = sim.f.to_numpy()

    sim.f.from_numpy(f)
    sim.collide_les(tau, 1e-9)
    f_check = sim.f.to_numpy()

    assert np.allclose(f_check, f_new, atol=1e-6)

# test 10: Free slip y reflects y
def test_freeYReflectY(sim):
    rho = np.ones((N, N, N))
    u = np.zeros((3, N, N, N))
    u[0] = 0.08
    u[1] = 0.05
    eu = np.einsum("qc,cxyz->qxyz", L3.E, u)
    usq = (u ** 2).sum(0)
    f = (L3.W[:, None, None, None] * rho * (1 + (3 * eu) + (4.5 * eu ** 2) - (1.5 * usq))).astype(np.float32)

    sim.f.from_numpy(f)
    sim.free_slip_y()
    sim.macroscopic()
    uu = sim.u.to_numpy()

    c = (N // 2, 0, N // 2) # a cell on the y=0 wall

    assert np.isclose(uu[0][c],  0.08, atol=1e-5)   # tangential kept
    assert np.isclose(uu[1][c], -0.05, atol=1e-5)   # normal flipped

# test 11: Free slip z reflects z
def test_freeZReflectZ(sim):
    rho = np.ones((N, N, N))
    u = np.zeros((3, N, N, N))
    u[0] = 0.08
    u[2] = 0.05
    eu = np.einsum("qc,cxyz->qxyz", L3.E, u)
    usq = (u ** 2).sum(0)
    f = (L3.W[:, None, None, None] * rho * (1 + (3 * eu) + (4.5 * eu ** 2) - (1.5 * usq))).astype(np.float32)

    sim.f.from_numpy(f)
    sim.free_slip_z()
    sim.macroscopic()
    uu = sim.u.to_numpy()

    c = (N // 2, N // 2, 0) # a cell on the y=0 wall
    
    assert np.isclose(uu[0][c],  0.08, atol=1e-5)   # tangential kept
    assert np.isclose(uu[2][c], -0.05, atol=1e-5)   # normal flipped