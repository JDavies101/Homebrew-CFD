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

# test 12: bounce back interpolation
def test_bounceBackInterp(sim):
    c = (5, 5, 5)
    # solid cell one step East of c along direction d (E-face dir)
    d = int(np.where((L3.E == [1, 0, 0]).all(1))[0][0])
    solid = np.zeros((N, N, N), np.int32); solid[6, 5, 5] = 1
    q = np.zeros((L3.Q, N, N, N), np.float32); q[d][c] = 0.5

    f = np.zeros((L3.Q, N, N, N), np.float32); f[d][c] = 0.3   # incoming pop toward wall
    sim.solid.from_numpy(solid); sim.q.from_numpy(q)
    sim.f.from_numpy(f); sim.fc.from_numpy(f)   # kernel reads fc as post-collision
    sim.bounce_back_interp()
    g = sim.f.to_numpy()

    ob = int(L3.OPP[d])
    assert np.isclose(g[ob][c], 0.3, atol=1e-6)   # q=0.5 -> reflected == incoming

# test 13: drag_interp sums c_i (f_in + f_out) over boundary links
def test_drag_interp(sim):
    c = (5, 5, 5)
    d = int(np.where((L3.E == [1, 0, 0]).all(1))[0][0])   # East, into solid
    ob = int(L3.OPP[d])
    solid = np.zeros((N, N, N), np.int32); solid[6, 5, 5] = 1
    q = np.zeros((L3.Q, N, N, N), np.float32); q[d][c] = 0.5

    fc = np.zeros((L3.Q, N, N, N), np.float32); fc[d][c] = 0.3   # incoming
    f  = np.zeros((L3.Q, N, N, N), np.float32); f[ob][c] = 0.2   # reflected
    sim.solid.from_numpy(solid); sim.q.from_numpy(q)
    sim.fc.from_numpy(fc); sim.f.from_numpy(f)
    sim.drag_interp()
    F = sim.force.to_numpy()

    # only link is East: F = c_i (f_in + f_out) = (1,0,0)*(0.3+0.2)
    assert np.isclose(F[0], 0.5, atol=1e-6)
    assert np.isclose(F[1], 0.0, atol=1e-6)
    assert np.isclose(F[2], 0.0, atol=1e-6)

# test 14: inlet_neem imposes u=(U,0,0) on the x=0 plane
def test_inlet_neem(sim):
    U = 0.1
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    f = np.tile(L3.W[:, None, None, None], (1, N, N, N)).astype(np.float32)  # rest state
    sim.f.from_numpy(f)
    sim.inlet_neem(U)
    sim.macroscopic()
    u = sim.u.to_numpy()
    assert np.allclose(u[0, 0], U, atol=1e-3)   # x-vel imposed
    assert np.allclose(u[1, 0], 0, atol=1e-3)
    assert np.allclose(u[2, 0], 0, atol=1e-3)

# test 15: inlet_neem_open drives open rows, leaves solid inlet columns alone
def test_inlet_neem_open(sim):
    U = 0.1
    solid = np.zeros((N, N, N), np.int32)
    solid[0:2, 0:4, :] = 1                       # a solid block spanning the inlet columns
    sim.solid.from_numpy(solid)
    f = np.tile(L3.W[:, None, None, None], (1, N, N, N)).astype(np.float32)
    sim.f.from_numpy(f)
    marker = sim.f.to_numpy()[:, 0, 2, 0].copy() # a solid inlet cell, pre-call
    sim.inlet_neem_open(U)
    g = sim.f.to_numpy()
    # open row untouched-by-solid gets the velocity
    sim.macroscopic(); u = sim.u.to_numpy()
    assert np.allclose(u[0, 0, 6, :], U, atol=1e-3)   # open row (j=6 > solid)
    # solid inlet column skipped
    assert np.allclose(g[:, 0, 2, 0], marker, atol=1e-6)

# test 16: outlet copies the second-to-last plane onto the last
def test_outlet(sim):
    f = rng.uniform(0.5, 1.5, (L3.Q, N, N, N)).astype(np.float32)
    sim.f.from_numpy(f)
    sim.outlet()
    g = sim.f.to_numpy()
    
    assert np.allclose(g[:, -1, :, :], g[:, -2, :, :])   # zero-gradient at exit
    assert np.allclose(g[:, :-1, :, :], f[:, :-1, :, :]) # interior untouched

# test 17: collide_full with TRT+LES+forcing together conserves mass, runs finite
def test_collide_full_combined(sim):
    f = rng.uniform(0.5, 1.5, (L3.Q, N, N, N)).astype(np.float32)
    sim.f.from_numpy(f)
    sim.collide_full(0.8, 0.1, 1e-6, 1)          # cs>0, gx>0, trt=1 all on
    g = sim.f.to_numpy()

    assert np.isfinite(g).all()
    assert np.isclose(g.sum(), f.sum(), rtol=1e-4)   # mass conserved