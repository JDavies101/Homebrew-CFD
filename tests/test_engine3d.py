# 3D engine invariants: moments, collision, streaming, walls, inlet
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L3
from src.turbulence.wall_function import friction_velocity
from src.engine import runtime 
from src.geometry.sponge import sponge

rng = np.random.default_rng(0)
N = 16
mask = np.zeros((N, N, N), np.int32)
mask[10, 10, 10] = 1

@pytest.fixture(scope="module")
def sim():
    return Simulation3D(N, N, N, "cpu", interp=True)

# per-cell mass and momentum, so conservation errors can't cancel across the field

def _moments(f):
    return f.sum(axis=0), np.einsum("qc, qxyz -> cxyz", L3.E, f)

# load f on a clean slate: no walls, no wall model, so every cell collides
def _load(sim, f):
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.lid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.nut_wall.from_numpy(np.zeros((N, N, N), np.float32))
    sim.f.from_numpy(f)

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

# test 3: BGK conserves mass and momentum per cell; periodic stream is an exact shift
def test_conservation3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    _load(sim, f)
    sim.collide(rng.uniform(0.5, 1.5))
    g = sim.f.to_numpy()
    rho0, m0 = _moments(f)
    rho1, m1 = _moments(g)
    assert np.allclose(rho1, rho0, atol=1e-4)
    assert np.allclose(m1, m0, atol=1e-4)

    sim.stream()
    h = sim.f.to_numpy()
    for q in range(L3.Q):  # f_new[q](x) = f[q](x - e_q)
        assert np.array_equal(h[q], np.roll(g[q], shift=tuple(L3.E[q]), axis=(0, 1, 2)))

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

# test 7: TRT conserves mass and momentum per cell
def test_collideTRT3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    _load(sim, f)
    sim.collide_trt(rng.uniform(0.5, 1.5))
    rho0, m0 = _moments(f)
    rho1, m1 = _moments(sim.f.to_numpy())
    assert np.allclose(rho1, rho0, atol=1e-4)
    assert np.allclose(m1, m0, atol=1e-4)

# test 8: LES is live (changes the result vs plain BGK) and still conserves per cell
def test_LES3d(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    tau = rng.uniform(0.6, 1.5)
    _load(sim, f); sim.collide(tau);           bgk = sim.f.to_numpy()
    _load(sim, f); sim.collide_les(tau, 0.16); les = sim.f.to_numpy()

    assert not np.allclose(les, bgk, atol=1e-6)            # the LES branch actually did something
    rho0, m0 = _moments(f); rho1, m1 = _moments(les)
    assert np.allclose(rho1, rho0, atol=1e-4)
    assert np.allclose(m1, m0, atol=1e-4)

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

# test 17: TRT + LES + forcing together. mass exact per cell, and each cell's x-momentum
# goes up by exactly gx per step (Guo). the old antisymmetric-rate bug breaks this
def test_collide_full_combined(sim):
    gx = 1e-2
    f = rng.uniform(0.5, 1.5, (L3.Q, N, N, N)).astype(np.float32)
    _load(sim, f)
    sim.collide_full(0.8, 0.1, gx, 1)
    g = sim.f.to_numpy()
    rho0, m0 = _moments(f); rho1, m1 = _moments(g)

    assert np.isfinite(g).all()
    assert np.allclose(rho1, rho0, atol=1e-4)
    assert np.allclose(m1[0] - m0[0], gx, atol=1e-4)
    assert np.allclose(m1[1:], m0[1:], atol=1e-4)

# test 18: init_equilibrium sets the right moments (rho and u recovered)
def test_init_equilibrium_moments(sim):
    ux = np.full((N, N, N), 0.05, np.float32)
    uy = np.full((N, N, N), -0.02, np.float32)
    uz = np.full((N, N, N), 0.01, np.float32)
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.init_equilibrium(ux, uy, uz)
    sim.macroscopic()
    rho, u = sim.rho.to_numpy(), sim.u.to_numpy()

    assert np.allclose(rho, 1, atol=1e-5)
    assert np.allclose(u[0], 0.05, atol=1e-5)
    assert np.allclose(u[1], -0.02, atol=1e-5)
    assert np.allclose(u[2], 0.01, atol=1e-5)

# test 19: an equilibrium state is a collision fixed point (no force) -- this fails if
# init_equilibrium's feq and collide_full's feq ever drift apart
def test_equilibrium_is_collision_fixed_point(sim):
    ux = (0.05 * (2 * rng.random((N, N, N)) - 1)).astype(np.float32)
    uy = (0.05 * (2 * rng.random((N, N, N)) - 1)).astype(np.float32)
    uz = (0.05 * (2 * rng.random((N, N, N)) - 1)).astype(np.float32)
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.init_equilibrium(ux, uy, uz)
    f0 = sim.f.to_numpy()
    sim.collide(0.8)                                  # BGK, no force -> feq is the fixed point

    assert np.allclose(sim.f.to_numpy(), f0, atol=1e-5)

# test 19b: collision skips wall nodes, so bounce-back hands populations back unchanged
@pytest.mark.parametrize("kernel", ["full", "reg"])
def test_collide_skip_walls(sim, kernel):
    f = rng.uniform(0.5, 1.5, (L3.Q, N, N, N)).astype(np.float32)
    solid = np.zeros((N, N, N), np.int32)
    solid[8, 8, 8] = 1
    sim.solid.from_numpy(solid)
    sim.lid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.nut_wall.from_numpy(np.zeros((N, N, N), np.float32))
    sim.f.from_numpy(f)
    if kernel == "full":
        sim.collide_full(0.8, 0.1, 1e-4, 1)
    else:
        sim.collide_reg(0.8, 0.1, 0.0)
    g = sim.f.to_numpy()

    assert np.array_equal(g[:, 8, 8, 8], f[:, 8, 8, 8])
    assert not np.allclose(g[:, 0, 0, 0], f[:, 0, 0, 0])

# test 20: nut_wall raises the local tau by 3*nu_t, at that node only, independent of LES.
# Run at cs=0 so it fails if the augmentation is gated inside the LES branch.
def test_nut_wall_augments_tau(sim):
    tau0, nut = 0.8, 0.03
    c = (8, 8, 8)
    f0 = np.tile(L3.W[:, None, None, None], (1, N, N, N)).astype(np.float32)
    f0[5][c] += 0.05                                   # a non-equilibrium perturbation at one node
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))

    sim.nut_wall.from_numpy(np.zeros((N, N, N), np.float32))
    sim.f.from_numpy(f0); sim.collide_full(tau0, 0.0, 0.0, 0)
    base = sim.f.to_numpy()

    sim.f.from_numpy(f0); sim.collide_full(tau0 + 3 * nut, 0.0, 0.0, 0)   # raise tau0 by hand
    manual = sim.f.to_numpy()

    w = np.zeros((N, N, N), np.float32); w[c] = nut
    sim.nut_wall.from_numpy(w)
    sim.f.from_numpy(f0); sim.collide_full(tau0, 0.0, 0.0, 0)             # via the wall field
    wm = sim.f.to_numpy()

    assert np.allclose(wm[:, 8, 8, 8], manual[:, 8, 8, 8], atol=1e-6)     # nut_wall == raising tau0
    assert np.allclose(wm[:, 0, 0, 0], base[:, 0, 0, 0], atol=1e-6)       # untouched elsewhere

# test 21: wall_model sets nut_wall from the log-law u_tau at wall-adjacent nodes, gated on y+>30
def test_wall_model(sim):
    nu, y1 = 0.01, 10.0
    solid = np.zeros((N, N, N), np.int32); solid[:, 0, :] = 1             # bottom wall row
    sim.solid.from_numpy(solid)
    u = np.zeros((3, N, N, N), np.float32)
    u[0, 4, 1, 4] = 0.737                                                 # y+ ~ 50 (>30): engages
    u[0, 6, 1, 6] = 0.02                                                  # y+ < 30: stays off
    sim.u.from_numpy(u)
    sim.nut_wall.from_numpy(np.ones((N, N, N), np.float32))               # nonzero -> check it resets
    sim.wall_model(nu, y1)
    nw = sim.nut_wall.to_numpy()

    u_tau = friction_velocity(0.737, y1, nu)
    expect = u_tau ** 2 * y1 / 0.737 - nu
    assert np.isclose(nw[4, 1, 4], expect, rtol=1e-4)                     # engaged node
    assert nw[6, 1, 6] == 0.0                                             # y+<30 -> off
    assert nw[8, 8, 8] == 0.0                                             # interior (not adjacent) -> reset

# test 22: regularized + LES conserves mass and momentum per cell
def test_regularization(sim):
    f = rng.uniform(0.5, 1.5, (19, N, N, N)).astype(np.float32)
    _load(sim, f)
    sim.collide_reg(rng.uniform(0.6, 1.5), 0.1, 0.0)
    rho0, m0 = _moments(f); rho1, m1 = _moments(sim.f.to_numpy())
    assert np.allclose(rho1, rho0, atol=1e-4)
    assert np.allclose(m1, m0, atol=1e-4)

# test 23: regularized collision has the CORRECT shear viscosity.
# A resolved, unforced shear wave u_x = U0 sin(k y) decays as exp(-nu k^2 t) with
# nu = c_s^2 (tau-1/2). Isolates viscosity: no forcing (avoids the Guo coupling), fully
# periodic (no walls), well-resolved (no ghost content) -> reg must match analytic and BGK.
def test_reg_shear_viscosity():
    nx, ny, nz = 4, 64, 4
    tau = 0.6
    nu_analytic = (tau - 0.5) / 3.0
    k = 2 * np.pi / ny
    U0 = 0.01
    steps = 1500
    yy = np.arange(ny)

    def measured_nu(step_fn):
        sim = Simulation3D(nx, ny, nz, "cpu")
        sim.solid.from_numpy(np.zeros((nx, ny, nz), np.int32))     # no walls: fully periodic
        ux = np.broadcast_to((U0 * np.sin(k * yy))[None, :, None],
                             (nx, ny, nz)).astype(np.float32).copy()
        zero = np.zeros((nx, ny, nz), np.float32)
        sim.init_equilibrium(ux, zero.copy(), zero.copy())

        def amp():
            sim.macroscopic()
            prof = sim.u.to_numpy()[0].mean(axis=(0, 2))          # u_x(y)
            return 2.0 / ny * np.sum(prof * np.sin(k * yy))       # sin-mode amplitude

        a0 = amp()
        for _ in range(steps):
            step_fn(sim)
            sim.stream()
        aT = amp()
        return -np.log(aT / a0) / (k * k * steps)

    nu_reg = measured_nu(lambda s: s.collide_reg(tau, 0.0, 0.0))
    nu_bgk = measured_nu(lambda s: s.collide(tau))

    assert abs(nu_reg / nu_analytic - 1) < 0.03    # regularized viscosity is correct
    assert abs(nu_reg / nu_bgk - 1) < 0.02         # and matches BGK on a resolved wave

# test 24: wall_model detects a non-y wall normal and uses the wall-PARALLEL speed.
# An x-normal wall: the model must drive off sqrt(uy^2+uz^2) and ignore the u_x normal
# component -- otherwise the generalized normal detection is wrong.
def test_wall_model_xwall():
    sim = Simulation3D(N, N, N, "cpu")                           # own sim: needs fresh zeroed fields
    nu, y1 = 0.01, 10.0
    solid = np.zeros((N, N, N), np.int32); solid[0, :, :] = 1     # wall at i=0 -> +x normal
    sim.solid.from_numpy(solid)
    u = np.zeros((3, N, N, N), np.float32)
    u[0, 1, 5, 5] = 0.3                                           # normal component -> must be excluded
    u[1, 1, 5, 5] = 0.737                                         # tangential (y): drives the model
    sim.u.from_numpy(u)
    sim.nut_wall.from_numpy(np.zeros((N, N, N), np.float32))
    sim.wall_model(nu, y1)
    nw = sim.nut_wall.to_numpy()

    u_tau = friction_velocity(0.737, y1, nu)                     # from the tangential speed only
    expect = u_tau ** 2 * y1 / 0.737 - nu
    assert np.isclose(nw[1, 5, 5], expect, rtol=1e-4)            # normal component correctly excluded

# test 25: second instance keeps the fixture
def test_second_instance_keeps_fixture(sim):
    sim.f.fill(1.0)
    s2 = Simulation3D(4, 4, 4, "cpu")
    
    assert not hasattr(s2, "q")
    assert not hasattr(s2, "fc")
    assert np.all(sim.f.to_numpy() == 1.0)

# test 26: runtime rejects switch
def test_runtime_rejects_switch(sim):
    with pytest.raises(RuntimeError):
        runtime.init("cuda")

# test 27: fast wall model matches slow wall model
def test_wall_model_matches(sim):
    solid = np.zeros((N, N, N), np.int32); solid[:, 0, :] = 1
    sim.solid.from_numpy(solid)
    f = rng.uniform(0.9, 1.1, (L3.Q, N, N, N)).astype(np.float32)
    sim.f.from_numpy(f); sim.nut_wall.from_numpy(np.zeros((N,N,N), np.float32))
    sim.macroscopic(); sim.wall_model(0.01, 10.0)
    slow = sim.nut_wall.to_numpy().copy()
    sim.nut_wall.from_numpy(np.zeros((N,N,N), np.float32))
    sim.build_wall_list(); sim.wall_model_fast(0.01, 10.0)
    assert np.allclose(sim.nut_wall.to_numpy(), slow, atol=1e-6)

# test 28: drag_body matches manual
def test_drag_body_matches_manual(sim):
    solid = np.zeros((N, N, N), np.int32)
    body = np.zeros((N, N, N), np.int32)
    solid[8, 8, 8] = 1; body[8, 8, 8] = 1                 # one body voxel = one solid voxel
    sim.solid.from_numpy(solid); sim.body.from_numpy(body)
    sim.f.from_numpy(rng.uniform(0.9, 1.1, (L3.Q, N, N, N)).astype(np.float32))
    sim.drag(); a = sim.force.to_numpy().copy()
    sim.drag_body(); b = sim.force.to_numpy()
    assert np.allclose(a, b)                               # body==solid here -> identical force

# test 29: WALE matches numpy
def test_les_wale_matches_numpy(sim):
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.lid.from_numpy(np.zeros((N, N, N), np.int32))
    u = rng.uniform(-0.05, 0.05, (3, N, N, N)).astype(np.float32)
    sim.u.from_numpy(u)
    cw = 0.5
    delta = 1.0
    eps = 1e-12
    sim.les_wale(cw)
    got = sim.nut_les.to_numpy()

    # numpy reference: same WALE formula on the same u
    u64 = u.astype(np.float64)
    g = np.zeros((3, 3, N, N, N))
    for a in range(3):
        for b in range(3):
            g[a, b] = np.gradient(u64[a], axis=b)          # d u_a / d x_b, unit spacing
    S = 0.5 * (g + g.transpose(1, 0, 2, 3, 4))             # symmetric strain
    gsq = np.einsum("ac...,cb...->ab...", g, g)            # g . g
    tr = gsq[0, 0] + gsq[1, 1] + gsq[2, 2]
    Sd = 0.5 * (gsq + gsq.transpose(1, 0, 2, 3, 4))
    for a in range(3):
        Sd[a, a] -= tr / 3.0                               # traceless
    SS = np.einsum("ab...,ab...->...", S, S)
    SdSd = np.einsum("ab...,ab...->...", Sd, Sd)
    nut_ref = (cw * delta) ** 2 * SdSd ** 1.5 / (SS ** 2.5 + SdSd ** 1.25 + eps)

    assert np.allclose(got[1:-1, 1:-1, 1:-1], nut_ref[1:-1, 1:-1, 1:-1], atol=1e-6)

# test 30: WALE zero when no flow
def test_les_wale_zero_on_quiescent(sim):
    sim.solid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.lid.from_numpy(np.zeros((N, N, N), np.int32))
    sim.u.from_numpy(np.zeros((3, N, N, N), np.float32))
    sim.les_wale(0.5)

    assert np.allclose(sim.nut_les.to_numpy(), 0.0)

# test 31: sponge is zero interior and max on boundary planes
def test_sponge_profile():
    
    s = sponge(40, 10, 30, width=8, nu_max=0.02)
    assert np.isclose(s[0, 5, 15], 0.02)
    assert np.isclose(s[20, 5, 15], 0.0)
    assert np.isclose(s[20, 5, 0], 0.02)