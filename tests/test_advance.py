# 2D timestep: at rest stays at rest, no NaNs, mass conserved
import numpy as np
from src.lbm.advance import initial, run
from src.lbm import lattice as lt
from src.lbm.moments import macroscopic

nx = 4
ny = 3
# test 1: rest stays at rest and stable
def test_rest_stays_at_rest():

    populations = initial(nx, ny)
    solid = np.zeros((nx, ny), dtype=bool)
    final_populations = run(populations, relaxation_time=1, solid=solid, steps=200)
    density, velocity = macroscopic(final_populations)

    assert not np.isnan(final_populations).any()
    assert np.allclose(velocity, 0)
    assert np.allclose(final_populations.sum(), populations.sum())

# test 2: one forced step (Guo) raises the total x-momentum by exactly g per cell and leaves y unchanged
def test_forced_step_adds_body_force():

    rng = np.random.default_rng(0)
    body_force_x = 1e-3
    populations = rng.uniform(0.9, 1.1, (9, nx, ny))
    solid = np.zeros((nx, ny), dtype=bool)
    stepped = run(populations, relaxation_time=0.8, solid=solid, steps=1, body_force_x=body_force_x)
    momentum = np.einsum("qc,qxy->c", lt.lattice_velocities, populations)
    stepped_momentum = np.einsum("qc,qxy->c", lt.lattice_velocities, stepped)

    assert np.isclose(stepped_momentum[0] - momentum[0], body_force_x * nx * ny, rtol=1e-10)
    assert np.isclose(stepped_momentum[1], momentum[1], atol=1e-12)
    assert np.isclose(stepped.sum(), populations.sum(), rtol=1e-12)
