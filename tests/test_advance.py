# 2D timestep: at rest stays at rest, no NaNs, mass conserved
import numpy as np
from src.lbm.advance import initial, run
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
