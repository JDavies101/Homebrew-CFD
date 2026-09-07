# 2D timestep: at rest stays at rest, no NaNs, mass conserved
import numpy as np
from src.lbm.advance import initial, run
from src.lbm.moments import macroscopic

nx = 4
ny = 3
# test 1: rest stays rest and stable
def test_stability():

    f = initial(nx, ny)
    solid = np.zeros((nx, ny), dtype=bool)
    f_new = run(f, tau=1, solid=solid, steps=200)

    rho, u = macroscopic(f_new)

    assert not np.isnan(f_new).any()
    assert np.allclose(u, 0)
    assert np.allclose(f_new.sum(), f.sum())
