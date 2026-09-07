# bounce-back: swap at the wall, fluid untouched, involution
import numpy as np
from src.lbm.boundary_conditions import bounce_back
from src.lbm import lattice as lt

rng = np.random.default_rng(0)
nx = 4
ny = 3
solid = np.zeros((nx, ny), dtype=bool)
solid[1, 1] = True
# test 1: check swap
def test_swap():
    f = rng.uniform(0.5, 1.5, (9, nx, ny))
    f[1, 1, 1] = 5
    f[3, 1, 1] = 2
    f_new = bounce_back(f, solid)
    assert f_new[1, 1, 1] == 2
    assert f_new[3, 1, 1] == 5

# test 2: fluid cells untouched
def test_untouched():
    f = rng.uniform(0.5, 1.5, (9, nx, ny))
    f[2, 0, 0] = 4
    f_new = bounce_back(f, solid)
    assert f_new[2, 0, 0] == 4

# test 3: involution
def test_involution():
    f = rng.uniform(0.5, 1.5, (9, nx, ny))
    f_new = bounce_back(f, solid)
    f_check = bounce_back(f_new, solid)
    assert np.allclose(f, f_check)