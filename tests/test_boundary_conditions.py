# bounce-back: swap at the wall, fluid untouched, involution
import numpy as np
from src.lbm.boundary_conditions import bounce_back

rng = np.random.default_rng(0)
nx = 4
ny = 3
solid = np.zeros((nx, ny), dtype=bool)
solid[1, 1] = True
# test 1: opposite populations swap on a solid node
def test_swap_on_solid_node():

    populations = rng.uniform(0.5, 1.5, (9, nx, ny))
    populations[1, 1, 1] = 5
    populations[3, 1, 1] = 2
    bounced_populations = bounce_back(populations, solid)

    assert bounced_populations[1, 1, 1] == 2
    assert bounced_populations[3, 1, 1] == 5

# test 2: every fluid node untouched
def test_fluid_nodes_untouched():

    populations = rng.uniform(0.5, 1.5, (9, nx, ny))
    bounced_populations = bounce_back(populations, solid)

    assert np.array_equal(bounced_populations[:, ~solid], populations[:, ~solid])

# test 3: bouncing twice returns the original populations
def test_bounce_back_is_involution():

    populations = rng.uniform(0.5, 1.5, (9, nx, ny))
    bounced_populations = bounce_back(populations, solid)
    bounced_twice_populations = bounce_back(bounced_populations, solid)

    assert np.allclose(populations, bounced_twice_populations)
