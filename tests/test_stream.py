# streaming: mass conserved, blob moves one cell, rest stays put
import numpy as np
from src.lbm.stream import stream

rng = np.random.default_rng(0)
nx = 4
ny = 3
# test 1: global mass conservation
def test_mass_conservation():

    populations = rng.uniform(0.5, 1.5, (9, nx, ny))
    streamed_populations = stream(populations)

    assert np.allclose(populations.sum(), streamed_populations.sum())

# test 2: a north-east blob moves one cell diagonally
def test_blob_moves_one_cell():

    populations = np.zeros((9, nx, ny))
    populations[5, 1, 1] = 1
    streamed_populations = stream(populations)

    assert np.allclose(streamed_populations[5, 2, 2], 1)

# test 3: rest population does not move
def test_rest_population_stays():

    populations = np.zeros((9, nx, ny))
    populations[0] = rng.uniform(0.5, 1.5, (nx, ny))
    streamed_populations = stream(populations)

    assert np.allclose(streamed_populations[0], populations[0])
