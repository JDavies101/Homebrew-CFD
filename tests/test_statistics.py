# block statistics: mean and standard error of block means
import numpy as np
from src.post.statistics import block_statistics

# test 1: known series gives the mean and standard error of its block means
def test_block_statistics_known_series():

    series = np.array([1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0])  # 3 blocks of 2, ragged tail (7) dropped
    mean, standard_error = block_statistics(series, 3)

    assert np.isclose(mean, 3.0)
    assert np.isclose(standard_error, 2.0 / np.sqrt(3.0))
