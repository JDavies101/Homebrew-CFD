# statistics of sampled series: block-averaged mean and standard error
import numpy as np

def block_statistics(series, block_count):
    """
    Block-averaged mean and standard error of a correlated series (ragged tail dropped).

    Returns (mean, standard_error).
    """

    usable_length = len(series) // block_count * block_count
    block_means = np.asarray(series[:usable_length]).reshape(block_count, -1).mean(axis=1)

    return block_means.mean(), block_means.std(ddof=1) / np.sqrt(block_count)
