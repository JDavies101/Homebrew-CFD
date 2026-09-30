# block statistics: mean and standard error of block means
import numpy as np
from src.post.statistics import block_statistics, dominant_frequency

# test 1: known series gives the mean and standard error of its block means
def test_block_statistics_known_series():

    series = np.array([1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0])  # 3 blocks of 2, ragged tail (7) dropped
    mean, standard_error = block_statistics(series, 3)

    assert np.isclose(mean, 3.0)
    assert np.isclose(standard_error, 2.0 / np.sqrt(3.0))

# test 2: dominant_frequency recovers a sine frequency that falls between FFT bins, by both estimates
def test_dominant_frequency_between_bins():

    sample_spacing = 20.0
    frequency = 0.0123 / sample_spacing  # cycles per step, deliberately off the FFT bin grid
    times = np.arange(1500) * sample_spacing
    series = 0.3 + np.sin(2.0 * np.pi * frequency * times + 0.4)  # offset and phase: the mean is removed internally
    spectral_frequency, zero_crossing_frequency = dominant_frequency(series, sample_spacing)
    bin_width = 1.0 / (1500 * sample_spacing)

    assert abs(spectral_frequency - frequency) < 0.01 * bin_width  # 100x finer than the raw FFT bin
    assert abs(zero_crossing_frequency / frequency - 1.0) < 1e-3