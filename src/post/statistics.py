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

def dominant_frequency(series, sample_spacing):
    """
    Dominant frequency of a periodic series by two independent estimates: Hann-windowed FFT peak refined by
    parabolic interpolation of the log magnitude, and the mean period between upward zero crossings.

    Returns (spectral_frequency, zero_crossing_frequency) in cycles per unit of sample_spacing.
    """

    signal = np.asarray(series, np.float64)
    signal = signal - signal.mean()
    sample_count = len(signal)

    # spectral estimate: peak bin (skip the zero-frequency bin), then a parabola through log |X| at k-1, k, k+1
    magnitude = np.abs(np.fft.rfft(signal * np.hanning(sample_count)))
    peak = 1 + int(np.argmax(magnitude[1:-1]))
    below = np.log(magnitude[peak - 1])
    center = np.log(magnitude[peak])
    above = np.log(magnitude[peak + 1])
    offset = 0.5 * (below - above) / (below - 2.0 * center + above)
    spectral_frequency = (peak + offset) / (sample_count * sample_spacing)

    # zero-crossing estimate: linearly interpolated times of upward crossings
    upward = np.nonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0))[0]
    crossing_times = (upward + signal[upward] / (signal[upward] - signal[upward + 1])) * sample_spacing
    zero_crossing_frequency = 1.0 / np.mean(np.diff(crossing_times))

    return spectral_frequency, zero_crossing_frequency