# streaming: shift each population one cell along its own direction
import numpy as np
from . import lattice as lt

def stream(populations):
    """
    Shift each population one cell along its lattice velocity (periodic).

    Returns the streamed populations.
    """

    streamed_populations = np.empty_like(populations)

    for q in range(lt.direction_count):
        streamed_populations[q] = np.roll(populations[q], shift=lt.lattice_velocities[q], axis=(0, 1))

    return streamed_populations
