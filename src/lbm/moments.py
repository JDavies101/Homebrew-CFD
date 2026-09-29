# density and velocity from the populations
import numpy as np
from . import lattice as lt

def macroscopic(populations):
    """
    Compute density and velocity from the populations (shape (9, nx, ny)).

    Returns density (nx, ny) and velocity (2, nx, ny).
    """

    # density per cell
    density = populations.sum(axis=0)
    # directional momentum per cell
    momentum_x = np.einsum("q,qxy->xy", lt.lattice_velocities[:, 0], populations)
    momentum_y = np.einsum("q,qxy->xy", lt.lattice_velocities[:, 1], populations)
    # velocity from momentum divided by density
    velocity = np.stack([momentum_x, momentum_y]) / density

    return density, velocity
