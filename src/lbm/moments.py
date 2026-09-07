# density and velocity from the populations
from . import lattice as lt
import numpy as np

def macroscopic(f):
    """
    From population f (shape (9, nx, ny)) compute density and velocity.

    Returns rho (nx, ny) and u (2, nx, ny)
    """

    # density per cell
    rho = f.sum(axis=0)
    # directional momentum per cell
    mom_x = np.einsum("q,qxy->xy", lt.E[:,0], f)
    mom_y = np.einsum("q,qxy->xy", lt.E[:,1], f)
    # velocity from momentum divided by density
    u = np.stack([mom_x, mom_y]) / rho
    return rho, u