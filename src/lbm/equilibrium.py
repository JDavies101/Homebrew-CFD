# equilibrium distribution from density and velocity
import numpy as np
from . import lattice as lt

def equilibrium(density, velocity):
    """
    Compute the D2Q9 equilibrium populations from density and velocity.

    Returns the equilibrium populations (9, nx, ny).
    """

    # c_q . u in every direction for every cell
    velocity_dot_direction = np.einsum("qc,cxy->qxy", lt.lattice_velocities, velocity)
    # u . u for every cell
    velocity_squared = np.einsum("cxy,cxy->xy", velocity, velocity)
    # discretized maxwell-boltzmann distribution to 2nd order in velocity
    equilibrium_populations = lt.lattice_weights[:, None, None] * density * (1 + 3 * velocity_dot_direction + 4.5 * velocity_dot_direction * velocity_dot_direction - 1.5 * velocity_squared)

    return equilibrium_populations
