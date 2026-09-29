# BGK collision: relax the populations toward equilibrium
import numpy as np
from .moments import macroscopic
from .equilibrium import equilibrium

def collide(populations, relaxation_time, solid=None):
    """
    Relax the populations toward equilibrium with the BGK operator.

    Returns the post-collision populations.
    """

    density, velocity = macroscopic(populations)
    equilibrium_populations = equilibrium(density, velocity)
    collided_populations = populations - 1 / relaxation_time * (populations - equilibrium_populations)

    if solid is not None:
        collided_populations = np.where(solid[None], populations, collided_populations)  # wall nodes keep their bounced populations

    return collided_populations
