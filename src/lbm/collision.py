# BGK collision: relax the populations toward equilibrium
import numpy as np
from .moments import macroscopic
from .equilibrium import equilibrium

def collide(f, tau, solid=None):
    rho, u = macroscopic(f)
    f_eq = equilibrium(rho, u)
    f_coll = f - 1 / tau * (f - f_eq)
    if solid is not None:
        f_coll = np.where(solid[None], f, f_coll) # wall nodes keep their bounced populations
    return f_coll