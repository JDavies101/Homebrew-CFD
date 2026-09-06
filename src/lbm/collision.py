import numpy as np
from .moments import macroscopic
from .equilibrium import equilibrium

def collide(f, tau):
    rho, u = macroscopic(f)
    f_eq = equilibrium(rho, u)
    f_coll = f - 1 / tau * (f - f_eq)
    return f_coll