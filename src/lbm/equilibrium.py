import numpy as np
from . import lattice as lt

def equilibrium(rho, u):

    # dot product of e_i*u in every direction for every cell
    eu = np.einsum("qc,cxy->qxy", lt.E, u)
    # dot product of u*u for every cell
    usqr = np.einsum("cxy,cxy->xy", u, u)
    # discretized maxwell-boltzmann distribution with 2nd order velocity (D2Q9)
    f_eq = lt.W[:, None, None] * rho * (1 + (3 * eu) + (4.5 * eu**2) - (1.5 * usqr))
    return f_eq