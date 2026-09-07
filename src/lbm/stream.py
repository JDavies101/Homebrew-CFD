# streaming: shift each population one cell along its own direction
import numpy as np
from . import lattice as lt

def stream(f):

    f_new = np.empty_like(f)

    for i in range(lt.Q):
        f_new[i] = np.roll(f[i], shift=lt.E[i], axis=(0,1))

    return f_new