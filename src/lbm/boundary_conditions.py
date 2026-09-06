import numpy as np
from . import lattice as lt

def bounce_back(f, solid):
    f_new = f.copy()
    for i in range(lt.Q):
        f_new[i, solid] = f[lt.OPP[i], solid]
    return f_new