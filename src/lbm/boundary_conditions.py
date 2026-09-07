# bounce-back wall and moving wall
import numpy as np
from . import lattice as lt

def bounce_back(f, solid):
    f_new = f.copy()
    for i in range(lt.Q):
        f_new[i, solid] = f[lt.OPP[i], solid]
    return f_new

def moving_wall(f, lid, u_wall):
    f_new = f.copy()
    for i in range(lt.Q):
        f_new[i, lid] = f[lt.OPP[i], lid] + 6* lt.W[i] * lt.E[i, 0] * u_wall
    return f_new