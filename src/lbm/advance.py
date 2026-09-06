import numpy as np
from .stream import stream
from .collision import collide
from .boundary_conditions import bounce_back
from .equilibrium import equilibrium
from . import lattice as lt
from .moments import macroscopic

def guo_source(u, F, tau):

    eu = np.einsum("qc,cxy->qxy", lt.E, u)
    eF = np.einsum("qc,cxy->qxy", lt.E, F)
    uF = np.einsum("cxy,cxy->xy", u, F)

    prefac = 1 - 1/ (2 * tau)
    S = prefac * lt.W[:, None, None] * (3 * (eF- uF) + 9* eu * eF)

    return S

def body_force(g):

    F = (lt.W * lt.E[:,0] * g * 3)[:, None, None]

    return F

def collide_forced(f, tau, F):

    rho, u_raw = macroscopic(f)
    u = u_raw + F / (2 * rho)
    f_eq = equilibrium(rho, u)
    S = guo_source(u, F, tau)

    return f - (1 / tau) * (f - f_eq) + S

def step(f, tau, solid, g):

    nx, ny = f.shape[1], f.shape[2]
    F = np.zeros((2, nx, ny))
    F[0] = g
    f_coll = collide_forced(f, tau, F)
    f_str = stream(f_coll)    
    f_bc = bounce_back(f_str, solid)

    return f_bc

def run(f, tau, solid, steps, g):

    for i in range(steps):
        f = step(f, tau, solid, g)

    return f

def initial(nx, ny):

    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))
    f = equilibrium(rho, u)

    return f