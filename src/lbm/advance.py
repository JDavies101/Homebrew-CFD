import numpy as np
from .stream import stream
from .collision import collide
from .boundary_conditions import bounce_back
from .equilibrium import equilibrium

def step(f, tau, solid):

    f_coll = collide(f, tau)
    f_str = stream(f_coll)    
    f_bc = bounce_back(f_str, solid)

    return f_bc

def run(f, tau, solid, steps):

    for i in range(steps):
        f = step(f, tau, solid)

    return f

def initial(nx, ny):

    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))
    f = equilibrium(rho, u)

    return f