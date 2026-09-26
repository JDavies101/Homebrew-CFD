# absorbing-layer (sponge) viscosity field: damps reflections at open and free-slip boundaries
import numpy as np


def sponge(nx, ny, nz, width, nu_max):
    # quadratic ramp: nu_max on the boundary plane, 0 at `width` cells in
    # applied at inlet (x=0), outlet (x=nx-1), and both side walls (z=0, z=nz-1)
    x = np.arange(nx)[:, None, None]
    z = np.arange(nz)[None, None, :]
    d = np.minimum(np.minimum(x, nx - 1 - x), np.minimum(z, nz - 1 - z)).astype(np.float32)
    r = np.clip(1.0 - d / width, 0.0, 1.0)
    return np.broadcast_to(nu_max * r ** 2, (nx, ny, nz)).astype(np.float32).copy()

def relax_profile(nx, nz, width_x, width_z, sigma_max):
    # 2D relaxation strength over (x, z): quadratic ramp in from the inlet/outlet planes and the z walls
    x = np.arange(nx)[:, None]
    z = np.arange(nz)[None, :]
    dx = np.minimum(x, nx - 1 - x).astype(np.float32)
    dz = np.minimum(z, nz - 1 - z).astype(np.float32)
    if width_x > 0:
        rx = np.clip(1.0 - dx / width_x, 0.0, 1.0)
    else:
        rx = np.zeros_like(dx)
    if width_z > 0:
        rz = np.clip(1.0 - dz / width_z, 0.0, 1.0)
    else:
        rz = np.zeros_like(dz)
    return (sigma_max * np.maximum(rx, rz) ** 2).astype(np.float32)