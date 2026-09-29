# absorbing-layer strength profiles for the relaxation layers (sigma fields)
import numpy as np

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