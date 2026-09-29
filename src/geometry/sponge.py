# absorbing-layer strength profiles for the relaxation layers (sigma fields)
import numpy as np

def relax_profile(nx, nz, width_x, width_z, sigma_max):
    """
    Relaxation strength over (x, z): quadratic ramp in from the inlet/outlet planes and the z walls.

    Returns a float32 array (nx, nz).
    """

    x = np.arange(nx)[:, None]
    z = np.arange(nz)[None, :]
    distance_x = np.minimum(x, nx - 1 - x).astype(np.float32)
    distance_z = np.minimum(z, nz - 1 - z).astype(np.float32)

    if width_x > 0:
        ramp_x = np.clip(1.0 - distance_x / width_x, 0.0, 1.0)
    else:
        ramp_x = np.zeros_like(distance_x)
    if width_z > 0:
        ramp_z = np.clip(1.0 - distance_z / width_z, 0.0, 1.0)
    else:
        ramp_z = np.zeros_like(distance_z)

    return (sigma_max * np.maximum(ramp_x, ramp_z) ** 2).astype(np.float32)
