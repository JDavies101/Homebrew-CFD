# cylinder solid mask: disc in x-y, spanning z
import numpy as np

def cylinder(nx, ny, nz, center_x, center_y, radius):
    """
    Build a z-spanning cylinder solid mask.

    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    solid = np.zeros((nx, ny, nz), np.int32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    disc = (X - center_x) ** 2 + (Y - center_y) ** 2 < radius ** 2  # (nx, ny) boolean
    solid[disc] = 1

    return solid

def spin_wall_velocity(nx, ny, nz, center_x, center_y, radius, angular_velocity):
    """
    Rigid rotation about the z axis on a band covering both sides of the surface (Bouzidi reads u_w on either side).

    Returns float32 (3, nx, ny, nz): u_w = omega x r inside the band, 0 elsewhere.
    """

    X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
    offset_x = X - center_x
    offset_y = Y - center_y
    band = np.sqrt(offset_x * offset_x + offset_y * offset_y) < radius + 2.0
    wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
    wall_velocity[0][band] = (-angular_velocity * offset_y[band])[:, None]
    wall_velocity[1][band] = (angular_velocity * offset_x[band])[:, None]

    return wall_velocity