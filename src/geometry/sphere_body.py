# sphere solid mask
import numpy as np

def sphere(nx, ny, nz, center_x, center_y, center_z, radius):
    """
    Build a sphere solid mask.

    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    solid = np.zeros((nx, ny, nz), np.int32)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ball = (X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2 < radius ** 2
    solid[ball] = 1

    return solid
