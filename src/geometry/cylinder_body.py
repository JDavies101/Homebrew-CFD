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
