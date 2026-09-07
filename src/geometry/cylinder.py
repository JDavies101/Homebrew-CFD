import numpy as np

def cylinder(nx, ny, nz, cx, cy, r):
    solid = np.zeros((nx, ny, nz), np.int32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing = "ij")
    disc = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2 # (nx, ny) boolean
    solid[disc] = 1
    return solid