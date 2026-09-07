# sphere solid mask
import numpy as np

def sphere(nx, ny, nz, cx, cy, cz, D):
        solid = np.zeros((nx, ny, nz), np.int32)
        X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing = "ij")
        ball = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 < (D / 2) ** 2
        solid[ball] = 1
        return solid