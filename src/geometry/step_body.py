# backward-facing step solid mask (Armaly)
import numpy as np

def step(nx, ny, nz, x_step, S):
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :]  = 1           # bottom wall
    solid[:, -1, :] = 1           # top wall
    solid[:x_step, 1:S+1, :] = 1  # step block: upstream, sitting on the bottom
    return solid