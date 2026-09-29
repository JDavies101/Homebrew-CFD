# backward-facing step solid mask (Armaly)
import numpy as np

def step(nx, ny, nz, x_step, step_height):
    """
    Build the channel walls plus the upstream step block (step_height 0 gives the two walls only).

    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1  # bottom wall
    solid[:, -1, :] = 1  # top wall
    solid[:x_step, 1 : step_height + 1, :] = 1  # step block: upstream, sitting on the bottom

    return solid
