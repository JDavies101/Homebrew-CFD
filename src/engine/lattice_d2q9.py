# D2Q9 lattice constants for the 2D taichi engine
import numpy as np

# number of directions, each cell has 9 populations
direction_count = 9  # Q
# lattice sound speed squared
sound_speed_squared = 1.0 / 3.0  # c_s^2
dimension = 2  # D

# int array of direction vectors
lattice_velocities = np.array([  # c_q (E)
    [0, 0],  # rest
    [1, 0],  # east
    [0, 1],  # north
    [-1, 0],  # west
    [0, -1],  # south
    [1, 1],  # north-east
    [-1, 1],  # north-west
    [-1, -1],  # south-west
    [1, -1]  # south-east
], dtype=np.int32)

# float array of weights
lattice_weights = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)  # w_q (W)
# int array of opposite indices
opposite_direction = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)  # OPP
