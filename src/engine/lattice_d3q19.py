# D3Q19 lattice constants: velocities, weights, opposites, y/z mirrors
import numpy as np

# number of directions, each cell has 19 populations
direction_count = 19  # Q
# lattice sound speed squared
sound_speed_squared = 1.0 / 3.0  # c_s^2
dimension = 3  # D

# int array of direction vectors
lattice_velocities = np.array([  # c_q (E)
    [0, 0, 0],  # rest
    [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],  # 6 faces
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],  # xy edges
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],  # xz edges
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]  # yz edges
], dtype=np.int32)

# float array of weights from the squared direction length: rest 1/3, face 1/18, edge 1/36
squared_length = (lattice_velocities * lattice_velocities).sum(axis=1)
lattice_weights = np.select([squared_length == 0, squared_length == 1, squared_length == 2], [1 / 3, 1 / 18, 1 / 36]).astype(np.float64)  # w_q (W)

# int array of opposite indices
opposite_direction = np.zeros(direction_count, dtype=np.int32)  # OPP
for q in range(direction_count):
    opposite_direction[q] = np.where((lattice_velocities == -lattice_velocities[q]).all(axis=1))[0][0]

# int arrays of mirrored indices: the direction reflected across a y wall (flip c_y) or a z wall (flip c_z)
mirror_y = np.zeros(direction_count, dtype=np.int32)  # MIRROR_Y
flip_y = np.array([1, -1, 1])
for q in range(direction_count):
    mirror_y[q] = np.where((lattice_velocities == lattice_velocities[q] * flip_y).all(axis=1))[0][0]
mirror_z = np.zeros(direction_count, dtype=np.int32)  # MIRROR_Z
flip_z = np.array([1, 1, -1])
for q in range(direction_count):
    mirror_z[q] = np.where((lattice_velocities == lattice_velocities[q] * flip_z).all(axis=1))[0][0]
