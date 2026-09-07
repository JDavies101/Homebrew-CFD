import numpy as np

Q = 9 
# lattice sound speed
CS2 = 1.0 / 3.0 
D = 2 # dimension

# int array of direction vectors
E = np.array([
    [0,0], # rest
    [1,0], # E
    [0,1], # N
    [-1,0], # W
    [0,-1], # S
    [1,1], # NE
    [-1,1], # NW
    [-1,-1], # SW
    [1,-1] # SE
], dtype=np.int32)

# float array of weights 
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64) 
# int array of opposite indices
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32) 