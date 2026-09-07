import numpy as np

Q = 19 
# lattice sound speed
CS2 = 1.0 / 3.0 
D = 3 # dimension

# int array of direction vectors
E = np.array([
    [0,0,0], # rest
    [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [0,0,1], [0,0,-1], # 6 faces
    [1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0], # xy edges
    [1,0,1], [-1,0,-1], [1,0,-1], [-1,0,1], # xz edges
    [0,1,1], [0,-1,-1], [0,1,-1], [0,-1,1] # yz edges
], dtype=np.int32)

# float array of weights 
sq = (E**2).sum(axis=1)
W = np.select([sq==0, sq==1, sq==2], [1/3, 1/18, 1/36]).astype(np.float64)
# int array of opposite indices
OPP = np.zeros(Q, dtype=np.int32)
for i in range(Q):
    OPP[i] = np.where((E == -E[i]).all(axis=1))[0][0]
# int array of opposite normal indices
MIRROR_Y = np.zeros(Q, dtype=np.int32)
flip = np.array([1, -1, 1])
for i in range(Q):
    MIRROR_Y[i] = np.where((E == E[i] * flip).all(axis = 1))[0][0]
MIRROR_Z = np.zeros(Q, dtype=np.int32)
flip_z = np.array([1, 1, -1])
for i in range(Q):
    MIRROR_Z[i] = np.where((E == E[i] * flip_z). all(axis = 1))[0][0]