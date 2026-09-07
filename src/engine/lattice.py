import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

Q = 9 
# lattice sound speed
CS2 = 1.0 / 3.0 
# dimension
D = 2

# int field of direction vectors
E = ti.field(ti.i32, shape=(Q, D))
E.from_numpy(np.array([
    [0,0], # rest
    [1,0], # E
    [0,1], # N
    [-1,0], # W
    [0,-1], # S
    [1,1], # NE
    [-1,1], # NW
    [-1,-1], # SW
    [1,-1] # SE
], dtype=np.int32))

# float field of weights 
W = ti.field(ti.f32, shape=(Q))
W.from_numpy(np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32))
# int field of opposite indices
OPP = ti.field(ti.i32, shape=(Q))
OPP.from_numpy(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32))