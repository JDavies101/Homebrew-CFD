import numpy as np
from src.lbm.advance import initial
from src.lbm.moments import macroscopic
from src.lbm.advance import stream
from src.lbm.collision import collide
from src.lbm.boundary_conditions import bounce_back, moving_wall
from src.post import plotting
import matplotlib.pyplot as plt

N = 128
U = 0.1
Re = 100
nu = U * N / Re
tau = 3 * nu + 0.5
steps = 15000

def profile():
    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((N, N), bool)
    solid[0,:] = solid[-1,:] = solid[:,0] = solid[:,-1] = True
    lid = np.zeros((N, N), bool); lid[:,-1] = True
    stationary = solid & ~lid          # the 3 fixed walls

    f = initial(N, N)
    for _ in range(steps):
        f = collide(f, tau)            # from src.lbm.collision
        f = stream(f)
        f = bounce_back(f, stationary)
        f = moving_wall(f, lid, U)
    rho, u = macroscopic(f)
    
    fig, ax = plotting.plot_velocity_magnitude(u)
    plt.savefig("src/examples/results/velocity_magnitude.png")
    fig, ax = plotting.plot_streamlines(u)
    plt.savefig("src/examples/results/streamlines.png")
    plt.show()

if __name__ == "__main__":
    profile()