# lid-driven cavity on the 2D numpy solver, plots the vortex
import numpy as np
from src.lbm.advance import initial
from src.lbm.moments import macroscopic
from src.lbm.advance import stream
from src.lbm.collision import collide
from src.lbm.boundary_conditions import bounce_back, moving_wall
from src.post import plotting
import matplotlib.pyplot as plt
from src.post.run_log import RunRecord

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
    run = RunRecord("cavity_2d_numpy", None, steps=steps, u_ref=U, nu=nu, tau=round(tau, 6), Re=Re,
                    geometry=f"N={N}", collision="BGK", sgs="none", walls="staircase BB",
                    boundaries="moving lid / no-slip walls", forcing="none", notes="NumPy CPU solver")
    for _ in range(steps):
        f = collide(f, tau, solid)            # from src.lbm.collision
        f = stream(f)
        f = bounce_back(f, stationary)
        f = moving_wall(f, lid, U)
    run.stop()
    rho, u = macroscopic(f)
    run.finish(u=u, rho=rho, solid=solid.astype(np.int32),
               metric="u_min/U centerline", value=round(float(u[0, N // 2, :].min() / U), 4), reference=-0.2109)
    
    fig, ax = plotting.plot_velocity_magnitude(u)
    plt.savefig("src/examples/results/velocity_magnitude.png")
    fig, ax = plotting.plot_streamlines(u)
    plt.savefig("src/examples/results/streamlines.png")
    plt.show()

if __name__ == "__main__":
    profile()