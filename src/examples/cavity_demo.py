# lid-driven cavity on the 2D numpy solver, plots the vortex
import numpy as np
import matplotlib.pyplot as plt
from src.lbm.advance import initial
from src.lbm.boundary_conditions import bounce_back, moving_wall
from src.lbm.collision import collide
from src.lbm.moments import macroscopic
from src.lbm.stream import stream
from src.post import plotting
from src.post.run_log import RunRecord

grid_size = 128
lid_velocity = 0.1
reynolds_number = 100
viscosity = lid_velocity * grid_size / reynolds_number
relaxation_time = 3 * viscosity + 0.5
steps = 15000

def main():
    """
    Run the NumPy cavity, log the centerline minimum against Ghia, plot speed and streamlines.
    """

    # geometry: 4 solid walls, top row is the moving lid
    solid = np.zeros((grid_size, grid_size), bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    lid = np.zeros((grid_size, grid_size), bool)
    lid[:, -1] = True
    stationary = solid & ~lid  # the 3 fixed walls

    populations = initial(grid_size, grid_size)
    run = RunRecord("cavity_2d_numpy", None, steps=steps, u_ref=lid_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"N={grid_size}", collision="BGK", sgs="none", walls="staircase BB",
                    boundaries="moving lid / no-slip walls", forcing="none", notes="NumPy CPU solver")
    for _ in range(steps):
        populations = collide(populations, relaxation_time, solid)
        populations = stream(populations)
        populations = bounce_back(populations, stationary)
        populations = moving_wall(populations, lid, lid_velocity)
    run.stop()
    density, velocity = macroscopic(populations)
    run.finish(u=velocity, rho=density, solid=solid.astype(np.int32),
               metric="u_min/U centerline", value=round(float(velocity[0, grid_size // 2, 1:-1].min() / lid_velocity), 4), reference=-0.2109)

    plotting.plot_velocity_magnitude(velocity)
    plt.savefig("src/examples/results/velocity_magnitude.png")
    plotting.plot_streamlines(velocity)
    plt.savefig("src/examples/results/streamlines.png")
    plt.show()

if __name__ == "__main__":
    main()
