# lid-driven cavity on the 2D taichi engine (gpu)
import numpy as np
import matplotlib.pyplot as plt
from src.engine.simulation import Simulation
from src.engine import lattice_d2q9 as d2q9
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
    Run the cavity, log the centerline minimum against Ghia, plot speed and streamlines.
    """

    sim = Simulation(grid_size, grid_size, backend="cuda")

    # geometry: 3 solid walls, top row = moving lid
    solid = np.zeros((grid_size, grid_size), np.int32)
    solid[0, :] = 1
    solid[-1, :] = 1
    solid[:, 0] = 1
    solid[:, -1] = 0  # the lid row (corners included) is lid, not solid
    lid = np.zeros((grid_size, grid_size), np.int32)
    lid[:, -1] = 1
    sim.solid.from_numpy(solid)
    sim.lid.from_numpy(lid)

    # start at rest equilibrium, then run
    sim.f.from_numpy(np.tile(d2q9.lattice_weights[:, None, None], (1, grid_size, grid_size)).astype(np.float32))
    run = RunRecord("cavity_2d_gpu", sim, steps=steps, u_ref=lid_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"N={grid_size}", collision="BGK", sgs="none", walls="staircase BB",
                    boundaries="moving lid / no-slip walls", forcing="none")
    sim.run(steps, relaxation_time, lid_velocity)
    run.stop()

    sim.macroscopic()
    velocity = sim.u.to_numpy()
    run.finish(metric="u_min/U centerline", value=round(float(velocity[0, grid_size // 2, :].min() / lid_velocity), 4), reference=-0.2109)
    plotting.plot_velocity_magnitude(velocity)
    plt.savefig("src/examples/results/velocity_magnitude_gpu.png")
    plotting.plot_streamlines(velocity)
    plt.savefig("src/examples/results/streamlines_gpu.png")
    plt.show()

if __name__ == "__main__":
    main()
