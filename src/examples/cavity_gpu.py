# lid-driven cavity on the 2D taichi engine (gpu)
import numpy as np
from src.engine.simulation import Simulation
from src.engine import lattice as L
from src.post import plotting
import matplotlib.pyplot as plt

N = 128
U = 0.1
Re = 100
nu = U * N / Re
tau = 3 * nu + 0.5
steps = 15000

def main():
    sim = Simulation(N, N, backend="cuda")

    # geometry: 4 solid walls, top row = moving lid
    solid = np.zeros((N, N), np.int32)
    solid[0,:] = solid[-1,:] = solid[:,0] = solid[:,-1] = 1
    lid = np.zeros((N, N), np.int32); lid[:,-1] = 1; solid[:,-1] = 0
    sim.solid.from_numpy(solid)
    sim.lid.from_numpy(lid)

    # init at rest equilibrium, then run
    sim.f.from_numpy(np.tile(L.W[:, None, None], (1, N, N)).astype(np.float32))
    sim.run(steps, tau, U)

    # read velocity, reuse your plotting
    sim.macroscopic()
    u = sim.u.to_numpy()
    plotting.plot_velocity_magnitude(u)
    plt.savefig("src/examples/results/velocity_magnitude_gpu.png")
    plotting.plot_streamlines(u)
    plt.savefig("src/examples/results/streamlines_gpu.png")
    plt.show()

if __name__ == "__main__":
    main()