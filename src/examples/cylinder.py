# flow past a cylinder (Phase 2 gate): Re=100 -> Cd ~1.3-1.4, Strouhal ~0.16-0.20
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.post.progress import Progress
from src.post.vtk import write_field
from src.geometry.wall_fraction import wall_fraction_cylinder

D = 45                      # cylinder diameter in cells
nx = 1800
ny = 600                   
nz = 4
U = 0.1                     # inlet speed (keep < ~0.1 for low Mach)
Re = 100
nu = U * D / Re
tau = 3 * nu + 0.5          # ~0.59, safely above 0.5
cx = 450                   
cy = ny // 2
steps = 60000
warmup = 30000              # discard transient before averaging
sample_every = 20           # sample force every N steps (avoids per-step GPU sync)
check_every = 500           # progress readout interval
A = D * nz                  # frontal area

# disc in x-y, spanning the periodic z -> a cylinder
def cylinder():
    solid = np.zeros((nx, ny, nz), np.int32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    disc = (X - cx) ** 2 + (Y - cy) ** 2 < (D / 2) ** 2
    solid[disc] = 1
    return solid

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(cylinder())
    sim.q.from_numpy(wall_fraction_cylinder(nx, ny, nz, cx, cy, D/2))
    sim.f.from_numpy(np.tile(L.W[:, None, None, None], (1, nx, ny, nz)).astype(np.float32))

    cd_samples = []
    fy_history = []
    prog = Progress(steps)
    for s in range(steps):
        sim.collide(tau)
        sim.fc.copy_from(sim.f)        # snapshot post-collision BEFORE streaming
        sim.drag()          # force measured post collision
        sim.stream()
        sim.inlet_neem(U) # use Guo non-equilibrium extrapolation
        sim.outlet()
        sim.free_slip_y() # top/bottom now free-slip instead of periodic
        sim.bounce_back_interp()       # replaces bounce_back for the cylinder

        if s >= warmup and s % sample_every == 0:
            F = sim.force.to_numpy()
            cd_samples.append(F[0] / (0.5 * 1.0 * U * U * A))
            fy_history.append(F[1])

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    sim.macroscopic()
    write_field("results/cylinder", sim.rho.to_numpy(), sim.u.to_numpy())
    u = sim.u.to_numpy()
    prog.done()
    U_eff = float(u[0, cx, 20, nz//2])   # same x as cylinder, but near the wall, out of the wake
    print(f"U_eff = {U_eff:.4f}  ->  effective Re = {U_eff*D/nu:.0f}")

    cd = float(np.mean(cd_samples))

    # Strouhal from the dominant frequency of the lift (F_y) signal
    fy = np.array(fy_history) - np.mean(fy_history)
    spec = np.abs(np.fft.rfft(fy))
    freqs = np.fft.rfftfreq(len(fy), d=sample_every)
    f_shed = freqs[1 + np.argmax(spec[1:])]
    strouhal = f_shed * D / U

    print(f"Re = {Re}, D = {D}, tau = {tau:.3f}, blockage = {D/ny:.1%}")
    print(f"Cd (time-averaged) = {cd:.3f}   [reference ~1.3-1.4]")
    print(f"Strouhal = {strouhal:.3f}        [reference ~0.16-0.20]")

if __name__ == "__main__":
    main()
