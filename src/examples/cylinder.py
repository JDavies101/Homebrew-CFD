# flow past a cylinder (Phase 2 gate): Re=100 -> Cd ~1.3-1.4, Strouhal ~0.16-0.20
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.post.vtk import write_field
from src.geometry.wall_fraction import wall_fraction_cylinder
from src.geometry.cylinder import cylinder
from src.post.run_log import RunRecord
import sys
from src.geometry.sponge import relax_profile

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
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0     # spin ratio omega R / U
omega = alpha * U / (D / 2)                                  # rad per step, + = counter-clockwise about z

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda", interp=True)
    sim.solid.from_numpy(cylinder(nx, ny, nz, cx, cy, D/2))
    sim.q.from_numpy(wall_fraction_cylinder(nx, ny, nz, cx, cy, D/2))
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(np.full((nx, ny, nz), U, np.float32), zero, zero)   # start in uniform flow
    sim.sigma.from_numpy(relax_profile(nx, nz, 24, 0, 0.1))                 # x absorbing layers
    if alpha != 0.0:
        X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
        dx = X - cx
        dy = Y - cy
        band = np.sqrt(dx * dx + dy * dy) < D / 2 + 2.0
        uw = np.zeros((3, nx, ny, nz), np.float32)
        uw[0][band] = (-omega * dy[band])[:, None]
        uw[1][band] = (omega * dx[band])[:, None]
        sim.uw.from_numpy(uw)

    cd_samples = []
    fy_history = []
    run = RunRecord("cylinder", sim, steps=steps, u_ref=U, nu=nu, tau=round(tau, 6), Re=Re,
                    geometry=f"D={D}, alpha={alpha}", warmup=warmup, collision="BGK", sgs="none", walls="Bouzidi",
                    boundaries="NEEM inlet / pressure outlet / free-slip y / periodic z", forcing="none",
                    sponge="relax x 24, sigma 0.1")
    prog = Progress(steps)
    for s in range(steps):
        sim.collide(tau)
        sim.sponge_relax(U)
        sim.fc.copy_from(sim.f)        # snapshot post-collision BEFORE streaming
        sim.stream()
        sim.inlet_neem(U) # use Guo non-equilibrium extrapolation
        sim.outlet_pressure(1.0)
        sim.free_slip_y() # top/bottom now free-slip instead of periodic
        sim.bounce_back_interp()       # replaces bounce_back for the cylinder
        sim.drag_interp()

        if s >= warmup and s % sample_every == 0:
            F = sim.force.to_numpy()
            cd_samples.append(F[0] / (0.5 * 1.0 * U * U * A))
            fy_history.append(F[1])

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    prog.done()                          # finish the bar (newline) before any other output
    run.stop()
    sim.macroscopic()
    write_field(f"results/cylinder_a{alpha:g}", sim.rho.to_numpy(), sim.u.to_numpy())
    u = sim.u.to_numpy()
    U_eff = float(u[0, cx, 20, nz//2])   # same x as cylinder, but near the wall, out of the wake
    print(f"U_eff = {U_eff:.4f}  ->  effective Re = {U_eff*D/nu:.0f}")

    cd = float(np.mean(cd_samples))
    cl = float(np.mean(fy_history)) / (0.5 * 1.0 * U * U * A)    # mean lift; Magnus: alpha > 0 (CCW) -> cl < 0
    print(f"Cl (time-averaged) = {cl:.3f}   [alpha = {alpha}; Magnus sign for CCW spin: negative]")

    # Strouhal from the dominant frequency of the lift (F_y) signal
    fy = np.array(fy_history) - np.mean(fy_history)
    spec = np.abs(np.fft.rfft(fy))
    freqs = np.fft.rfftfreq(len(fy), d=sample_every)
    f_shed = freqs[1 + np.argmax(spec[1:])]
    strouhal = f_shed * D / U

    print(f"Re = {Re}, D = {D}, tau = {tau:.3f}, blockage = {D/ny:.1%}")
    print(f"Cd (time-averaged) = {cd:.3f}   [reference ~1.3-1.4]")
    print(f"Strouhal = {strouhal:.3f}        [reference ~0.16-0.20]")
    run.finish(metric="Cd", value=round(cd, 4), reference=1.4, other=f"Cl {cl:.3f}; St {strouhal:.3f}; U_eff {U_eff:.4f}")

if __name__ == "__main__":
    main()
