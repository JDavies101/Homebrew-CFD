import numpy as np
from src.geometry.ahmed import ahmed
from src.post.plotting import plot_mask_slice, plot_velocity_slice
from src.engine.simulation3d import Simulation3D
from src.post.progress import Progress
from src.post.vtk import write_field

H = 32                      # small for iteration; go to 48 for real runs
Lb = round(1044/288 * H)
Wb = round(389 / 288 * H)
x0 = Lb                       # 1 body-length upstream
nx = 6 * Lb                   # up + body + 4-length wake
ny = round(3.75 * H)          # ~2.5 H of air above the body
nz = round(3.667 * H)         # ~10% blockage
U = 0.05                    # low Mach
Re_H = 100                  # laminar-ish separated wake; machinery, not the reference Cd
nu = U * H / Re_H           # ~0.0053 -> tau ~0.516 (TRT)
tau = 3*nu + 0.5
A = Wb * H                 # frontal area for Cd
T_ft = nx / U                        # flow-through time in steps (~14000 at H=32)
warmup = round(5 * T_ft)             # establish flow + wake
steps  = round(11 * T_ft)            # + ~6 flow-throughs (~30 shedding periods) to average
sample_every = 25   # sample force every N steps (avoids per-step GPU sync)
check_every  = max(1, steps // 300) # progress readout interval

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")

    body = ahmed(nx, ny, nz, x0, H)

    plot_mask_slice(body, axis=2, index=nz//2)[0].savefig("results/ahmed_xy.png")   # side view
    plot_mask_slice(body, axis=1, index=10)[0].savefig("results/ahmed_xz.png")      # plan view

    solid = body.copy()
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1   # + floor & ceiling (no-slip)
    sim.solid.from_numpy(solid); sim.body.from_numpy(body)

    sim.init_equilibrium(np.full((nx, ny, nz), U, np.float32),   # u_x = U everywhere
                     np.zeros((nx, ny, nz), np.float32),
                     np.zeros((nx, ny, nz), np.float32))

    cd = []
    prog = Progress(steps)
    for s in range(steps):

        sim.collide_full(tau, 0, 0, 1)   # TRT
        if s >= warmup and s % sample_every == 0:
            sim.drag_body()                  # body-only momentum exchange
        sim.drag_body()                  # body-only momentum exchange, pre-stream
        sim.stream()
        sim.inlet_neem_open(U)           # reuse: drives open rows, skips the solid floor at the inlet
        sim.outlet()
        sim.free_slip_z()                # side walls
        sim.bounce_back()                # floor + ceiling + body

        if s >= warmup and s % sample_every == 0:
            cd.append(float(sim.force.to_numpy()[0]) / (0.5 * U*U * A))

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)
    
    sim.macroscopic()
    u = sim.u.to_numpy()
    write_field("results/ahmed", sim.rho.to_numpy(), u)
    plot_velocity_slice(u, axis=2, index=nz//2, comp=0)[0].savefig("results/ahmed_wake.png", dpi=130)
    prog.done()
    print("Cd =", np.mean(cd))

if __name__ == "__main__":
    main()
