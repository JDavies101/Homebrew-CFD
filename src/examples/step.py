# flow past a cylinder (Phase 2 gate): Re=100 -> Cd ~1.3-1.4, Strouhal ~0.16-0.20
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.post.vtk import write_field
from src.geometry.step import step

S = 30
ny = int(S/0.485) + 2      # gives ER ~ 1.94
nx = 60 + 26*S             # x_step=60 (2S), plenty of downstream
nz = 4
x_step = 60
U = 0.1
Re = 100
h = (ny-2) - S             # inlet channel height
nu = U * (2*h) / Re        # Armaly: Re on hydraulic diameter 2h
tau = 3*nu + 0.5
steps = 60000
sample_every = 20           # sample force every N steps (avoids per-step GPU sync)
check_every = 500           # progress readout interval

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(step(nx, ny, nz, x_step, S))
    sim.f.from_numpy(np.tile(L.W[:,None,None,None], (1,nx,ny,nz)).astype(np.float32))
    
    prog = Progress(steps)
    for s in range(steps):
        sim.collide_trt(tau)
        sim.stream()
        sim.inlet_neem_open(U)
        sim.outlet()
        sim.bounce_back()          # walls + step, all no-slip
        # progress/health as in cylinder

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    sim.macroscopic()
    ux = sim.u.to_numpy()[0, :, 1, nz//2]     # u_x along the floor
    xs = x_step
    sign = ux[xs:] > 0
    i_re = xs + np.argmax(sign)               # first index where u_x > 0 after the step
    x_r = i_re - x_step

    floor = sim.u.to_numpy()[0, x_step:, 1, nz//2]
    neg = np.where(floor < 0)[0]
    if len(neg):
        i0 = neg[0]                                   # bubble start
        after = np.where(floor[i0:] > 0)[0]
        x_r = (i0 + after[0]) if len(after) else -1   # reattach = first positive after reversal
        print(f"\nx_r/S = {x_r/S:.2f}")
    else:
        print("\nno recirculation")
    np.save("results/floor.npy", floor)

    u = sim.u.to_numpy()
    col = u[0, x_step-5, S+1:ny-1, :]      # open inlet rows, all z
    U_mean = float(col.mean())
    Re_eff = U_mean * 2*h / nu
    print(f"U_mean = {U_mean:.4f}   Armaly Re = {Re_eff:.0f}   x_r/S = {x_r/S:.2f}")

if __name__ == "__main__":
    main()
