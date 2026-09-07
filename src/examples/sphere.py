# flow past a sphere
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.post.vtk import write_field

D = 20                      # sphere diameter in cells
nx = 384
ny = 128
nz = 128
U = 0.1                     # inlet speed (keep < ~0.1 for low Mach)
Re = 50                     # low Re -> high tau (~0.8), damps free-slip-corner instability
nu = U * D / Re
tau = 3 * nu + 0.5
cx = 120
cy = ny // 2
cz = nz // 2
steps = 20000               # sphere wake is steady at this Re -> reaches steady state
check_every = 500           # progress + health readout interval
A = np.pi * (D/2) ** 2                 # frontal area

def sphere():
    solid = np.zeros((nx, ny, nz), np.int32)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),indexing="ij")
    ball = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2< (D / 2) ** 2
    solid[ball] = 1
    return solid

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(sphere())
    sim.f.from_numpy(np.tile(L.W[:, None, None, None], (1, nx, ny, nz)).astype(np.float32))

    prog = Progress(steps)
    for s in range(steps):
        sim.collide_trt(tau)
        sim.drag()          # force measured post collision
        sim.stream()
        sim.inlet(U)        # equilibrium inlet: stable at free-slip corners (NEEM diverges there)
        sim.outlet()
        sim.free_slip_y()
        sim.free_slip_z()
        sim.bounce_back()

        if s % check_every == 0:
            fnp = sim.f.to_numpy()
            nan_cells = np.isnan(fnp).any(axis=0)
            if nan_cells.any():                     # localize the first NaNs
                w = np.argwhere(nan_cells)
                prog.done()
                print(f"NaN at step {s}: {len(w)} cells  "
                      f"x[{w[:,0].min()}-{w[:,0].max()}] "
                      f"y[{w[:,1].min()}-{w[:,1].max()}] "
                      f"z[{w[:,2].min()}-{w[:,2].max()}]")
                return
            prog.update(s, float(np.abs(fnp).max()))
    prog.done()

    # steady flow -> single force reading
    sim.macroscopic()
    write_field("results/sphere", sim.rho.to_numpy(), sim.u.to_numpy())
    u = sim.u.to_numpy()
    U_eff = float(u[0, cx, 20, nz//2])   # near the wall, out of the wake
    Re_eff = U_eff * D / nu
    F = sim.force.to_numpy()
    cd = float(F[0] / (0.5 * 1.0 * U_eff * U_eff * A))   # normalize by the actual free-stream
    cd_ref = (24/Re_eff)*(1 + 0.15*Re_eff**0.687)        # Schiller-Naumann at the MEASURED Re
    print(f"U_eff = {U_eff:.4f}  ->  effective Re = {Re_eff:.0f}")
    print(f"Cd = {cd:.3f}   [Schiller-Naumann at Re={Re_eff:.0f} ~ {cd_ref:.2f}]")

if __name__ == "__main__":
    main()
