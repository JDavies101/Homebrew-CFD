# flow past a sphere
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice_d3q19 as L
from src.post.progress import Progress
from src.post.vtk import write_field
from src.geometry.sphere_body import sphere
from src.geometry.wall_fraction import wall_fraction_sphere
from src.post.run_log import RunRecord
import sys
from src.geometry.mesh import icosphere, write_stl, read_stl
from src.geometry.mesh_distance import sdf_from_mesh, q_from_mesh
from src.geometry.sponge import relax_profile

D = 20                      # sphere diameter in cells
nx = 384
ny = 128
nz = 128
U = 0.1                     # inlet speed (keep < ~0.1 for low Mach)
Re = 50                     # low Re -> tau ~0.62, damps the free-slip-corner instability
nu = U * D / Re
tau = 3 * nu + 0.5
cx = 120
cy = ny // 2
cz = nz // 2
geom = sys.argv[1] if len(sys.argv) > 1 else "analytic"    # "analytic" or "stl"
steps = 20000               # sphere wake is steady at this Re -> reaches steady state
check_every = 500           # progress + health readout interval
A = np.pi * (D/2) ** 2                 # frontal area

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda", interp=True)
    
    ref = sphere(nx, ny, nz, cx, cy, cz, D/2)
    if geom == "stl":
        write_stl("results/sphere.stl", icosphere((cx, cy, cz), D/2, subdivisions=5))
        tris = read_stl("results/sphere.stl")
        grid, phi = sdf_from_mesh(tris, nx, ny, nz, backend="cuda")
        solid = (grid < 0.0).astype(np.int32)
        q = q_from_mesh(tris, phi, nx, ny, nz, backend="cuda")
        print(f"STL: {len(tris)} triangles, mask cells differing from analytic: {int((solid != ref).sum())}")
    else:
        solid = ref
        q = wall_fraction_sphere(nx, ny, nz, cx, cy, cz, D/2)
    sim.solid.from_numpy(solid)
    sim.set_wall_fractions(q)

    sim.init_equilibrium(np.full((nx, ny, nz), U, np.float32),      # start in uniform flow at U
                         np.zeros((nx, ny, nz), np.float32),
                         np.zeros((nx, ny, nz), np.float32))
    sim.sigma.from_numpy(relax_profile(nx, nz, 24, 0, 0.1))           # x absorbing layers only

    run = RunRecord("sphere", sim, steps=steps, u_ref=U, nu=nu, tau=round(tau, 6), Re=Re,
                    geometry=f"D={D}, {geom}", collision="TRT", sgs="none", walls="Bouzidi",
                    boundaries="regularized NEEM inlet / pressure outlet / free-slip y,z", forcing="none")
    prog = Progress(steps)
    for s in range(steps):
        sim.collide_trt(tau)
        sim.sponge_relax(U)            # absorb acoustic waves at inlet/outlet (free-stream target)
        sim.fc.copy_from(sim.f)        # snapshot post-collision
        sim.stream()
        sim.inlet_neem_open(U)          # velocity inlet: u = U imposed, rho from the interior, regularized f_neq
        sim.outlet_pressure(1.0)        # pins the mean density; the zero-gradient outlet let it drift to 1.08
        sim.free_slip_y()
        sim.free_slip_z()
        sim.bounce_back_interp()

        if s == steps - 5000:
            sim.drag_interp() # drag only when it is read
            F_mid = float(sim.force.to_numpy()[0])

        if s % check_every == 0:
            hi = sim.f_absmax()                      # cheap GPU reduction, no full copy
            if not np.isfinite(hi) or hi > 1e29:     # NaN/inf: pull f once to localize
                prog.done()
                fnp = sim.f.to_numpy()
                bad = ~np.isfinite(fnp).all(axis=0)
                w = np.argwhere(bad)
                print(f"blow-up at step {s}: {len(w)} cells  "
                      f"x[{w[:,0].min()}-{w[:,0].max()}] "
                      f"y[{w[:,1].min()}-{w[:,1].max()}] "
                      f"z[{w[:,2].min()}-{w[:,2].max()}]")
                run.finish(metric="blow-up step", value=s, status="bad", reason="NaN/inf in f")
                return
            prog.update(s, float(hi))
    
    prog.done()
    run.stop()

    # steady flow -> single force reading
    sim.macroscopic()
    write_field(f"results/sphere_{geom}", sim.rho.to_numpy(), sim.u.to_numpy())
   
    u = sim.u.to_numpy()
    U_eff = float(u[0, cx, 20, nz//2])   # near the wall, out of the wake
    sim.drag_interp() # force on the final state (f and fc from the last step)
    F = sim.force.to_numpy()
    cd = float(F[0] / (0.5 * 1.0 * U * U * A))            # nominal free stream: the inlet now delivers U
    cd_ref = (24 / Re) * (1 + 0.15 * Re ** 0.687)          # Schiller-Naumann at the nominal Re

    drift = (float(F[0]) - F_mid) / float(F[0])
    print(f"drag change over last 5000 steps: {100 * drift:+.2f}%  (steady if |.| < 0.5%)")
    
    print(f"U_eff beside sphere = {U_eff:.4f} (check: ~U, slightly above from blockage)")
    print(f"Cd = {cd:.3f}   [Schiller-Naumann at Re={Re} ~ {cd_ref:.2f}]")
    run.finish(metric="Cd", value=round(cd, 4), reference=round(cd_ref, 4), other=f"U_eff {U_eff:.4f} drift {100 * drift:+.2f}%")

if __name__ == "__main__":
    main()
