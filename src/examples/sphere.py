# flow past a sphere at Re 50: Bouzidi walls from the analytic sphere or an STL icosphere
# usage: python -m src.examples.sphere [analytic|stl]
import sys
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.mesh import icosphere, write_stl, read_stl
from src.geometry.mesh_distance import sdf_from_mesh, q_from_mesh
from src.geometry.sphere_body import sphere
from src.geometry.sponge import relax_profile
from src.geometry.wall_fraction import wall_fraction_sphere
from src.post.progress import Progress
from src.post.run_log import RunRecord
from src.post.vtk import write_field

geometry_source = sys.argv[1] if len(sys.argv) > 1 else "analytic"  # "analytic" or "stl"

# geometry
diameter = 20  # D, cells
nx = 384
ny = 128
nz = 128
center_x = 120
center_y = ny // 2
center_z = nz // 2
frontal_area = np.pi * (diameter / 2) ** 2

# flow
free_stream_velocity = 0.1  # U, keep < ~0.1 for low Mach
reynolds_number = 50  # low Re -> tau ~0.62, damps the free-slip-corner instability
viscosity = free_stream_velocity * diameter / reynolds_number
relaxation_time = 3 * viscosity + 0.5

# timing
steps = 20000  # sphere wake is steady at this Re -> reaches steady state
check_every = 500  # progress + health readout interval

def main():
    """
    Run the sphere to steady state and log Cd against Schiller-Naumann.
    """

    sim = Simulation3D(nx, ny, nz, backend="cuda", interp=True)

    # geometry: solid mask + Bouzidi wall fractions
    analytic_solid = sphere(nx, ny, nz, center_x, center_y, center_z, diameter / 2)
    if geometry_source == "stl":
        write_stl("results/sphere.stl", icosphere((center_x, center_y, center_z), diameter / 2, subdivisions=5))
        triangles = read_stl("results/sphere.stl")
        grid, phi = sdf_from_mesh(triangles, nx, ny, nz, backend="cuda")
        solid = (grid < 0.0).astype(np.int32)
        wall_fractions = q_from_mesh(triangles, phi, nx, ny, nz, backend="cuda")
        print(f"STL: {len(triangles)} triangles, mask cells differing from analytic: {int((solid != analytic_solid).sum())}")
    else:
        solid = analytic_solid
        wall_fractions = wall_fraction_sphere(nx, ny, nz, center_x, center_y, center_z, diameter / 2)
    sim.solid.from_numpy(solid)
    sim.set_wall_fractions(wall_fractions)

    # start in uniform flow at U
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(np.full((nx, ny, nz), free_stream_velocity, np.float32), zero, zero)
    sim.sigma.from_numpy(relax_profile(nx, nz, 24, 0, 0.1))  # x absorbing layers only

    run = RunRecord("sphere", sim, steps=steps, u_ref=free_stream_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"D={diameter}, {geometry_source}", collision="TRT", sgs="none", walls="Bouzidi",
                    boundaries="regularized NEEM inlet / pressure outlet / free-slip y,z", forcing="none")
    progress = Progress(steps)
    for time_step in range(steps):
        sim.collide_trt(relaxation_time)
        sim.sponge_relax(free_stream_velocity)  # absorb acoustic waves at inlet/outlet (free-stream target)
        sim.fc.copy_from(sim.f)  # snapshot post-collision
        sim.stream()
        sim.inlet_neem_open(free_stream_velocity)  # velocity inlet: u = U imposed, rho from the interior, regularized f_neq
        sim.outlet_pressure(1.0)  # pins the mean density; the zero-gradient outlet let it drift to 1.08
        sim.free_slip_y()
        sim.free_slip_z()
        sim.bounce_back_interp()

        # drag only when it is read
        if time_step == steps - 5000:
            sim.drag_interp()
            force_x_before = float(sim.force.to_numpy()[0])

        if time_step % check_every == 0:
            max_population = sim.f_absmax()  # cheap GPU reduction, no full copy
            # NaN / inf: pull f once to localize
            if not np.isfinite(max_population) or max_population > 1e29:
                progress.done()
                populations = sim.f.to_numpy()
                bad_cells = np.argwhere(~np.isfinite(populations).all(axis=0))
                print(f"blow-up at step {time_step}: {len(bad_cells)} cells  "
                      f"x[{bad_cells[:, 0].min()}-{bad_cells[:, 0].max()}] "
                      f"y[{bad_cells[:, 1].min()}-{bad_cells[:, 1].max()}] "
                      f"z[{bad_cells[:, 2].min()}-{bad_cells[:, 2].max()}]")
                run.finish(metric="blow-up step", value=time_step, status="bad", reason="NaN/inf in f")
                return
            progress.update(time_step, float(max_population))

    progress.done()
    run.stop()

    # steady flow -> single force reading
    sim.macroscopic()
    write_field(f"results/sphere_{geometry_source}", sim.rho.to_numpy(), sim.u.to_numpy())

    velocity = sim.u.to_numpy()
    effective_velocity = float(velocity[0, center_x, 20, nz // 2])  # near the wall, out of the wake
    sim.drag_interp()  # force on the final state (f and fc from the last step)
    force = sim.force.to_numpy()
    drag_coefficient = float(force[0] / (0.5 * 1.0 * free_stream_velocity * free_stream_velocity * frontal_area))  # nominal free stream: the inlet delivers U
    reference_drag_coefficient = (24 / reynolds_number) * (1 + 0.15 * reynolds_number ** 0.687)  # Schiller-Naumann at the nominal Re

    drift = (float(force[0]) - force_x_before) / float(force[0])
    print(f"drag change over last 5000 steps: {100 * drift:+.2f}%  (steady if |.| < 0.5%)")
    print(f"U_eff beside sphere = {effective_velocity:.4f} (check: ~U, slightly above from blockage)")
    print(f"Cd = {drag_coefficient:.3f}   [Schiller-Naumann at Re={reynolds_number} ~ {reference_drag_coefficient:.2f}]")
    run.finish(metric="Cd", value=round(drag_coefficient, 4), reference=round(reference_drag_coefficient, 4), other=f"U_eff {effective_velocity:.4f} drift {100 * drift:+.2f}%")

if __name__ == "__main__":
    main()
