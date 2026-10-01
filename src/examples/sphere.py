# flow past a sphere (Re 50 default): Bouzidi walls from the analytic sphere or an STL icosphere
# usage: python -m src.examples.sphere [analytic|stl] [--diameter D] [--width W] [--velocity U] [--reynolds Re]
import argparse
import numpy as np
from src.geometry.mesh import icosphere, write_stl, read_stl
from src.geometry.mesh_distance import sdf_from_mesh, q_from_mesh
from src.geometry.sphere_body import sphere
from src.geometry.wall_fraction import wall_fraction_sphere
from src.post.vtk import write_field
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case
from src.run.runner import run_case

# command-line options (defaults reproduce the reference case, run 66)
parser = argparse.ArgumentParser()
parser.add_argument("geometry_source", nargs="?", default="analytic", choices=["analytic", "stl"])
parser.add_argument("--diameter", type=int, default=20)  # D, cells (envelope E1)
parser.add_argument("--width", type=float, default=6.4)  # cross-section in diameters (envelope E4)
parser.add_argument("--velocity", type=float, default=0.1)  # U
parser.add_argument("--reynolds", type=float, default=50)
options = parser.parse_args()
geometry_source = options.geometry_source

# geometry: proportions of the reference domain (384 x 128 x 128 at D = 20)
diameter = options.diameter
nx = round(19.2 * diameter)
ny = round(options.width * diameter)
nz = ny
center_x = 6 * diameter
center_y = ny // 2
center_z = nz // 2
frontal_area = np.pi * (diameter / 2) ** 2

# flow
free_stream_velocity = options.velocity
reynolds_number = options.reynolds  # nu = U D / Re and tau follow in Flow

# timing: 100 convective times D / U (20000 steps at the defaults), drift over the last quarter
steps = round(100 * diameter / free_stream_velocity)
drift_window = steps // 4
check_every = 500  # progress + health readout interval

def main():
    """
    Run the sphere to steady state and log Cd against Schiller-Naumann.
    """

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

    # start in uniform flow at U, x absorbing layers only
    case = Case(name="sphere", tag=geometry_source,
                flow=Flow(free_stream_velocity=free_stream_velocity, reynolds_number=reynolds_number, reference_length=diameter),
                domain=Domain(nx=nx, ny=ny, nz=nz, y_boundary="free_slip", side_walls="free_slip", relax_width_x=24, relax_sigma=0.1,
                              layer_kind="separate"),
                turbulence=Turbulence(sgs="none", smagorinsky_constant=0.0),
                timing=Timing(steps_override=steps, warmup_override=0, sample_window="none", check_every=check_every),
                parts=[Part(name="sphere", solid=solid, reference_area=frontal_area, wall_fractions=wall_fractions)],
                collision="trt", inlet="neem_open", start="uniform")

    # drag at the start of the drift window
    force_x_before = 0.0

    def read_force_before(sim, time_step):
        """
        Read the drag once, drift_window steps before the end.

        Returns False (never stops the run).
        """

        nonlocal force_x_before
        if time_step == steps - drift_window:
            sim.drag_interp()
            force_x_before = float(sim.force.to_numpy()[0])

        return False

    result = run_case(case, after_step=read_force_before, geometry=f"D={diameter}, W={options.width:g}D, {geometry_source}",
                      walls="Bouzidi sphere", boundaries="regularized NEEM inlet / pressure outlet / free-slip y,z")
    sim = result.sim
    run = result.run
    if result.blow_up_step >= 0:
        run.finish(metric="blow-up step", value=result.blow_up_step, status="bad", reason="NaN/inf in f")
        return

    # steady flow -> single force reading
    write_field(f"results/sphere_{geometry_source}", sim.rho.to_numpy(), sim.u.to_numpy())

    velocity = sim.u.to_numpy()
    effective_velocity = float(velocity[0, center_x, 20, nz // 2])  # near the wall, out of the wake
    sim.drag_interp()  # force on the final state (f and fc from the last step)
    force = sim.force.to_numpy()
    drag_coefficient = float(force[0] / (0.5 * 1.0 * free_stream_velocity * free_stream_velocity * frontal_area))  # nominal free stream: the inlet delivers U
    reference_drag_coefficient = (24 / reynolds_number) * (1 + 0.15 * reynolds_number ** 0.687)  # Schiller-Naumann at the nominal Re

    drift = (float(force[0]) - force_x_before) / float(force[0])
    print(f"drag change over last {drift_window} steps: {100 * drift:+.2f}%  (steady if |.| < 0.5%)")
    print(f"U_eff beside sphere = {effective_velocity:.4f} (check: ~U, slightly above from blockage)")
    print(f"Cd = {drag_coefficient:.3f}   [Schiller-Naumann at Re={reynolds_number} ~ {reference_drag_coefficient:.2f}]")
    run.finish(metric="Cd", value=round(drag_coefficient, 4), reference=round(reference_drag_coefficient, 4), other=f"U_eff {effective_velocity:.4f} drift {100 * drift:+.2f}%")

if __name__ == "__main__":
    main()
