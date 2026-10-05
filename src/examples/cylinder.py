# flow past a cylinder (Phase 2 gate): Re=100 -> Cd ~1.3-1.4, Strouhal ~0.16-0.20; optional spin (Magnus)
# usage: python -m src.examples.cylinder [spin_ratio] [--velocity U] [--diameter D] [--reynolds Re]
import argparse
import numpy as np
from src.geometry.cylinder_body import cylinder, spin_wall_velocity
from src.geometry.wall_fraction import wall_fraction_cylinder
from src.post.vtk import write_field
from src.post.statistics import dominant_frequency
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case
from src.run.runner import run_case

# command-line options (defaults reproduce the reference case, run 71 at spin 1)
parser = argparse.ArgumentParser()
parser.add_argument("spin_ratio", nargs="?", type=float, default=0.0)  # alpha = omega R / U
parser.add_argument("--velocity", type=float, default=0.1)  # U (envelope E3: Mach sweep at fixed Re)
parser.add_argument("--diameter", type=int, default=45)  # D, cells
parser.add_argument("--reynolds", type=float, default=100)
options = parser.parse_args()
spin_ratio = options.spin_ratio

# geometry: proportions of the reference domain (1800 x 600 x 4 at D = 45)
diameter = options.diameter
nx = 40 * diameter
ny = round(40 / 3 * diameter)
nz = 4
center_x = 10 * diameter
center_y = ny // 2
frontal_area = diameter * nz

# flow
free_stream_velocity = options.velocity
reynolds_number = options.reynolds
viscosity = free_stream_velocity * diameter / reynolds_number
relaxation_time = 3 * viscosity + 0.5  # ~0.59 at the defaults
angular_velocity = spin_ratio * free_stream_velocity / (diameter / 2)  # omega, rad per step, + = counter-clockwise about z

# timing: same number of flow-throughs for every U (60000 / 30000 at the defaults)
steps = round(6000 / free_stream_velocity)
warmup = steps // 2  # discard transient before averaging
sample_every = 20  # sample force every N steps (avoids per-step GPU sync)
check_every = 500  # progress readout interval

def main():
    """
    Run the cylinder, time-average Cd and Cl, get the Strouhal number from the lift spectrum.
    """

    # rigid rotation on a band covering both sides of the surface
    wall_velocity = None
    if spin_ratio != 0.0:
        wall_velocity = spin_wall_velocity(nx, ny, nz, center_x, center_y, diameter / 2, angular_velocity)

    body = Part(name="cylinder", solid=cylinder(nx, ny, nz, center_x, center_y, diameter / 2), reference_area=frontal_area,
                wall_fractions=wall_fraction_cylinder(nx, ny, nz, center_x, center_y, diameter / 2), wall_velocity=wall_velocity)
    case = Case(name="cylinder", tag=f"a{spin_ratio:g}_U{free_stream_velocity:g}",
                flow=Flow(free_stream_velocity=free_stream_velocity, reynolds_number=reynolds_number, reference_length=diameter),
                domain=Domain(nx=nx, ny=ny, nz=nz, y_boundary="free_slip", relax_width_x=24, relax_sigma=0.1, layer_kind="separate"),
                turbulence=Turbulence(sgs="none", smagorinsky_constant=0.0),
                timing=Timing(steps_override=steps, warmup_override=warmup, sample_every=sample_every, check_every=check_every),
                parts=[body], collision="bgk", inlet="neem", start="uniform")
    result = run_case(case, geometry=f"D={diameter}, alpha={spin_ratio}, U={free_stream_velocity:g}",
                      walls="Bouzidi cylinder", boundaries="NEEM inlet / pressure outlet / free-slip y / periodic z")
    sim = result.sim
    run = result.run
    drag_coefficients = result.force_coefficients[:, 0, 0]
    lift_force_history = result.forces[:, 0, 1]

    write_field(f"results/cylinder_a{spin_ratio:g}", sim.rho.to_numpy(), sim.u.to_numpy())
    velocity = sim.u.to_numpy()
    effective_velocity = float(velocity[0, center_x, 20, nz // 2])  # same x as the cylinder, near the wall, out of the wake
    print(f"U_eff = {effective_velocity:.4f}  ->  effective Re = {effective_velocity * diameter / viscosity:.0f}")

    drag_coefficient = float(np.mean(drag_coefficients))
    lift_coefficient = float(np.mean(lift_force_history)) / (0.5 * 1.0 * free_stream_velocity * free_stream_velocity * frontal_area)  # Magnus: alpha > 0 (CCW) -> Cl < 0
    print(f"Cl (time-averaged) = {lift_coefficient:.3f}   [alpha = {spin_ratio}; Magnus sign for CCW spin: negative]")

    # Strouhal from the lift (F_y) signal: refined spectral peak, cross-checked by zero crossings
    np.save(f"results/cylinder_lift_a{spin_ratio:g}_U{free_stream_velocity:g}.npy", np.array(lift_force_history))
    shedding_frequency, zero_crossing_frequency = dominant_frequency(lift_force_history, sample_every)
    strouhal = shedding_frequency * diameter / free_stream_velocity
    strouhal_zero_crossing = zero_crossing_frequency * diameter / free_stream_velocity

    print(f"Re = {reynolds_number}, D = {diameter}, tau = {relaxation_time:.3f}, blockage = {diameter / ny:.1%}")
    print(f"Cd (time-averaged) = {drag_coefficient:.3f}   [reference ~1.3-1.4]")
    print(f"Strouhal = {strouhal:.4f} (zero crossings {strouhal_zero_crossing:.4f})")
    run.finish(metric="Cd", value=round(drag_coefficient, 4), reference=1.4, other=f"Cl {lift_coefficient:.3f}; St {strouhal:.4f} (zc {strouhal_zero_crossing:.4f}); U_eff {effective_velocity:.4f}")

if __name__ == "__main__":
    main()
