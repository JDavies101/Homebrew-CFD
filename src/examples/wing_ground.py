# inverted NACA 4412 wing in ground effect (quasi-2D, periodic z): downforce vs ride height, moving ground
# usage: python -m src.examples.wing_ground [h_over_c] [angle_degrees] [--chord C] [--reynolds Re]
import argparse
import numpy as np
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf
from src.geometry.sdf import solid_from_sdf, q_from_sdf
from src.post.plotting import plot_velocity_slice
from src.post.statistics import block_statistics
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case
from src.run.runner import run_case

# command-line options (defaults reproduce run 74)
parser = argparse.ArgumentParser()
parser.add_argument("ride_height_ratio", nargs="?", type=float, default=0.3)  # h / c, lowest point of the wing above the ground
parser.add_argument("angle_degrees", nargs="?", type=float, default=4.0)  # incidence, positive = more downforce
parser.add_argument("--chord", type=int, default=80)  # c, cells (envelope: relative-accuracy check)
parser.add_argument("--reynolds", type=float, default=5000)  # Re on chord (envelope: Re sweep)
options = parser.parse_args()
ride_height_ratio = options.ride_height_ratio
angle_degrees = options.angle_degrees
section = "4412"
tag = f"h{ride_height_ratio:g}_a{angle_degrees:g}" + ("" if options.chord == 80 else f"_c{options.chord}") + ("" if options.reynolds == 5000 else f"_re{options.reynolds:g}")

# geometry
chord = options.chord  # c, cells
nx = 10 * chord
ny = 5 * chord
nz = 4  # periodic z: quasi-2D section
leading_edge_x = 3 * chord
ground_plane_y = 0.5  # halfway bounce-back plane in front of floor row 0
ride_height = ride_height_ratio * chord
reference_area = chord * nz  # c x span, for the coefficients

def main():
    """
    Place the wing at the requested ride height, run to a time-averaged downforce, log the run.
    """

    # wing: inverted section, lowest point at the ride height above the ground plane
    polygon_x, polygon_y = naca_four_digit(section)
    placed_x, placed_y = place_section(polygon_x, polygon_y, chord, angle_degrees, leading_edge_x, ground_plane_y + ride_height)
    phi = extruded_section_sdf(placed_x, placed_y)
    wing = Part(name="wing", solid=solid_from_sdf(phi, nx, ny, nz), reference_area=reference_area, wall_fractions=q_from_sdf(phi, nx, ny, nz))

    case = Case(name="wing_ground", tag=tag,
                flow=Flow(free_stream_velocity=0.05, reynolds_number=options.reynolds, reference_length=chord),
                domain=Domain(nx=nx, ny=ny, nz=nz, floor="moving", ceiling="moving", relax_width_x=round(0.3 * chord), relax_sigma=0.1),
                turbulence=Turbulence(sgs="wale", wale_constant=0.5, smagorinsky_constant=0.04),
                timing=Timing(ramp_flow_throughs=1.0, warmup_flow_throughs=4.0, total_flow_throughs=10.0, sample_every=25),
                parts=[wing])
    result = run_case(case, geometry=f"NACA {section} inverted, c={chord}, h/c={ride_height_ratio:g}, alpha={angle_degrees:g}",
                      walls="Bouzidi wing / staircase floor+ceiling",
                      boundaries="NEEM-open inlet (ramped) / pressure outlet / moving floor + ceiling at U / periodic z")
    sim = result.sim
    run = result.run
    solid = result.solid

    # time-averaged coefficients (part 0 = wing)
    downforce_coefficients = -result.force_coefficients[:, 0, 1]
    drag_coefficients = result.force_coefficients[:, 0, 0]
    np.save(f"results/wing_ground_cl_{tag}.npy", downforce_coefficients)
    mean_downforce, downforce_error = block_statistics(downforce_coefficients, 5)
    mean_drag, drag_error = block_statistics(drag_coefficients, 5)
    half = len(downforce_coefficients) // 2
    print(f"h/c = {ride_height_ratio:g}   -CL = {mean_downforce:.4f} +/- {downforce_error:.4f}   CD = {mean_drag:.4f} +/- {drag_error:.4f}")
    print(f"drift -CL: 1st half {downforce_coefficients[:half].mean():.4f}, 2nd half {downforce_coefficients[half:].mean():.4f}")

    # mid-span velocity around the wing
    velocity = sim.u.to_numpy()
    velocity[:, solid.astype(bool)] = np.nan
    fig, ax = plot_velocity_slice(velocity, axis=2, index=nz // 2, component=0)
    ax.set_xlim(leading_edge_x - chord, leading_edge_x + 3 * chord)
    ax.set_ylim(0, 2 * chord)
    fig.savefig(f"results/wing_ground_{tag}.png", dpi=150)

    run.finish(metric="-CL", value=round(float(mean_downforce), 4), se=round(float(downforce_error), 4),
               drift_1st=round(float(downforce_coefficients[:half].mean()), 4), drift_2nd=round(float(downforce_coefficients[half:].mean()), 4),
               other=f"CD {mean_drag:.4f} +/- {drag_error:.4f}")

if __name__ == "__main__":
    main()