# inverted NACA 4412 wing in ground effect (quasi-2D, periodic z): downforce vs ride height, moving ground
# usage: python -m src.examples.wing_ground [h_over_c] [angle_degrees] [--chord C]
import argparse
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf
from src.geometry.sdf import solid_from_sdf, q_from_sdf
from src.geometry.sponge import relax_profile
from src.post.plotting import plot_velocity_slice
from src.post.progress import Progress
from src.post.run_log import RunRecord
from src.post.statistics import block_statistics

# command-line options (defaults reproduce run 74)
parser = argparse.ArgumentParser()
parser.add_argument("ride_height_ratio", nargs="?", type=float, default=0.3)  # h / c, lowest point of the wing above the ground
parser.add_argument("angle_degrees", nargs="?", type=float, default=4.0)  # incidence, positive = more downforce
parser.add_argument("--chord", type=int, default=80)  # c, cells (envelope: relative-accuracy check)
options = parser.parse_args()
ride_height_ratio = options.ride_height_ratio
angle_degrees = options.angle_degrees
section = "4412"
tag = f"h{ride_height_ratio:g}_a{angle_degrees:g}" + ("" if options.chord == 80 else f"_c{options.chord}")

# geometry
chord = options.chord  # c, cells
nx = 10 * chord
ny = 5 * chord
nz = 4  # periodic z: quasi-2D section
leading_edge_x = 3 * chord
ground_plane_y = 0.5  # halfway bounce-back plane in front of floor row 0
ride_height = ride_height_ratio * chord
reference_area = chord * nz  # c x span, for the coefficients

# flow
free_stream_velocity = 0.05  # U
reynolds_number = 5000  # on chord
viscosity = free_stream_velocity * chord / reynolds_number
relaxation_time = 3 * viscosity + 0.5  # ~0.5024 -> regularized collision
smagorinsky_floor = 0.04
wale_constant = 0.5

# relaxation layers (x only)
relax_width_x = round(0.3 * chord)  # 24 cells at c = 80
relax_sigma = 0.1

# timing
flow_through_steps = nx / free_stream_velocity
warmup = round(4 * flow_through_steps)
steps = round(10 * flow_through_steps)
ramp = round(flow_through_steps)
sample_every = 25
check_every = max(1, steps // 300)

def main():
    """
    Place the wing at the requested ride height, run to a time-averaged downforce, log the run.
    """

    sim = Simulation3D(nx, ny, nz, "cuda", interp=True)

    # wing: inverted section, lowest point at the ride height above the ground plane
    polygon_x, polygon_y = naca_four_digit(section)
    placed_x, placed_y = place_section(polygon_x, polygon_y, chord, angle_degrees, leading_edge_x, ground_plane_y + ride_height)
    phi = extruded_section_sdf(placed_x, placed_y)
    wing = solid_from_sdf(phi, nx, ny, nz)

    # floor + ceiling rows, both moving with the free stream (moving ground; shear-free ceiling)
    solid = wing.copy()
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
    wall_velocity[0, :, 0, :] = free_stream_velocity
    wall_velocity[0, :, -1, :] = free_stream_velocity
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(wall_velocity)
    sim.body.from_numpy(wing)  # part 1 = wing
    sim.set_wall_fractions(q_from_sdf(phi, nx, ny, nz))  # after body: link_part is read from it
    fluid = solid == 0

    # start from rest, inlet ramps up
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(zero, zero, zero)
    sim.sigma.from_numpy(relax_profile(nx, nz, relax_width_x, 0, relax_sigma))

    downforce_coefficients = []
    drag_coefficients = []
    dynamic_pressure_area = 0.5 * free_stream_velocity * free_stream_velocity * reference_area
    run = RunRecord("wing_ground", sim, steps=steps, u_ref=free_stream_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"NACA {section} inverted, c={chord}, h/c={ride_height_ratio:g}, alpha={angle_degrees:g}", warmup=warmup,
                    avg_Tft=round((steps - warmup) / flow_through_steps, 1), collision="regularized",
                    sgs=f"wale cw={wale_constant} + smag floor cs={smagorinsky_floor}", wall_model="off", walls="Bouzidi wing / staircase floor+ceiling",
                    boundaries="NEEM-open inlet (ramped) / pressure outlet / moving floor + ceiling at U / periodic z", forcing="none",
                    sponge=f"relax x {relax_width_x}, sigma {relax_sigma}")

    progress = Progress(steps)
    for time_step in range(steps):
        # cosine inlet ramp from rest; moving walls ramp with it
        ramp_fraction = min(time_step / ramp, 1.0)
        inlet_velocity = free_stream_velocity * 0.5 * (1.0 - np.cos(np.pi * ramp_fraction))
        wall_speed_scale = inlet_velocity / free_stream_velocity

        sim.macroscopic()
        sim.les_wale(wale_constant, wall_speed_scale)
        sim.collide_reg(relaxation_time, smagorinsky_floor, 0.0, inlet_velocity, 0.0, 0)
        sim.fc.copy_from(sim.f)  # post-collision snapshot for Bouzidi and drag_interp
        sim.stream()
        sim.inlet_neem_open(inlet_velocity)
        sim.outlet_pressure(1.0)
        sim.bounce_back(wall_speed_scale)  # floor + ceiling (moving), wing nodes (overwritten by Bouzidi below)
        sim.bounce_back_interp(wall_speed_scale)

        # forces on part 1 (wing) only on sample steps
        if time_step >= warmup and time_step % sample_every == 0:
            sim.drag_interp(wall_speed_scale)
            wing_force = sim.part_force.to_numpy()[1]
            downforce_coefficients.append(-wing_force[1] / dynamic_pressure_area)
            drag_coefficients.append(wing_force[0] / dynamic_pressure_area)

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))

    progress.done()
    run.stop()
    sim.macroscopic()

    # time-averaged coefficients
    downforce_coefficients = np.asarray(downforce_coefficients)
    drag_coefficients = np.asarray(drag_coefficients)
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