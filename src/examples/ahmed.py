# Ahmed body in a wind tunnel: WALE + Smagorinsky floor, relaxation layers, static or moving ground
# usage: python -m src.examples.ahmed [slant_angle] [nose round|square] [sgs wale|smag] [ground static|moving] [--height H] [--no-wall-model]
import argparse
import numpy as np
from src.geometry.ahmed_body import ahmed_body
from src.post.plotting import plot_mask_slice, plot_velocity_slice
from src.post.statistics import block_statistics
from src.post.vtk import write_field
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case
from src.run.runner import run_case

# command-line options (defaults reproduce run 73)
parser = argparse.ArgumentParser()
parser.add_argument("slant_angle", nargs="?", type=int, default=25)  # degrees
parser.add_argument("nose", nargs="?", default="round", choices=["round", "square"])
parser.add_argument("sgs", nargs="?", default="wale", choices=["wale", "smag"])
parser.add_argument("ground", nargs="?", default="static", choices=["static", "moving"])
parser.add_argument("--height", type=int, default=32)  # H, cells (envelope E6)
parser.add_argument("--no-wall-model", action="store_true")  # E6/E7: wall model off (nut_wall stays 0)
options = parser.parse_args()
slant_angle = options.slant_angle
nose = options.nose
sgs = options.sgs
ground = options.ground
tag = f"phi{slant_angle}" + ("" if nose == "round" else f"_{nose}") + ("" if sgs == "smag" else f"_{sgs}") + ("" if ground == "static" else "_mground") + ("" if options.height == 32 else f"_H{options.height}") + ("_nowm" if options.no_wall_model else "")

# geometry: tunnel sized from the body height
body_height = options.height  # small for iteration; go to 48 for real runs
body_length = round(1044 / 288 * body_height)
body_width = round(389 / 288 * body_height)
body_x_start = body_length  # 1 body-length upstream
nx = 6 * body_length  # upstream + body + 4-length wake
ny = round(3.75 * body_height)  # ~2.5 H of air above the body
nz = round(3.667 * body_height)  # ~10% blockage
frontal_area = body_width * body_height  # for Cd

# flow
free_stream_velocity = 0.05  # U, low Mach
reynolds_number = 30000  # Re_H: laminar-ish separated wake; machinery, not the reference Cd
viscosity = free_stream_velocity * body_height / reynolds_number  # nu ~5.3e-5, tau ~0.50016

# turbulence
smagorinsky_floor = 0.04  # Smagorinsky floor under WALE: damps grid-scale noise where WALE's nut ~ 0
smagorinsky_constant = 0.084 if sgs == "smag" else smagorinsky_floor  # c_s
wale_constant = 0.5  # c_w

# relaxation layers, scaled with the body so every resolution sees the same physical layers (24 / 12 cells, 1/2000 at H = 32)
relax_width_x = round(0.75 * body_height)  # x layers (inlet/outlet), cells
relax_width_z = round(0.375 * body_height)  # z-wall layers, cells
relax_sigma = 0.1
relax_mean_rate = 1.0 / (62.5 * body_height)  # z-layer running-mean rate, memory ~ one body-height transit x 62.5

# timing, in flow-throughs T_ft = nx / U (~14000 steps at H=32)
ramp_flow_throughs = 1.0
warmup_flow_throughs = 5.0  # establish flow + wake
total_flow_throughs = 11.0  # + ~6 flow-throughs (~30 shedding periods) to average
sample_every = 25  # sample force every N steps (avoids per-step GPU sync)
mean_every = 500  # mean-velocity snapshot interval

def slant_check(mid_velocity, mid_body):
    """
    Mean flow in the first fluid cell above the slant, mid-span. u_t = velocity along the slant
    (downstream-and-down); u_t > 0 means the mean flow follows the surface (attached), < 0 reversed.

    Returns (u_t / U per column, attached fraction).
    """

    slant_direction = np.array([np.cos(np.radians(slant_angle)), -np.sin(np.radians(slant_angle))])
    x_rear = body_x_start + body_length
    slant_length_x = round(round(222 / 288 * body_height) * np.cos(np.radians(slant_angle)))
    tangential_velocity = []
    for i in range(x_rear - slant_length_x, x_rear):
        j = np.nonzero(mid_body[i])[0].max() + 1  # first fluid cell above the surface
        tangential_velocity.append(mid_velocity[0, i, j] * slant_direction[0] + mid_velocity[1, i, j] * slant_direction[1])
    tangential_velocity = np.array(tangential_velocity)

    print(f"slant u_t/U (first fluid cell, {len(tangential_velocity)} columns):", np.round(tangential_velocity / free_stream_velocity, 3))
    attached = float(np.mean(tangential_velocity > 0.05 * free_stream_velocity))
    print(f"attached fraction (u_t > 0.05 U): {attached:.2f}")  # threshold: ~0 is dead water, not attached

    return tangential_velocity / free_stream_velocity, attached

def main():
    """
    Build the tunnel, run to a time-averaged Cd, save plots / fields, log the run.
    """

    # geometry
    body = ahmed_body(nx, ny, nz, body_x_start, body_height, slant_angle, nose)
    plot_mask_slice(body, axis=2, index=nz // 2)[0].savefig("results/ahmed_xy.png")  # side view
    plot_mask_slice(body, axis=1, index=10)[0].savefig("results/ahmed_xz.png")  # plan view

    case = Case(name="ahmed", tag=tag,
                flow=Flow(free_stream_velocity=free_stream_velocity, reynolds_number=reynolds_number, reference_length=body_height),
                domain=Domain(nx=nx, ny=ny, nz=nz, floor=ground, ceiling="static", side_walls="free_slip", relax_width_x=relax_width_x,
                              relax_width_z=relax_width_z, relax_sigma=relax_sigma, relax_mean_rate=relax_mean_rate),
                turbulence=Turbulence(sgs="wale" if sgs == "wale" else "smagorinsky", wale_constant=wale_constant,
                                      smagorinsky_constant=smagorinsky_constant, wall_model=not options.no_wall_model),
                timing=Timing(ramp_flow_throughs=ramp_flow_throughs, warmup_flow_throughs=warmup_flow_throughs,
                              total_flow_throughs=total_flow_throughs, sample_every=sample_every),
                parts=[Part(name="ahmed", solid=body, reference_area=frontal_area)],
                allow_below_floor=True)  # tau below the 0.501 floor: LES carries stability
    warmup = case.warmup_steps()

    # mean velocity snapshots, post-collision (same point in the step as before the port)
    velocity_sum = np.zeros((3, nx, ny, nz), np.float32)
    velocity_sample_count = 0

    def sample_mean_velocity(sim, time_step):
        """
        Add the current velocity to the running sum every mean_every steps after warmup.
        """

        nonlocal velocity_sum, velocity_sample_count
        if time_step >= warmup and time_step % mean_every == 0:
            sim.macroscopic()
            velocity_sum += sim.u.to_numpy()
            velocity_sample_count += 1

    result = run_case(case, after_collide=sample_mean_velocity, geometry=f"H={body_height} phi={slant_angle} {nose}", walls="staircase BB",
                      boundaries=f"NEEM-open inlet / pressure outlet / free-slip z / no-slip floor+ceiling / {ground} floor")
    sim = result.sim
    run = result.run

    # mean flow: mid-span slice, slant attachment, plots
    mean_velocity = velocity_sum / velocity_sample_count
    k_mid = nz // 2
    np.save(f"results/ahmed_umean_mid_{tag}.npy", mean_velocity[:, :, :, k_mid])  # mid-span mean, re-analysis
    slant_tangential, slant_attached = slant_check(mean_velocity[:, :, :, k_mid], body[:, :, k_mid])
    mean_velocity[:, body.astype(bool)] = np.nan  # hide the body interior
    fig, ax = plot_velocity_slice(mean_velocity, axis=2, index=nz // 2, component=0)
    fig.savefig(f"results/ahmed_mean_{tag}.png", dpi=130)
    ax.set_xlim(body_x_start - 20, body_x_start + 60)
    ax.set_ylim(0, 2 * body_height)
    fig.savefig(f"results/ahmed_mean_nose_{tag}.png", dpi=160)
    ax.set_xlim(body_x_start + body_length - 50, body_x_start + body_length + 80)
    ax.set_ylim(0, 2 * body_height)
    fig.savefig(f"results/ahmed_mean_slant_{tag}.png", dpi=160)

    # instantaneous field
    velocity = sim.u.to_numpy()
    write_field(f"results/ahmed_{tag}", sim.rho.to_numpy(), velocity)
    plot_velocity_slice(velocity, axis=2, index=nz // 2, component=0)[0].savefig(f"results/ahmed_wake_{tag}.png", dpi=130)

    # drag statistics
    drag_coefficients = result.forces[:, 0, 0].astype(np.float64) / (0.5 * free_stream_velocity * free_stream_velocity * frontal_area)  # float64, as before the port
    np.save(f"results/ahmed_cd_{tag}.npy", drag_coefficients)  # raw series for re-analysis
    for block_count in (5, 10, 20):
        mean, standard_error = block_statistics(drag_coefficients, block_count)
        print(f"Cd = {mean:.4f} +/- {standard_error:.4f}   ({block_count} blocks)")
    half = len(drag_coefficients) // 2
    print(f"drift: 1st half {drag_coefficients[:half].mean():.4f}, 2nd half {drag_coefficients[half:].mean():.4f}")
    print("wall-engaged nodes:", int((sim.nut_wall.to_numpy() > 0).sum()))

    if sgs == "wale":
        sim.macroscopic()
        sim.les_wale(wale_constant)
        eddy_viscosity = sim.nut_les.to_numpy()
        fluid_final = sim.solid.to_numpy() == 0
        print(f"nut_les/nu: mean {eddy_viscosity[fluid_final].mean() / viscosity:.2f}  max {eddy_viscosity[fluid_final].max() / viscosity:.2f}")

    mean_5_blocks, standard_error_5_blocks = block_statistics(drag_coefficients, 5)
    run.finish(metric="Cd", value=round(float(mean_5_blocks), 4), se=round(float(standard_error_5_blocks), 4),
               drift_1st=round(float(drag_coefficients[:half].mean()), 4), drift_2nd=round(float(drag_coefficients[half:].mean()), 4),
               other=f"slant attached {slant_attached:.2f}; slant u_t/U first 4 cols {np.round(slant_tangential[:4], 3).tolist()}")

if __name__ == "__main__":
    main()
