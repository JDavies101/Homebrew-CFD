# Ahmed body in a wind tunnel: WALE + Smagorinsky floor, relaxation layers, static or moving ground
# usage: python -m src.examples.ahmed [slant_angle] [nose round|square] [sgs wale|smag] [ground static|moving] [--height H] [--no-wall-model]
import argparse
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.ahmed_body import ahmed_body
from src.geometry.sponge import relax_profile
from src.post.plotting import plot_mask_slice, plot_velocity_slice
from src.post.progress import Progress
from src.post.run_log import RunRecord
from src.post.statistics import block_statistics
from src.post.vtk import write_field

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
viscosity = free_stream_velocity * body_height / reynolds_number  # nu ~5.3e-5
relaxation_time = 3 * viscosity + 0.5  # tau ~0.50016
body_force_x = 0.0

# turbulence
smagorinsky_floor = 0.04  # Smagorinsky floor under WALE: damps grid-scale noise where WALE's nut ~ 0
smagorinsky_constant = 0.084 if sgs == "smag" else smagorinsky_floor  # c_s
wale_constant = 0.5  # c_w

# relaxation layers, scaled with the body so every resolution sees the same physical layers (24 / 12 cells, 1/2000 at H = 32)
relax_width_x = round(0.75 * body_height)  # x layers (inlet/outlet), cells
relax_width_z = round(0.375 * body_height)  # z-wall layers, cells
relax_sigma = 0.1
relax_mean_rate = 1.0 / (62.5 * body_height)  # z-layer running-mean rate, memory ~ one body-height transit x 62.5

# timing
flow_through_steps = nx / free_stream_velocity  # T_ft (~14000 at H=32)
warmup = round(5 * flow_through_steps)  # establish flow + wake
steps = round(11 * flow_through_steps)  # + ~6 flow-throughs (~30 shedding periods) to average
ramp = round(flow_through_steps)
sample_every = 25  # sample force every N steps (avoids per-step GPU sync)
mean_every = 500  # mean-velocity snapshot interval
check_every = max(1, steps // 300)  # progress readout interval

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

    sim = Simulation3D(nx, ny, nz, backend="cuda")

    # geometry
    body = ahmed_body(nx, ny, nz, body_x_start, body_height, slant_angle, nose)
    plot_mask_slice(body, axis=2, index=nz // 2)[0].savefig("results/ahmed_xy.png")  # side view
    plot_mask_slice(body, axis=1, index=10)[0].savefig("results/ahmed_xz.png")  # plan view

    # floor & ceiling (no-slip)
    solid = body.copy()
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)
    if ground == "moving":
        wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
        wall_velocity[0, :, 0, :] = free_stream_velocity  # rolling road: floor row moves at the free-stream speed
        sim.uw.from_numpy(wall_velocity)
    sim.body.from_numpy(body)
    sim.build_wall_list()
    fluid = solid == 0

    # start from rest, inlet ramps up
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(zero, zero, zero)

    sim.sigma.from_numpy(relax_profile(nx, nz, relax_width_x, 0, relax_sigma))  # x layers only, free-stream target
    sim.sigma_z.from_numpy(relax_profile(nx, nz, 0, relax_width_z, relax_sigma))  # z layers only, running-mean target

    velocity_sum = np.zeros((3, nx, ny, nz), np.float32)
    velocity_sample_count = 0
    drag_coefficients = []
    run = RunRecord("ahmed", sim, steps=steps, u_ref=free_stream_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"H={body_height} phi={slant_angle} {nose}", warmup=warmup, avg_Tft=round((steps - warmup) / flow_through_steps, 1),
                    collision="regularized", sgs=f"smag cs={smagorinsky_constant}" if sgs == "smag" else f"wale cw={wale_constant} + smag floor cs={smagorinsky_constant}",
                    wall_model="off" if options.no_wall_model else "log-law y+>30", walls="staircase BB", forcing="none",
                    boundaries=f"NEEM-open inlet / pressure outlet / free-slip z / no-slip floor+ceiling / {ground} floor",
                    sponge=f"relax x {relax_width_x} (free stream) / z {relax_width_z} (running mean from ramp end, alpha {relax_mean_rate:.1e}), sigma {relax_sigma}")

    progress = Progress(steps)
    for time_step in range(steps):
        # cosine inlet ramp from rest
        ramp_fraction = min(time_step / ramp, 1.0)
        inlet_velocity = free_stream_velocity * 0.5 * (1.0 - np.cos(np.pi * ramp_fraction))
        wall_speed_scale = inlet_velocity / free_stream_velocity

        # eddy viscosities (WALE needs current u, not needed on the smag path)
        if sgs == "wale":
            sim.macroscopic()
            sim.les_wale(wale_constant, wall_speed_scale)  # moving walls enter the gradients at their ramped speed
        if not options.no_wall_model:
            sim.wall_model_fast(viscosity, wall_speed_scale)

        # z layers start from the flow at the end of the ramp
        if time_step == ramp:
            sim.macroscopic()
            sim.rho_bar.copy_from(sim.rho)
            sim.u_bar.copy_from(sim.u)
        z_layers_on = 1 if time_step >= ramp else 0
        sim.collide_reg(relaxation_time, smagorinsky_constant, body_force_x, inlet_velocity, relax_mean_rate, z_layers_on)

        # samples
        if time_step >= warmup and time_step % mean_every == 0:
            sim.macroscopic()
            velocity_sum += sim.u.to_numpy()
            velocity_sample_count += 1
        if time_step >= warmup and time_step % sample_every == 0:
            sim.drag_body()
            drag_coefficients.append(float(sim.force.to_numpy()[0]) / (0.5 * free_stream_velocity * free_stream_velocity * frontal_area))

        sim.stream()
        sim.inlet_neem_open(inlet_velocity)  # drives open rows, skips the solid floor at the inlet
        sim.outlet_pressure(1.0)
        sim.free_slip_z()  # side walls
        sim.bounce_back(wall_speed_scale)  # floor + ceiling + body; moving floor ramps with the inlet

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))  # fluid cells only

    progress.done()  # finish the bar (newline) before any other output
    run.stop()
    sim.macroscopic()

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
    drag_coefficients = np.asarray(drag_coefficients)
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
