# turbulent plane channel Re_tau=180 -> log law (Moser-Kim-Mansour); DNS with WALE or Smagorinsky
# milestone: trip and *sustain* turbulence (U_c bounded near ~17.7 u_tau, RMS steady)
# at fixed forcing, relaminarizing means running away to the laminar state -> diverges
import numpy as np
from src.post.plotting import plot_law_of_wall
from src.run.case import Flow, Domain, Turbulence, Timing, Case
from src.run.runner import run_case
from src.turbulence.wall_function import friction_velocity

# flow in wall units
friction_reynolds_number = 180  # Re_tau
u_tau = 0.0045  # small -> U_c ~ 17.7 u_tau stays low Mach
half_height = 64  # delta, cells; first node sits 0.5 off the wall -> y+ ~1.4
viscosity = u_tau * half_height / friction_reynolds_number  # nu = 0.0016, tau ~0.505: near 0.5 -> TRT mandatory, BGK would blow up
body_force_x = u_tau ** 2 / half_height  # tau_w = rho g delta -> u_tau = sqrt(g delta)
centerline_velocity = 17.7 * u_tau  # U_c, turbulent centerline from the log law (~0.08)

# box
ny = 2 * half_height + 2  # solid rows j = 0, j = ny - 1; halfway walls -> H = ny - 2 = 2 delta
nx = round(337 * half_height / friction_reynolds_number)  # hold L_x+ ~ 337 as delta shrinks (coarsening sweep)
nz = round(169 * half_height / friction_reynolds_number)  # hold L_z+ ~ 169

# trip: rolls + streaks + noise
roll_pair_count = max(1, nz // 30)  # 2 pairs at Re_tau = 180 (nz = 60), 1 on the coarse box
spanwise_wavenumber = 2 * np.pi * roll_pair_count / nz  # beta
roll_amplitude = 0.10 * centerline_velocity  # u_y
streak_amplitude = 0.10 * centerline_velocity  # u_x
noise_amplitude = 0.05 * centerline_velocity  # broadband
rng = np.random.default_rng(0)  # reproducible initial condition across trip attempts

# collision and turbulence model
collision_operator = "full"  # "reg" (collide_reg, needed near tau = 0.5) or "full" (collide_full, the Re_tau = 180 DNS)
sgs = "wale"  # "wale" or "smag"
smagorinsky_constant = 0.084 if sgs == "smag" else 0.0  # 0.084 = the old cs = 0.1 that over-damped this case
wale_constant = 0.5

# timing
steps = round(600000 * half_height / 64)  # ~constant turnovers across the delta sweep
warmup = steps // 6  # discard ~10 turnovers of transient
check_every = max(1, round(steps / 300))  # ~300 readouts, independent of run length
turnover_steps = half_height / u_tau
sample_every = max(1, round(turnover_steps / 50))  # ~50 velocity snapshots per eddy turnover (each one is a GPU sync)

def add_rolls_and_streaks(velocity_x, velocity_y, velocity_z, eta, z):
    """
    Add the SSP seed in place: divergence-free streamwise rolls (u_y, u_z) plus streaks (u_x).
    eta = y / delta and z are broadcastable coordinate arrays.
    """

    stream_function_amplitude = roll_amplitude / spanwise_wavenumber
    wall_window = (1 - eta ** 2) ** 2
    velocity_y += -stream_function_amplitude * spanwise_wavenumber * wall_window * np.sin(spanwise_wavenumber * z)
    velocity_z += (stream_function_amplitude / half_height) * 4.0 * eta * (1 - eta ** 2) * np.cos(spanwise_wavenumber * z)
    velocity_x += streak_amplitude * wall_window * np.sin(spanwise_wavenumber * z)

def initial_velocity():
    """
    Mean parabola + rolls / streaks + noise.

    Returns velocity_x, velocity_y, velocity_z (float32, (nx, ny, nz)).
    """

    y = np.arange(ny) - (ny - 1) / 2.0
    eta = (y / half_height)[None, :, None]
    z = np.arange(nz)[None, None, :]

    velocity_x = np.broadcast_to(centerline_velocity * (1 - eta ** 2), (nx, ny, nz)).astype(np.float32).copy()
    velocity_y = np.zeros((nx, ny, nz), np.float32)
    velocity_z = np.zeros((nx, ny, nz), np.float32)

    add_rolls_and_streaks(velocity_x, velocity_y, velocity_z, eta, z)

    # noise is the only x-dependent term -> lets the x-invariant rolls break down to 3D
    velocity_x += (noise_amplitude * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)
    velocity_y += (noise_amplitude * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)
    velocity_z += (noise_amplitude * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)

    return velocity_x, velocity_y, velocity_z

def fold(profile):
    """
    Average the two symmetric halves of a wall-normal profile.

    Returns the folded profile (same length).
    """

    return 0.5 * (profile + profile[::-1])

def main():
    """
    Trip and sustain channel turbulence, average profiles, compare with the log law and MKM.
    """

    velocity_x, velocity_y, velocity_z = initial_velocity()
    case = Case(name="channel_turbulent", tag=f"{friction_reynolds_number}_{sgs}",
                flow=Flow(free_stream_velocity=u_tau, reynolds_number=friction_reynolds_number, reference_length=half_height, body_force_x=body_force_x),
                domain=Domain(nx=nx, ny=ny, nz=nz, x_boundary="periodic", relax_width_x=0, layer_kind="fused" if collision_operator == "reg" else "separate"),
                turbulence=Turbulence(sgs="wale" if sgs == "wale" else "smagorinsky", wale_constant=wale_constant, smagorinsky_constant=smagorinsky_constant),
                timing=Timing(steps_override=steps, warmup_override=warmup, sample_window="none", check_every=check_every),
                collision="regularized" if collision_operator == "reg" else "trt", inlet="none", start="custom",
                initial_velocity=np.stack([velocity_x, velocity_y, velocity_z]))

    # plane-averaged mean and mean-square profiles
    mean_x_samples = []
    square_x_samples = []
    square_y_samples = []
    square_z_samples = []

    def sample_profiles(sim, time_step):
        """
        Append plane-averaged profiles every sample_every steps after warmup.

        Returns False (never stops the run).
        """

        if time_step >= warmup and time_step % sample_every == 0:
            sim.macroscopic()
            velocity = sim.u.to_numpy()
            mean_x_samples.append(velocity[0].mean(axis=(0, 2)))
            square_x_samples.append((velocity[0] ** 2).mean(axis=(0, 2)))
            square_y_samples.append((velocity[1] ** 2).mean(axis=(0, 2)))
            square_z_samples.append((velocity[2] ** 2).mean(axis=(0, 2)))

        return False

    result = run_case(case, after_step=sample_profiles, u_ref=centerline_velocity, reference=18.3 if friction_reynolds_number == 180 else 20.8,
                      geometry=f"delta={half_height} (Re_tau {friction_reynolds_number})", walls="halfway BB",
                      boundaries="periodic x,z / no-slip y walls")
    sim = result.sim
    run = result.run

    # refresh WALE on the final field
    sim.les_wale(wale_constant)
    eddy_viscosity = sim.nut_les.to_numpy()
    fluid = sim.solid.to_numpy() == 0
    print(f"nut_les/nu: mean {eddy_viscosity[fluid].mean() / viscosity:.2f}  max {eddy_viscosity[fluid].max() / viscosity:.2f}")

    # mean and rms profiles (ny,)
    mean_velocity = np.mean(np.array(mean_x_samples), axis=0)  # mean profile -> the log law
    rms_x = np.sqrt(np.mean(square_x_samples, axis=0) - mean_velocity ** 2)  # variance about the space-time mean
    rms_y = np.sqrt(np.mean(square_y_samples, axis=0))
    rms_z = np.sqrt(np.mean(square_z_samples, axis=0))

    # friction velocity from the force balance (exact) and from the resolved wall shear
    effective_half_height = half_height + 0.0  # wall exactly halfway (the old +0.33 was wall nodes being collided)
    u_tau_force_balance = np.sqrt(body_force_x * effective_half_height)
    first_node_velocity = 0.5 * (mean_velocity[1] + mean_velocity[-2])  # mean u_x at the first fluid node off each wall
    first_node_distance = effective_half_height - (half_height - 0.5)  # its wall distance in cells (0.5)
    u_tau_wall_shear = np.sqrt(viscosity * first_node_velocity / first_node_distance)  # tau_w = nu du/dy
    y_from_center = np.arange(ny) - (ny - 1) / 2.0
    wall_distance = effective_half_height - np.abs(y_from_center)  # cells (< 0 on the solid rows)
    asymmetry = float(np.abs(mean_velocity - mean_velocity[::-1]).max() / mean_velocity.max())  # pre-fold top/bottom check

    # wall units, one half, fluid nodes only (drop the wall row)
    half_slice = slice(1, ny // 2)
    y_plus = (fold(wall_distance) * u_tau_force_balance / viscosity)[half_slice]
    u_plus = (fold(mean_velocity) / u_tau_force_balance)[half_slice]
    rms_x_plus = (fold(rms_x) / u_tau_force_balance)[half_slice]
    rms_y_plus = (fold(rms_y) / u_tau_force_balance)[half_slice]
    rms_z_plus = (fold(rms_z) / u_tau_force_balance)[half_slice]

    # wall function check at a node in the log layer
    log_layer_index = int(np.argmin(np.abs(y_plus - 40)))
    log_layer_velocity = u_plus[log_layer_index] * u_tau_force_balance  # dimensional velocity there
    log_layer_distance = y_plus[log_layer_index] * viscosity / u_tau_force_balance  # dimensional wall distance
    print(f"wall-fn u_tau = {friction_velocity(log_layer_velocity, log_layer_distance, viscosity):.5f}  vs force-balance {u_tau_force_balance:.5f}")

    rms_peak_index = int(rms_x_plus.argmax())
    print(f"centerline U+   = {u_plus[-1]:.2f}         [MKM ~18.3]")
    print(f"u'_rms peak     = {rms_x_plus[rms_peak_index]:.2f} at y+ = {y_plus[rms_peak_index]:.1f}   [MKM ~2.65 at y+~15]")
    print(f"first node y+   = {y_plus[0]:.2f}          [near-wall resolution]")
    print(f"wall-shear err  = {100 * (u_tau_wall_shear / u_tau_force_balance - 1):+.1f}%          [->0 resolved, grows coarse]")
    print(f"top/bottom asym = {asymmetry * 100:.1f}%          [convergence, want a few %]")

    fig, _ = plot_law_of_wall(y_plus, u_plus, rms_x_plus, rms_y_plus, rms_z_plus)
    fig.savefig(f"results/channel_loglaw_{friction_reynolds_number}_{sgs}.png", dpi=130)
    print(f"wrote results/channel_loglaw_{friction_reynolds_number}_{sgs}.png")

    run.finish(metric="U+ centerline", value=round(float(u_plus[-1]), 2),
               other=(f"u'rms {rms_x_plus[rms_peak_index]:.2f} @ y+{y_plus[rms_peak_index]:.1f}; "
                      f"wall-shear {100 * (u_tau_wall_shear / u_tau_force_balance - 1):+.1f}%; asym {asymmetry * 100:.1f}%"))

if __name__ == "__main__":
    main()
