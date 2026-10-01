# shear-wave decay on a periodic box: measured vs analytic viscosity and stability as tau -> 1/2 (envelope E2)
# usage: python -m src.examples.shear_wave [bgk|trt|reg] [relaxation_time] [ny] [noise_level]
import sys
import numpy as np
from src.run.case import Flow, Domain, Turbulence, Timing, Case
from src.run.runner import run_case

# command-line options
collision_model = sys.argv[1] if len(sys.argv) > 1 else "bgk"  # "bgk", "trt" or "reg"
relaxation_time = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
ny = int(sys.argv[3]) if len(sys.argv) > 3 else 32  # points per wavelength
noise_level = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0  # 0 = pure wave, 1 = seed of 1e-4 U0

# geometry and flow
nx = 4
nz = 4
wave_amplitude = 0.05  # U0, low Mach
viscosity = (relaxation_time - 0.5) / 3.0  # analytic nu = c_s^2 (tau - 1/2)
wavenumber = 2 * np.pi / ny
decay_rate = viscosity * wavenumber * wavenumber  # nu k^2

# timing: ~one e-folding of the wave, capped
steps = min(round(1.0 / decay_rate), 400000)
sample_every = max(1, steps // 200)

def mode_and_residual(velocity, sine):
    """
    Amplitude of the sin(k y) mode of u_x, and the rms of everything else in the velocity field.

    Returns (amplitude, residual_rms).
    """

    profile = velocity[0].mean(axis=(0, 2))  # <u_x>(y)
    amplitude = 2.0 / ny * np.sum(profile * sine)
    residual = velocity.copy()
    residual[0] -= amplitude * sine[None, :, None]

    return amplitude, float(np.sqrt(np.mean(residual * residual)))

def main():
    """
    Decay the shear wave, fit the viscosity, report accuracy and stability, log the run.
    """

    # initial field: the wave plus a small broadband seed for the unstable (ghost) modes
    rng = np.random.default_rng(0)
    y = np.arange(ny)
    sine = np.sin(wavenumber * y)
    seed = 1e-4 * wave_amplitude * noise_level
    velocity_x = (wave_amplitude * sine[None, :, None] + seed * rng.standard_normal((nx, ny, nz))).astype(np.float32)
    velocity_y = (seed * rng.standard_normal((nx, ny, nz))).astype(np.float32)
    velocity_z = (seed * rng.standard_normal((nx, ny, nz))).astype(np.float32)

    collision = {"bgk": "bgk", "trt": "trt", "reg": "regularized"}[collision_model]
    case = Case(name="shear_wave", tag=f"{collision_model}_{relaxation_time:g}",
                flow=Flow(free_stream_velocity=wave_amplitude, relaxation_time_override=relaxation_time),
                domain=Domain(nx=nx, ny=ny, nz=nz, x_boundary="periodic", y_boundary="periodic", relax_width_x=0,
                              layer_kind="fused" if collision == "regularized" else "separate"),
                turbulence=Turbulence(sgs="none", smagorinsky_constant=0.0),
                timing=Timing(steps_override=steps, warmup_override=0, sample_window="none"),
                collision=collision, inlet="none", start="custom", initial_velocity=np.stack([velocity_x, velocity_y, velocity_z]),
                allow_below_floor=True)  # the stability map probes tau down to 0.5 on purpose

    # decay samples; step n = time_step + 1 is the state after n collide + stream steps
    sample_times = [0]
    amplitudes = []
    start_residual = 0.0
    residual = 0.0
    blow_up_step = -1

    def sample_start(sim):
        """
        Mode amplitude and residual of the initial field.
        """

        nonlocal start_residual, residual
        sim.macroscopic()
        start_amplitude, start_residual = mode_and_residual(sim.u.to_numpy(), sine)
        amplitudes.append(start_amplitude)
        residual = start_residual

    def sample_decay(sim, time_step):
        """
        Every sample_every steps: stop on blow-up (|f| > 10), else record the mode amplitude.

        Returns True to stop the run.
        """

        nonlocal residual, blow_up_step
        step_count = time_step + 1
        if step_count % sample_every != 0:
            return False
        max_population = sim.f_absmax()
        if not np.isfinite(max_population) or max_population > 10.0:
            blow_up_step = step_count
            return True
        sim.macroscopic()
        amplitude, residual = mode_and_residual(sim.u.to_numpy(), sine)
        sample_times.append(step_count)
        amplitudes.append(amplitude)

        return False

    result = run_case(case, before_loop=sample_start, after_step=sample_decay, tau=relaxation_time, collision=collision_model,
                      geometry=f"{nx}x{ny}x{nz} periodic, k=2pi/{ny}, noise {noise_level:g}", walls="none", boundaries="fully periodic")
    run = result.run
    if result.blow_up_step >= 0:
        blow_up_step = result.blow_up_step + 1

    if blow_up_step > 0:
        print(f"{collision_model} tau={relaxation_time}: blow-up at step {blow_up_step}")
        run.finish(metric="nu error %", value="blow-up", status="bad", reason=f"blow-up at step {blow_up_step}",
                   other=f"steps planned {steps}")
        return

    # viscosity from the decay slope: log A = log A0 - nu k^2 t
    slope = np.polyfit(np.array(sample_times, np.float64), np.log(np.array(amplitudes, np.float64)), 1)[0]
    measured_viscosity = -slope / (wavenumber * wavenumber)
    viscosity_error = 100.0 * (measured_viscosity / viscosity - 1.0)
    stable = residual <= start_residual  # non-wave content decayed, not grew
    decay_fraction = amplitudes[-1] / amplitudes[0]

    print(f"{collision_model} tau={relaxation_time}: nu {measured_viscosity:.4e} vs {viscosity:.4e} ({viscosity_error:+.2f}%), "
          f"A/A0 {decay_fraction:.3f}, residual {start_residual:.2e} -> {residual:.2e} ({'stable' if stable else 'GROWING'})")
    run.finish(metric="nu error %", value=round(viscosity_error, 3), reference=0.0,
               other=f"nu {measured_viscosity:.4e} vs {viscosity:.4e}; A/A0 {decay_fraction:.3f}; residual {start_residual:.2e}->{residual:.2e} {'stable' if stable else 'growing'}")

if __name__ == "__main__":
    main()