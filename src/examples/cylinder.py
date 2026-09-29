# flow past a cylinder (Phase 2 gate): Re=100 -> Cd ~1.3-1.4, Strouhal ~0.16-0.20; optional spin (Magnus)
# usage: python -m src.examples.cylinder [spin_ratio]
import sys
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.cylinder_body import cylinder
from src.geometry.sponge import relax_profile
from src.geometry.wall_fraction import wall_fraction_cylinder
from src.post.progress import Progress
from src.post.run_log import RunRecord
from src.post.vtk import write_field

spin_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0  # alpha = omega R / U

# geometry
diameter = 45  # D, cells
nx = 1800
ny = 600
nz = 4
center_x = 450
center_y = ny // 2
frontal_area = diameter * nz

# flow
free_stream_velocity = 0.1  # U, keep < ~0.1 for low Mach
reynolds_number = 100
viscosity = free_stream_velocity * diameter / reynolds_number
relaxation_time = 3 * viscosity + 0.5  # ~0.59, safely above 0.5
angular_velocity = spin_ratio * free_stream_velocity / (diameter / 2)  # omega, rad per step, + = counter-clockwise about z

# timing
steps = 60000
warmup = 30000  # discard transient before averaging
sample_every = 20  # sample force every N steps (avoids per-step GPU sync)
check_every = 500  # progress readout interval

def main():
    """
    Run the cylinder, time-average Cd and Cl, get the Strouhal number from the lift spectrum.
    """

    sim = Simulation3D(nx, ny, nz, backend="cuda", interp=True)
    sim.solid.from_numpy(cylinder(nx, ny, nz, center_x, center_y, diameter / 2))
    sim.set_wall_fractions(wall_fraction_cylinder(nx, ny, nz, center_x, center_y, diameter / 2))
    fluid = sim.solid.to_numpy() == 0

    # start in uniform flow
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(np.full((nx, ny, nz), free_stream_velocity, np.float32), zero, zero)
    sim.sigma.from_numpy(relax_profile(nx, nz, 24, 0, 0.1))  # x absorbing layers

    # rigid rotation on a band covering both sides of the surface
    if spin_ratio != 0.0:
        X, Y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
        offset_x = X - center_x
        offset_y = Y - center_y
        band = np.sqrt(offset_x * offset_x + offset_y * offset_y) < diameter / 2 + 2.0
        wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
        wall_velocity[0][band] = (-angular_velocity * offset_y[band])[:, None]
        wall_velocity[1][band] = (angular_velocity * offset_x[band])[:, None]
        sim.uw.from_numpy(wall_velocity)

    drag_coefficients = []
    lift_force_history = []
    run = RunRecord("cylinder", sim, steps=steps, u_ref=free_stream_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"D={diameter}, alpha={spin_ratio}", warmup=warmup, collision="BGK", sgs="none", walls="Bouzidi",
                    boundaries="NEEM inlet / pressure outlet / free-slip y / periodic z", forcing="none",
                    sponge="relax x 24, sigma 0.1")
    progress = Progress(steps)
    for time_step in range(steps):
        sim.collide(relaxation_time)
        sim.sponge_relax(free_stream_velocity)
        sim.fc.copy_from(sim.f)  # snapshot post-collision before streaming
        sim.stream()
        sim.inlet_neem(free_stream_velocity)  # Guo non-equilibrium extrapolation
        sim.outlet_pressure(1.0)
        sim.free_slip_y()  # top/bottom free-slip
        sim.bounce_back_interp()  # Bouzidi on the cylinder

        # drag only on sample steps
        if time_step >= warmup and time_step % sample_every == 0:
            sim.drag_interp()
            force = sim.force.to_numpy()
            drag_coefficients.append(force[0] / (0.5 * 1.0 * free_stream_velocity * free_stream_velocity * frontal_area))
            lift_force_history.append(force[1])

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))  # fluid cells only

    progress.done()  # finish the bar (newline) before any other output
    run.stop()
    sim.macroscopic()
    write_field(f"results/cylinder_a{spin_ratio:g}", sim.rho.to_numpy(), sim.u.to_numpy())
    velocity = sim.u.to_numpy()
    effective_velocity = float(velocity[0, center_x, 20, nz // 2])  # same x as the cylinder, near the wall, out of the wake
    print(f"U_eff = {effective_velocity:.4f}  ->  effective Re = {effective_velocity * diameter / viscosity:.0f}")

    drag_coefficient = float(np.mean(drag_coefficients))
    lift_coefficient = float(np.mean(lift_force_history)) / (0.5 * 1.0 * free_stream_velocity * free_stream_velocity * frontal_area)  # Magnus: alpha > 0 (CCW) -> Cl < 0
    print(f"Cl (time-averaged) = {lift_coefficient:.3f}   [alpha = {spin_ratio}; Magnus sign for CCW spin: negative]")

    # Strouhal from the dominant frequency of the lift (F_y) signal
    lift_fluctuation = np.array(lift_force_history) - np.mean(lift_force_history)
    spectrum = np.abs(np.fft.rfft(lift_fluctuation))
    frequencies = np.fft.rfftfreq(len(lift_fluctuation), d=sample_every)
    shedding_frequency = frequencies[1 + np.argmax(spectrum[1:])]
    strouhal = shedding_frequency * diameter / free_stream_velocity

    print(f"Re = {reynolds_number}, D = {diameter}, tau = {relaxation_time:.3f}, blockage = {diameter / ny:.1%}")
    print(f"Cd (time-averaged) = {drag_coefficient:.3f}   [reference ~1.3-1.4]")
    print(f"Strouhal = {strouhal:.3f}        [reference ~0.16-0.20]")
    run.finish(metric="Cd", value=round(drag_coefficient, 4), reference=1.4, other=f"Cl {lift_coefficient:.3f}; St {strouhal:.3f}; U_eff {effective_velocity:.4f}")

if __name__ == "__main__":
    main()
