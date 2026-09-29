# backward-facing step (Armaly): reattachment length x_r/S at Re=100 (reference ~3)
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.step_body import step
from src.post.progress import Progress
from src.post.run_log import RunRecord

# geometry
step_height = 30  # S
ny = int(step_height / 0.485) + 2  # gives expansion ratio ~1.94
nx = 60 + 26 * step_height  # x_step = 60 (2S), plenty of downstream
nz = 4
x_step = 60
inlet_height = (ny - 2) - step_height  # h, inlet channel height

# flow
inlet_velocity = 0.1  # U
reynolds_number = 100
viscosity = inlet_velocity * (2 * inlet_height) / reynolds_number  # Armaly: Re on hydraulic diameter 2h
relaxation_time = 3 * viscosity + 0.5

# timing
steps = 60000
ramp = 8000  # cosine inlet ramp from rest (avoids trapping a start-up acoustic wave)
check_every = 500  # progress readout interval

def reattachment_length(floor_velocity):
    """
    Distance from the step to where the first-fluid-row x velocity turns positive again,
    by linear interpolation of the zero crossing (sub-cell).

    Returns x_r in cells, or -1 when there is no recirculation or no reattachment.
    """

    reversed_cells = np.where(floor_velocity < 0)[0]
    if len(reversed_cells) == 0:
        print("no recirculation")
        return -1

    bubble_start = reversed_cells[0]
    positive_after = np.where(floor_velocity[bubble_start:] > 0)[0]
    if len(positive_after) == 0:
        return -1

    first_positive = bubble_start + positive_after[0]
    last_reversed_value = floor_velocity[first_positive - 1]  # last non-positive sample (<= 0)
    first_positive_value = floor_velocity[first_positive]  # first positive sample (> 0)

    return (first_positive - 1) + (-last_reversed_value) / (first_positive_value - last_reversed_value)

def main():
    """
    Run the step to steady state and log x_r/S.
    """

    sim = Simulation3D(nx, ny, nz, backend="cuda")
    solid = step(nx, ny, nz, x_step, step_height)
    sim.solid.from_numpy(solid)
    fluid = solid == 0
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(zero, zero, zero)  # at rest; the inlet ramps up

    run = RunRecord("step", sim, steps=steps, u_ref=inlet_velocity, nu=viscosity, tau=round(relaxation_time, 6), Re=reynolds_number,
                    geometry=f"S={step_height} ER~1.94", collision="TRT", sgs="none", walls="staircase BB",
                    boundaries="NEEM-open inlet (ramped) / pressure outlet / no-slip walls / periodic z", forcing="none")
    progress = Progress(steps)
    for time_step in range(steps):
        sim.collide_trt(relaxation_time)
        sim.stream()
        ramp_fraction = min(time_step / ramp, 1.0)
        sim.inlet_neem_open(inlet_velocity * 0.5 * (1.0 - np.cos(np.pi * ramp_fraction)))
        sim.outlet_pressure(1.0)  # pins the mean density (zero-gradient outlet let it drift)
        sim.bounce_back()  # walls + step, all no-slip

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))  # fluid cells only

    progress.done()
    run.stop()
    sim.macroscopic()

    # reattachment on the first fluid row downstream of the step
    floor_velocity = sim.u.to_numpy()[0, x_step:, 1, nz // 2]
    x_reattachment = reattachment_length(floor_velocity)
    np.save("results/floor.npy", floor_velocity)

    # achieved inlet bulk velocity and Reynolds number
    velocity = sim.u.to_numpy()
    inlet_column = velocity[0, x_step - 5, step_height + 1 : ny - 1, :]  # open inlet rows, all z
    mean_inlet_velocity = float(inlet_column.mean())
    effective_reynolds_number = mean_inlet_velocity * 2 * inlet_height / viscosity

    if x_reattachment > 0:
        print(f"U_mean = {mean_inlet_velocity:.4f}   Armaly Re = {effective_reynolds_number:.0f}   x_r/S = {x_reattachment / step_height:.2f}")
    else:
        print(f"U_mean = {mean_inlet_velocity:.4f}   Armaly Re = {effective_reynolds_number:.0f}")
    run.finish(metric="x_r/S", value=round(x_reattachment / step_height, 3) if x_reattachment > 0 else "none", reference=3.0,
               other=f"Armaly Re {effective_reynolds_number:.0f}; U_mean {mean_inlet_velocity:.4f}")

if __name__ == "__main__":
    main()
