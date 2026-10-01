# backward-facing step (Armaly): reattachment length x_r/S at Re=100 (reference ~3)
import numpy as np
from src.geometry.step_body import step
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case
from src.run.runner import run_case

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

    # step block and wall rows as one staircase part, at rest; the inlet ramps up
    case = Case(name="step", tag="",
                flow=Flow(free_stream_velocity=inlet_velocity, reynolds_number=reynolds_number, reference_length=2 * inlet_height),
                domain=Domain(nx=nx, ny=ny, nz=nz, relax_width_x=0, layer_kind="separate"),
                turbulence=Turbulence(sgs="none", smagorinsky_constant=0.0),
                timing=Timing(steps_override=steps, warmup_override=0, ramp_override=ramp, sample_window="none", check_every=check_every),
                parts=[Part(name="step", solid=step(nx, ny, nz, x_step, step_height), reference_area=1.0)],
                collision="trt", inlet="neem_open", start="rest_ramp")
    result = run_case(case, geometry=f"S={step_height} ER~1.94", walls="staircase BB",
                      boundaries="NEEM-open inlet (ramped) / pressure outlet / no-slip walls / periodic z")
    sim = result.sim
    run = result.run

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
