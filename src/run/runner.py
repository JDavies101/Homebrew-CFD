# generic time loop: builds the simulation from a Case, runs it, returns the sampled part forces
from dataclasses import dataclass
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.sponge import relax_profile
from src.post.progress import Progress
from src.post.run_log import RunRecord

blow_up_population = 1e29 # f_absmax above this (or NaN / inf) stops the run

@dataclass
class RunResult:
    """
    What the example needs after the loop.
    """

    sim: Simulation3D
    run: RunRecord # open record: the example calls run.finish
    solid: np.ndarray # union of parts and wall rows
    forces: np.ndarray # (samples, parts, 3) raw part forces, float32
    force_coefficients: np.ndarray # (samples, parts, 3): F / (1/2 U^2 A) per part, float32
    blow_up_step: int # -1 when the run finished
    stopped_step: int # -1 unless stopped by progress_callback
    steps_completed: int # steps actually run (less than planned after a stop or blow-up)

def record_fields_from_case(case, steps, warmup):
    """
    Run-log columns that follow from the Case alone.

    Returns a dict for RunRecord.
    """

    domain = case.domain
    turbulence = case.turbulence
    fields = dict(steps=steps, u_ref=case.flow.free_stream_velocity, nu=case.flow.viscosity, tau=round(case.flow.relaxation_time, 6),
                  collision={"bgk": "BGK", "trt": "TRT", "regularized": "regularized"}[case.collision],
                  wall_model="log-law y+>30" if turbulence.wall_model else "off",
                  forcing="body force gx" if case.flow.body_force_x != 0.0 else "none")

    # subgrid model
    if turbulence.sgs == "none":
        fields["sgs"] = "none"
    elif turbulence.sgs == "smagorinsky":
        fields["sgs"] = f"smag cs={turbulence.smagorinsky_constant}"
    elif turbulence.smagorinsky_constant > 0.0:
        fields["sgs"] = f"wale cw={turbulence.wale_constant} + smag floor cs={turbulence.smagorinsky_constant}"
    else:
        fields["sgs"] = f"wale cw={turbulence.wale_constant}"

    # relaxation layers
    if domain.relax_width_z > 0:
        fields["sponge"] = (f"relax x {domain.relax_width_x} (free stream) / z {domain.relax_width_z} (running mean from ramp end, "
                            f"alpha {domain.relax_mean_rate:.1e}), sigma {domain.relax_sigma}")
    elif domain.relax_width_x > 0:
        fields["sponge"] = f"relax x {domain.relax_width_x}, sigma {domain.relax_sigma}"
    else:
        fields["sponge"] = "none"

    # Re and flow-through columns only where they mean something
    if case.flow.relaxation_time_override is None:
        fields["Re"] = case.flow.reynolds_number
    if domain.x_boundary == "inflow":
        fields["warmup"] = warmup
        fields["avg_Tft"] = round((steps - warmup) / case.flow_through_steps(), 1)
    if case.allow_below_floor:
        fields["notes"] = "tau below floor (allowed)"

    return fields

def report_blow_up(sim, time_step):
    """
    Print where f went non-finite, pulling f once.
    """

    populations = sim.f.to_numpy()
    bad_cells = np.argwhere(~np.isfinite(populations).all(axis=0))
    if len(bad_cells) == 0:
        print(f"Blow-up at step {time_step}: |f| above {blow_up_population:g}, no NaN yet")
        return
    print(f"Blow-up at step {time_step}: {len(bad_cells)} cells  "
          f"x[{bad_cells[:, 0].min()}-{bad_cells[:, 0].max()}] "
          f"y[{bad_cells[:, 1].min()}-{bad_cells[:, 1].max()}] "
          f"z[{bad_cells[:, 2].min()}-{bad_cells[:, 2].max()}]")

def run_case(case, backend="cuda", before_loop=None, after_collide=None, after_step=None, progress_callback=None, **record_fields):
    """
    Validate, build and run the case. Hooks: before_loop(sim) once after initialization;
    after_collide(sim, time_step) between collision and streaming; after_step(sim, time_step) at the end of
    each step, returning True stops the run. record_fields (geometry, walls, boundaries, ...) go to the run
    log and override the columns derived from the Case. progress_callback(time_step, steps, max_velocity,
    force_coefficients) runs at every health check (force_coefficients: the samples so far); returning
    True stops the run.

    Returns a RunResult.
    """

    case.validate()
    domain = case.domain
    flow = case.flow
    turbulence = case.turbulence
    nx = domain.nx
    ny = domain.ny
    nz = domain.nz
    free_stream_velocity = flow.free_stream_velocity
    bouzidi = any(part.wall_fractions is not None for part in case.parts)
    staircase = any(part.wall_fractions is None for part in case.parts)
    periodic = (domain.x_boundary == "periodic", domain.y_boundary == "periodic", domain.side_walls == "periodic")
    sim = Simulation3D(nx, ny, nz, backend, interp=bouzidi, periodic=periodic)

    # parts: part id = index + 1, union mask, floor and ceiling rows
    body = np.zeros((nx, ny, nz), np.int32)
    for part_id, part in enumerate(case.parts, start=1):
        body[part.solid == 1] = part_id
    solid = (body > 0).astype(np.int32)
    if domain.y_boundary == "walls":
        solid[:, 0, :] = 1
        solid[:, -1, :] = 1

    # wall velocity: moving floor / ceiling at U, rigid part surfaces
    wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
    if domain.y_boundary == "walls" and domain.floor == "moving":
        wall_velocity[0, :, 0, :] = free_stream_velocity
    if domain.y_boundary == "walls" and domain.ceiling == "moving":
        wall_velocity[0, :, -1, :] = free_stream_velocity
    for part in case.parts:
        if part.wall_velocity is not None:
            wall_velocity += part.wall_velocity

    # upload: body before wall fractions (link_part is read from body), solid and uw before the wall list
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(wall_velocity)
    sim.body.from_numpy(body)
    if bouzidi:
        wall_fractions = np.zeros((19, nx, ny, nz), np.float32)
        link_parts = np.zeros((19, nx, ny, nz), np.int8)
        for part_id, part in enumerate(case.parts, start=1):
            wall_fractions = np.maximum(wall_fractions, part.wall_fractions)
            link_parts[part.wall_fractions > 0.0] = part_id
        sim.set_wall_fractions(wall_fractions, link_parts)
    if turbulence.wall_model:
        sim.build_wall_list()
    fluid = solid == 0

    # initial field
    zero = np.zeros((nx, ny, nz), np.float32)
    if case.start == "rest_ramp":
        sim.init_equilibrium(zero, zero, zero)
    elif case.start == "uniform":
        sim.init_equilibrium(np.full((nx, ny, nz), free_stream_velocity, np.float32), zero, zero)
    else:
        sim.init_equilibrium(case.initial_velocity[0], case.initial_velocity[1], case.initial_velocity[2])

    # relaxation layers (sigma stays 0 when both widths are 0)
    if domain.relax_width_x > 0:
        sim.sigma.from_numpy(relax_profile(nx, nz, domain.relax_width_x, 0, domain.relax_sigma))
    if domain.relax_width_z > 0:
        sim.sigma_z.from_numpy(relax_profile(nx, nz, 0, domain.relax_width_z, domain.relax_sigma))

    # step counts, coefficient scales, per-step constants
    steps = case.total_steps()
    warmup = case.warmup_steps()
    ramp = case.ramp_steps()
    sample_every = case.timing.sample_every
    check_every = case.timing.check_every if case.timing.check_every is not None else max(1, steps // 300)
    ramped = case.start == "rest_ramp" and case.inlet != "none"
    sampling = case.timing.sample_window == "series" and len(case.parts) > 0
    dynamic_pressure_areas = [0.5 * free_stream_velocity * free_stream_velocity * part.reference_area for part in case.parts]
    relaxation_time = flow.relaxation_time
    smagorinsky_constant = turbulence.smagorinsky_constant if turbulence.sgs != "none" else 0.0
    trt = 1 if case.collision == "trt" else 0
    forces = []
    force_coefficients = []
    blow_up_step = -1
    stopped_step = -1

    if before_loop is not None:
        before_loop(sim)
    record = record_fields_from_case(case, steps, warmup)
    record.update(record_fields)
    run = RunRecord(case.name, sim, **record)

    progress = Progress(steps)
    for time_step in range(steps):
        # cosine inlet ramp from rest; moving walls ramp with it
        inlet_velocity = free_stream_velocity if case.inlet != "none" else 0.0
        wall_speed_scale = 1.0
        if ramped:
            ramp_fraction = min(time_step / ramp, 1.0)
            inlet_velocity = free_stream_velocity * 0.5 * (1.0 - np.cos(np.pi * ramp_fraction))
            wall_speed_scale = inlet_velocity / free_stream_velocity

        # eddy viscosities (WALE needs the current u)
        if turbulence.sgs == "wale":
            sim.macroscopic()
            sim.les_wale(turbulence.wale_constant, wall_speed_scale)
        if turbulence.wall_model:
            sim.wall_model_fast(flow.viscosity, wall_speed_scale)

        # z layers start from the flow at the end of the ramp
        if domain.relax_width_z > 0 and time_step == ramp:
            sim.macroscopic()
            sim.rho_bar.copy_from(sim.rho)
            sim.u_bar.copy_from(sim.u)
        z_layers_on = 1 if domain.relax_width_z > 0 and time_step >= ramp else 0

        # collision, separate x layers
        if case.collision == "regularized":
            sim.collide_reg(relaxation_time, smagorinsky_constant, flow.body_force_x, inlet_velocity, domain.relax_mean_rate, z_layers_on)
        else:
            sim.collide_full(relaxation_time, smagorinsky_constant, flow.body_force_x, trt)
        if domain.layer_kind == "separate" and domain.relax_width_x > 0:
            sim.sponge_relax(inlet_velocity)

        # staircase part forces are read after collision, before streaming
        sample_step = sampling and time_step >= warmup and time_step % sample_every == 0
        if sample_step and staircase:
            sim.drag_body()
        if after_collide is not None:
            after_collide(sim, time_step)

        # streaming and boundaries
        if bouzidi:
            sim.fc.copy_from(sim.f) # post-collision snapshot for Bouzidi and drag_interp
        sim.stream()
        if case.inlet == "neem_open":
            sim.inlet_neem_open(inlet_velocity) # drives open rows, skips solid rows at the inlet
        elif case.inlet == "neem":
            sim.inlet_neem(inlet_velocity)
        if domain.x_boundary == "inflow":
            sim.outlet_pressure(1.0)
        if domain.y_boundary == "free_slip":
            sim.free_slip_y()
        if domain.side_walls == "free_slip":
            sim.free_slip_z()
        if domain.y_boundary == "walls" or staircase:
            sim.bounce_back(wall_speed_scale) # wall rows and part nodes (Bouzidi overwrites its links below)
        if bouzidi:
            sim.bounce_back_interp(wall_speed_scale)
            if sample_step:
                sim.drag_interp(wall_speed_scale)

        # per-part forces; divide each float32 row by a python float (keeps float32)
        if sample_step:
            part_force = sim.part_force.to_numpy()
            forces.append([part_force[part_id] for part_id in range(1, len(case.parts) + 1)])
            force_coefficients.append([part_force[part_id] / dynamic_pressure_areas[part_id - 1] for part_id in range(1, len(case.parts) + 1)])

        if after_step is not None and after_step(sim, time_step):
            break

        # health: cheap GPU reduction first, then the progress readout
        if time_step % check_every == 0:
            max_population = sim.f_absmax()
            if not np.isfinite(max_population) or max_population > blow_up_population:
                blow_up_step = time_step
                break
            sim.macroscopic()
            max_velocity = float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid])))
            progress.update(time_step, max_velocity)
            if progress_callback is not None and progress_callback(time_step, steps, max_velocity, force_coefficients):
                stopped_step = time_step
                break
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))

    # steps actually run: the record's MLUPS and the progress bar use this, not the plan
    early_stop_step = max(blow_up_step, stopped_step)
    steps_completed = early_stop_step + 1 if early_stop_step >= 0 else steps
    progress.done(steps_completed)
    run.stop()
    run.steps = steps_completed

    if blow_up_step >= 0:
        report_blow_up(sim, blow_up_step)
    sim.macroscopic()

    return RunResult(sim, run, solid, np.asarray(forces), np.asarray(force_coefficients), blow_up_step, stopped_step, steps_completed)