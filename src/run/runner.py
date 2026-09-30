# generic time loop: builds the simulation from a Case, runs it, returns the sampled part forces
from dataclasses import dataclass
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.geometry.sponge import relax_profile
from src.post.progress import Progress
from src.post.run_log import RunRecord

@dataclass
class RunResult:
    """
    What the example needs after the loop.
    """

    sim: Simulation3D
    run: RunRecord  # open record: the example calls run.finish
    solid: np.ndarray  # union of parts and wall rows
    force_coefficients: np.ndarray  # (samples, parts, 3): F / (1/2 U^2 A) per part

def check_ported(case):
    """
    Raise for settings Case accepts but the runner does not run yet; each port removes its lines.
    """

    unported = []
    if case.collision != "regularized":
        unported.append("collision")
    if case.inlet != "neem_open" or case.start != "rest_ramp":
        unported.append("inlet / start")
    if case.domain.x_boundary != "inflow" or case.domain.y_boundary != "walls" or case.domain.side_walls != "periodic":
        unported.append("boundaries")
    if case.domain.layer_kind != "fused" or case.domain.relax_width_z > 0:
        unported.append("layers")
    if case.turbulence.sgs != "wale" or case.turbulence.wall_model:
        unported.append("turbulence")
    if case.flow.body_force_x != 0.0 or case.timing.sample_window != "series":
        unported.append("forcing / sampling")
    if not case.parts or any(part.wall_fractions is None or part.wall_velocity is not None for part in case.parts):
        unported.append("parts")
    if unported:
        raise NotImplementedError(f"runner does not run yet: {', '.join(unported)}")

def run_case(case, **record_fields):
    """
    Validate, build and run the case; record_fields (geometry, walls, boundaries) go to the run log.

    Returns a RunResult.
    """

    case.validate()
    check_ported(case)
    domain = case.domain
    nx = domain.nx
    ny = domain.ny
    nz = domain.nz
    free_stream_velocity = case.flow.free_stream_velocity
    sim = Simulation3D(nx, ny, nz, "cuda", interp=True)

    # parts: part id = index + 1, union mask, floor and ceiling rows
    body = np.zeros((nx, ny, nz), np.int32)
    for part_id, part in enumerate(case.parts, start=1):
        body[part.solid == 1] = part_id
    solid = (body > 0).astype(np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1

    # moving floor / ceiling at U
    wall_velocity = np.zeros((3, nx, ny, nz), np.float32)
    if domain.floor == "moving":
        wall_velocity[0, :, 0, :] = free_stream_velocity
    if domain.ceiling == "moving":
        wall_velocity[0, :, -1, :] = free_stream_velocity

    # upload: body before wall fractions (link_part is read from body)
    wall_fractions = np.zeros((19, nx, ny, nz), np.float32)
    for part in case.parts:
        wall_fractions = np.maximum(wall_fractions, part.wall_fractions)
    sim.solid.from_numpy(solid)
    sim.uw.from_numpy(wall_velocity)
    sim.body.from_numpy(body)
    sim.set_wall_fractions(wall_fractions)
    fluid = solid == 0

    # start from rest, x relaxation layers
    zero = np.zeros((nx, ny, nz), np.float32)
    sim.init_equilibrium(zero, zero, zero)
    sim.sigma.from_numpy(relax_profile(nx, nz, domain.relax_width_x, 0, domain.relax_sigma))

    # step counts and coefficient scales
    steps = case.total_steps()
    warmup = case.warmup_steps()
    ramp = case.ramp_steps()
    sample_every = case.timing.sample_every
    check_every = case.timing.check_every if case.timing.check_every is not None else max(1, steps // 300)
    dynamic_pressure_areas = [0.5 * free_stream_velocity * free_stream_velocity * part.reference_area for part in case.parts]
    relaxation_time = case.flow.relaxation_time
    turbulence = case.turbulence
    force_coefficients = []

    run = RunRecord(case.name, sim, steps=steps, u_ref=free_stream_velocity, nu=case.flow.viscosity, tau=round(relaxation_time, 6),
                    Re=case.flow.reynolds_number, warmup=warmup, avg_Tft=round((steps - warmup) / case.flow_through_steps(), 1),
                    collision=case.collision, sgs=f"wale cw={turbulence.wale_constant} + smag floor cs={turbulence.smagorinsky_constant}",
                    wall_model="off", forcing="none", sponge=f"relax x {domain.relax_width_x}, sigma {domain.relax_sigma}",
                    notes="tau below floor (allowed)" if case.allow_below_floor else "", **record_fields)

    progress = Progress(steps)
    for time_step in range(steps):
        # cosine inlet ramp from rest; moving walls ramp with it
        ramp_fraction = min(time_step / ramp, 1.0)
        inlet_velocity = free_stream_velocity * 0.5 * (1.0 - np.cos(np.pi * ramp_fraction))
        wall_speed_scale = inlet_velocity / free_stream_velocity

        sim.macroscopic()
        sim.les_wale(turbulence.wale_constant, wall_speed_scale)
        sim.collide_reg(relaxation_time, turbulence.smagorinsky_constant, case.flow.body_force_x, inlet_velocity, domain.relax_mean_rate, 0)
        sim.fc.copy_from(sim.f)  # post-collision snapshot for Bouzidi and drag_interp
        sim.stream()
        sim.inlet_neem_open(inlet_velocity)
        sim.outlet_pressure(1.0)
        sim.bounce_back(wall_speed_scale)  # walls and part nodes (Bouzidi overwrites its links below)
        sim.bounce_back_interp(wall_speed_scale)

        # per-part coefficients on sample steps; divide each float32 row by a python float (keeps float32)
        if time_step >= warmup and time_step % sample_every == 0:
            sim.drag_interp(wall_speed_scale)
            part_force = sim.part_force.to_numpy()
            force_coefficients.append([part_force[part_id] / dynamic_pressure_areas[part_id - 1] for part_id in range(1, len(case.parts) + 1)])

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()[:, fluid]))))

    progress.done()
    run.stop()
    sim.macroscopic()

    return RunResult(sim, run, solid, np.asarray(force_coefficients))