# studies: the new-study wizard's answers turned into a 
# case file (sizing rules, physical to lattice units)
import math
from dataclasses import dataclass
from src.run.case import Flow, Domain, Turbulence, Timing, minimum_relaxation_time, check_choice
from src.run.case_file import CaseFile

fluid_viscosities = {"air": 1.516e-5, "water": 1.004e-6} # nu at 20 C, m^2/s
resolution_cells = {"coarse": 40, "medium": 64, "fine": 96} # cells across body length
study_choices = {
    "problem": ("free_stream", "ground_static", "ground_moving", "wheel"),
    "study": ("averaged", "unsteady"),
    "fluid": tuple(fluid_viscosities),
    "resolution": tuple(resolution_cells),
}

@dataclass
class StudyAnswers:
    """
    What the new-study wizard asks; every field has a default, so the defaults alone give a runnable case.
    """

    # required pages
    problem: str = "free_stream"
    study: str = "averaged"
    speed: float = 50.0 # m/s
    body_length: float = 0.5 # m
    fluid: str = "air"
    resolution: str = "medium"

    # more options
    name: str = "untitled"
    thin_span: bool = False # quasi-2D: nz = 4, periodic z
    lattice_velocity: float = 0.05 # U, lattice units
    reynolds_number: float | None = None # Re given directly, else from speed, length, and fluid
    length_multiple: float = 6.0 # domain length / body length
    height_multiple: float | None = None # None -> 3 free stream, 2 on the ground
    width_multiple: float = 3.0
    sgs: str = "wale"
    allow_below_floor: bool = False # keep the physical Re and run under the tau floor

def physical_reynolds_number(answers):
    """
    Re of the real problem.

    Returns U L / nu from speed, body length and fluid, or the Re given directly.
    """

    if answers.reynolds_number is not None:
        return answers.reynolds_number
    
    return answers.speed * answers.body_length / fluid_viscosities[answers.fluid]

def resolved_reynolds_number(answers, reference_length):
    """
    Re the case runs at: the physical Re, capped where tau reaches its floor unless allow_below_floor.

    Returns the Re for Flow.
    """

    physical = physical_reynolds_number(answers)
    if answers.allow_below_floor:
        return physical
    
    # nu at the floor: (tau_min - 1/2) / 3
    # floor() keeps the tau on or above the tau_min after rounding
    floor_viscosity = (minimum_relaxation_time - 0.5) / 3.0
    floor_reynolds_number = math.floor(answers.lattice_velocity * reference_length / floor_viscosity)

    return min(physical, floor_reynolds_number)

def case_from_answers(answers):
    """
    Size the domain from the body length in cells and set boundaries and timing from problem and study.

    Returns a CaseFile with no parts (geometry is added afterwards).
    """

    for name, choices in study_choices.items():
        check_choice(name, getattr(answers, name), choices)

    on_ground = answers.problem != "free_stream"
    reference_length = resolution_cells[answers.resolution]
    height_multiple = answers.height_multiple if answers.height_multiple is not None else (2.0 if on_ground else 3.0)

    # domain in body lengths
    nx = round(answers.length_multiple * reference_length)
    ny = round(height_multiple * reference_length)
    nz = 4 if answers.thin_span else round(answers.width_multiple * reference_length)
    side_walls = "periodic" if answers.thin_span else "free_slip"

    # ground: walls in y, the ceiling moves at U so it carries no boundary layer
    if on_ground:
        floor = "static" if answers.problem == "ground_static" else "moving"
        domain = Domain(nx=nx, ny=ny, nz=nz, y_boundary="walls", floor=floor, ceiling="moving", side_walls=side_walls)
    else:
        domain = Domain(nx=nx, ny=ny, nz=nz, y_boundary="free_slip", side_walls=side_walls)

    # unsteady: longer record and denser samples for the shedding frequency
    if answers.study == "unsteady":
        timing = Timing(total_flow_throughs=20.0, sample_every=10)
    else:
        timing = Timing()

    flow = Flow(free_stream_velocity=answers.lattice_velocity, reynolds_number=float(resolved_reynolds_number(answers, reference_length)),
                reference_length=float(reference_length))

    return CaseFile(name=answers.name, flow=flow, domain=domain, turbulence=Turbulence(sgs=answers.sgs), timing=timing,
                    allow_below_floor=answers.allow_below_floor)