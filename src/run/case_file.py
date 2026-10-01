# case files: the JSON description of a run (geometry by reference, settings by value) and its build into a Case
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf
from src.geometry.mesh import read_stl
from src.geometry.mesh_distance import sdf_from_mesh, q_from_mesh
from src.geometry.sdf import solid_from_sdf, q_from_sdf
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case, check_choice

schema_version = 1 # bump when a field is renamed or removed; old files then need a migration

geometry_choices = {"kind": ("stl", "naca"), "wall": ("bouzidi", "staircase")}

@dataclass
class GeometrySpec:
    """
    One part as the file stores it: where the shape comes from and how it sits on the grid.
    """

    name: str
    kind: str # "stl" or "naca"
    reference_area: float # A for the coefficients, cells^2
    wall: str = "bouzidi" # "bouzidi" or "staircase"
    path: str = "" # stl: file, relative to the case file
    cells_per_unit: float = 1.0 # stl: grid cells per STL length unit
    offset: list = field(default_factory=lambda: [0.0, 0.0, 0.0]) # stl: translation in cells, after scaling
    section: str = "4412" # naca: four-digit section
    chord: float = 80.0 # naca: c, cells
    angle_degrees: float = 4.0 # naca: incidence, positive = more downforce
    leading_edge: list = field(default_factory=lambda: [240.0, 24.5]) # naca: lowest-point placement (x, y), cells

@dataclass
class CaseFile:
    """
    Everything a case file holds; build_case turns it into a runnable Case.
    """

    name: str
    flow: Flow
    domain: Domain
    turbulence: Turbulence
    timing: Timing
    geometry: list[GeometrySpec] = field(default_factory=list)
    collision: str = "regularized"
    inlet: str = "neem_open"
    start: str = "rest_ramp" # "custom" is not available from a file (no initial field stored)
    allow_below_floor: bool = False

def save_case_file(case_file, path):
    """
    Write the case as indented JSON with its schema version.
    """

    data = {"schema_version": schema_version, **asdict(case_file)}
    Path(path).write_text(json.dumps(data, indent=2))

def load_case_file(path):
    """
    Read a case file; unknown keys raise (strict), missing keys take their defaults.

    Returns a CaseFile with STL paths made absolute.
    """

    path = Path(path)
    data = json.loads(path.read_text())
    version = data.pop("schema_version", None)
    if version != schema_version:
        raise ValueError(f"case file schema {version}, this build reads {schema_version}")
    
    geometry = [GeometrySpec(**spec) for spec in data.pop("geometry", [])]
    for spec in geometry:
        if spec.kind == "stl":
            spec.path = str((path.parent / spec.path).resolve())
    
    return CaseFile(flow=Flow(**data.pop("flow")), domain=Domain(**data.pop("domain")), turbulence=Turbulence(**data.pop("turbulence")),
                    timing=Timing(**data.pop("timing")), geometry=geometry, **data)

def build_part(spec, domain, backend):
    """
    Voxelize one geometry spec on the grid: solid mask, plus Bouzidi wall fractions when wall = "bouzidi".

    Returns a Part.
    """

    check_choice("geometry kind", spec.kind, geometry_choices["kind"])
    check_choice("geometry wall", spec.wall, geometry_choices["wall"])
    nx = domain.nx
    ny = domain.ny
    nz = domain.nz

    if spec.kind == "stl":
        triangles = read_stl(spec.path) * spec.cells_per_unit + np.asarray(spec.offset, np.float64)
        grid, phi = sdf_from_mesh(triangles, nx, ny, nz, backend=backend)
        solid = (grid < 0.0).astype(np.int32)
        wall_fractions = q_from_mesh(triangles, phi, nx, ny, nz, backend=backend) if spec.wall == "bouzidi" else None
    else:
        polygon_x, polygon_y = naca_four_digit(spec.section)
        placed_x, placed_y = place_section(polygon_x, polygon_y, spec.chord, spec.angle_degrees, spec.leading_edge[0], spec.leading_edge[1])
        phi = extruded_section_sdf(placed_x, placed_y)
        solid = solid_from_sdf(phi, nx, ny, nz)
        wall_fractions = q_from_sdf(phi, nx, ny, nz) if spec.wall == "bouzidi" else None

    return Part(name=spec.name, solid=solid, reference_area=spec.reference_area, wall_fractions=wall_fractions)

def build_case(case_file, backend="cuda"):
    """
    Build every part and assemble the runnable Case.

    Returns a Case (not yet validated; run_case validates).
    """

    parts = [build_part(spec, case_file.domain, backend) for spec in case_file.geometry]

    return Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                timing=case_file.timing, parts=parts, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                allow_below_floor=case_file.allow_below_floor)