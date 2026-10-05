# case files: the JSON description of a run (geometry by reference, settings by value) and its build into a Case
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf
from src.geometry.mesh import read_stl
from src.geometry.mesh_distance import sdf_from_mesh, q_from_mesh
from src.geometry.sdf import solid_from_sdf, q_from_sdf, node_values
from src.geometry.ahmed_body import ahmed_body
from src.geometry.cylinder_body import cylinder, spin_wall_velocity
from src.geometry.sphere_body import sphere
from src.geometry.wall_fraction import wall_fraction_sphere, wall_fraction_cylinder
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case, check_choice

schema_version = 1 # bump when a field is renamed or removed; old files then need a migration

geometry_choices = {"kind": ("stl", "naca", "sphere", "cylinder", "ahmed"), "wall": ("bouzidi", "staircase"), "nose": ("round", "square")}

@dataclass
class GeometrySpec:
    """
    One part as the file stores it: where the shape comes from and how it sits on the grid. Fields unused by a kind are ignored.
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

    center: list = field(default_factory=lambda: [120.0, 64.0, 64.0]) # sphere: (x, y, z), cylinder: (x, y, unused); cells
    radius: float = 10.0 # sphere, cylinder: R, cells
    spin_ratio: float = 0.0 # cylinder: alpha = omega R / U, + = counter-clockwise about z
    x_start: int = 116 # ahmed: nose position, cells
    body_height: int = 32 # ahmed: H, cells (every other size follows from it)
    slant_angle: int = 25 # ahmed: rear slant, degrees
    nose: str = "round" # ahmed: "round" or "square"

def stl_spec(path, inspection, domain, name, cells_per_unit=None):
    """
    First placement of an imported STL: largest scale that fits a quarter of the domain length and half its height
    and width, front at x = nx / 4, centred in y and z; every value is a starting point to edit.
    cells_per_unit given: used as it is, no fit.

    Returns a GeometrySpec.
    """

    lower_corner = inspection["lower_corner"]
    size = np.maximum(inspection["size"], 1e-30)
    if cells_per_unit is None:
        fit_scale = min(domain.nx / 4 / size[0], domain.ny / 2 / size[1], domain.nz / 2 / size[2])
        cells_per_unit = float(f"{fit_scale:.3g}")

    placed_size = size * cells_per_unit

    # front at nx / 4 centered in y and z
    target_lower = np.array([domain.nx / 4, (domain.ny - placed_size[1]) / 2, 
                             (domain.nz - placed_size[2]) / 2])
    offset = target_lower - lower_corner * cells_per_unit
    reference_area = placed_size[1] * placed_size[2] # bounding box frontal area, cells^2

    return GeometrySpec(name=name, kind="stl", reference_area=float(round(reference_area, 1)), 
                        path=str(path), cells_per_unit=cells_per_unit, 
                        offset=[float(round(value, 2)) for value in offset])
 
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

def portable_path(file_path, base_directory):
    """
    A file path as a case file stores it: relative with forward slashes when the file sits inside the case's
    folder (so the folder can move), else absolute; a path already relative is relative to the case's folder
    and is kept as it is.

    Returns the string.
    """

    if not Path(file_path).is_absolute():
        return Path(file_path).as_posix()

    file_path = Path(file_path).resolve()
    base_directory = Path(base_directory).resolve()
    if file_path.is_relative_to(base_directory):
        return file_path.relative_to(base_directory).as_posix()
    
    return str(file_path)

def save_case_file(case_file, path):
    """
    Write the case as indented JSON with its schema version; STL paths inside the case's folder are stored relative.
    """

    path = Path(path)
    data = {"schema_version": schema_version, **asdict(case_file)}
    for spec in data["geometry"]:
        if spec["kind"] == "stl" and spec["path"]:
            spec["path"] = portable_path(spec["path"], path.parent)

    path.write_text(json.dumps(data, indent=2))

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

def part_free_case(case_file):
    """
    The Case of a case file without building geometry: for checks and derived numbers before a run.

    Returns a Case with no parts.
    """

    return Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                timing=case_file.timing, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                allow_below_floor=case_file.allow_below_floor)  

def build_part(spec, domain, backend, free_stream_velocity):
    """
    Voxelize one geometry spec on the grid: solid mask, plus Bouzidi wall fractions when wall = "bouzidi",
    plus the rigid surface velocity of a spinning cylinder (omega from the free stream U).

    Returns a Part.
    """

    check_choice("geometry kind", spec.kind, geometry_choices["kind"])
    check_choice("geometry wall", spec.wall, geometry_choices["wall"])
    nx = domain.nx
    ny = domain.ny
    nz = domain.nz
    wall_velocity = None # at rest unless a cylinder spins

    if spec.kind == "stl":
        triangles = read_stl(spec.path) * spec.cells_per_unit + np.asarray(spec.offset, np.float64)
        grid, phi = sdf_from_mesh(triangles, nx, ny, nz, backend=backend)
        solid = (grid < 0.0).astype(np.int32)
        wall_fractions = q_from_mesh(triangles, phi, nx, ny, nz, backend=backend) if spec.wall == "bouzidi" else None

    elif spec.kind == "naca":
        polygon_x, polygon_y = naca_four_digit(spec.section)
        placed_x, placed_y = place_section(polygon_x, polygon_y, spec.chord, spec.angle_degrees, spec.leading_edge[0], spec.leading_edge[1])
        phi = extruded_section_sdf(placed_x, placed_y)
        node_phi = node_values(phi, nx, ny, nz, extruded=True)
        solid = solid_from_sdf(phi, nx, ny, nz, node_phi)
        wall_fractions = q_from_sdf(phi, nx, ny, nz, node_phi=node_phi) if spec.wall == "bouzidi" else None

    elif spec.kind == "sphere":
        center_x, center_y, center_z = spec.center
        solid = sphere(nx, ny, nz, center_x, center_y, center_z, spec.radius)
        wall_fractions = wall_fraction_sphere(nx, ny, nz, center_x, center_y, center_z, spec.radius) if spec.wall == "bouzidi" else None
    
    elif spec.kind == "cylinder":
        center_x = spec.center[0]
        center_y = spec.center[1]
        solid = cylinder(nx, ny, nz, center_x, center_y, spec.radius)
        wall_fractions = wall_fraction_cylinder(nx, ny, nz, center_x, center_y, spec.radius) if spec.wall == "bouzidi" else None
        if spec.spin_ratio != 0.0:
            angular_velocity = spec.spin_ratio * free_stream_velocity / spec.radius # omega = alpha U / R
            wall_velocity = spin_wall_velocity(nx, ny, nz, center_x, center_y, spec.radius, angular_velocity)
            
    else:
        if spec.wall != "staircase":
            raise ValueError("ahmed: staircase walls only (the mask has no Bouzidi fractions yet)")
        
        check_choice("ahmed nose", spec.nose, geometry_choices["nose"])
        solid = ahmed_body(nx, ny, nz, spec.x_start, spec.body_height, spec.slant_angle, spec.nose)
        wall_fractions = None

    return Part(name=spec.name, solid=solid, reference_area=spec.reference_area,
                wall_fractions=wall_fractions, wall_velocity=wall_velocity)

def build_case(case_file, backend="cuda"):
    """
    Build every part and assemble the runnable Case.

    Returns a Case (not yet validated; run_case validates).
    """

    parts = [build_part(spec, case_file.domain, backend, case_file.flow.free_stream_velocity) for spec in case_file.geometry]

    return Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                timing=case_file.timing, parts=parts, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                allow_below_floor=case_file.allow_below_floor)