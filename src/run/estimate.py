# pre-run estimate: device memory from the solver's field layout, run time from a measured throughput
from src.engine import lattice_d3q19 as d3q19
from src.run.case_file import part_free_case

field_bytes = 4 # every solver field is f32 or i32
max_parts = 8 # Simulation3D default: part_force has max_parts + 1 rows
default_throughput = {"cuda": 600.0, "cpu": 20.0} # MLUPS until this machine has measured its own (run 170: 644 on cuda)

def device_bytes(nx, ny, nz, bouzidi):
    """
    Bytes Simulation3D allocates in __init__ (the fields fields_gb counts), field by field.

    Returns the byte count.
    """

    direction_count = d3q19.direction_count
    dimension = d3q19.dimension
    cells = nx * ny * nz

    # per cell: f, f_new, rho, u, solid, lid, uw, nut_wall, nut_les, rho_bar, u_bar, body
    per_cell = 2 * direction_count + 1 + dimension + 1 + 1 + dimension + 1 + 1 + 1 + dimension + 1
    if bouzidi:
        per_cell += 2 * direction_count # q, fc

    # per (x, z) column: sigma, sigma_z; constants: force, part_force, c_q, w_q, opposite, mirror_y, mirror_z
    per_column = 2
    constants = dimension + (max_parts + 1) * dimension + direction_count * dimension + 4 * direction_count

    return field_bytes * (per_cell * cells + per_column * nx * nz + constants)

def case_device_bytes(case_file):
    """
    Device memory for a case file: Bouzidi fields when any part has Bouzidi walls.

    Returns the byte count.
    """

    domain = case_file.domain
    bouzidi = any(spec.wall == "bouzidi" for spec in case_file.geometry)

    return device_bytes(domain.nx, domain.ny, domain.nz, bouzidi)

def run_seconds(case_file, throughput_mlups):
    """
    Time stepping the whole run at a given throughput (start-up and geometry not included).

    Returns seconds.
    """

    domain = case_file.domain
    cells = domain.nx * domain.ny * domain.nz

    return cells * part_free_case(case_file).total_steps() / (throughput_mlups * 1e6)

def format_duration(seconds):
    """
    A run time as people read it: seconds under a minute, minutes under an hour, else hours and minutes.

    Returns the string.
    """

    if seconds < 60:
        return f"{seconds:.0f} s"

    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"

    return f"{minutes // 60} h {minutes % 60} min"