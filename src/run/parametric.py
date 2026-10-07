# studies: solver and timing settings, optionally one swept parameter, one case file per run
import copy
import json
from dataclasses import dataclass, fields, asdict, replace
from pathlib import Path
import numpy as np
from src.run.case import Timing, check_choice
from src.run.case_file import save_case_file, load_case_file
from src.run.project import check_project_name
from src.run.run_folder import case_hash, write_json_atomic

study_schema_version = 2
study_types = {"steady": "Steady (averaged forces)"}

# parameter -> {geometry kind: (field, index into a list field or None)}
sweep_parameters = {"position_y": {"naca": ("leading_edge", 1), "stl": ("offset", 1), "sphere": ("center", 1), "cylinder": ("center", 1)},
                    "angle": {"naca": ("angle_degrees", None), "ahmed": ("slant_angle", None)}}
sweep_labels = {"position_y": "Position y (cells)", "angle": "Angle (degrees)"}

@dataclass
class StudySolver:
    """
    The solver choices a study sets for its runs (the case file keeps the same fields as the defaults a new study copies).
    """

    collision: str = "regularized"
    inlet: str = "neem_open"
    start: str = "rest_ramp"
    allow_below_floor: bool = False

def study_defaults(case_file):
    """
    The solver and timing settings a new study starts from: the case's own.

    Returns (StudySolver, Timing), both copies.
    """

    solver = StudySolver(case_file.collision, case_file.inlet, case_file.start, case_file.allow_below_floor)

    return solver, replace(case_file.timing)

def apply_study_settings(case_file, solver, timing):
    """
    A deep copy of the case with the study's solver and timing fields set; nothing else changes.

    Returns a CaseFile.
    """

    solver_names = [item.name for item in fields(StudySolver)]
    applied = copy.deepcopy(case_file)
    for name, value in solver.items():
        if name not in solver_names:
            raise ValueError(f"{name} is not a study solver setting")

        setattr(applied, name, value)

    applied.timing = replace(applied.timing, **timing)

    return applied

def check_study_name(name):
    """
    Raise if a study name cannot be a folder name everywhere (letters, digits, space, _ - . only; not empty).
    """

    try:
        check_project_name(name)
    except ValueError:
        raise ValueError(f"study name {name!r}: use letters, digits, space, _ - . only") from None

def default_study_name(studies_directory, taken):
    """
    The first name of the form Study N not used by a folder in the studies directory or by a name in taken.

    Returns the name.
    """

    number = 1
    while f"Study {number}" in taken or (Path(studies_directory) / f"Study {number}").exists():
        number += 1

    return f"Study {number}"

def sweep_choices(spec):
    """
    The parameters a part can be swept over, from its geometry kind.

    Returns a list of parameter names (empty when none apply).
    """

    return [parameter for parameter, kinds in sweep_parameters.items() if spec.kind in kinds]

def current_value(spec, parameter):
    """
    The part's present value of a sweep parameter.

    Returns the number.
    """

    field_name, index = sweep_parameters[parameter][spec.kind]
    value = getattr(spec, field_name)

    return value if index is None else value[index]

def make_variants(case_file, part_name, parameter, values, study_name):
    """
    One deep copy of the case per value, each with only the swept field changed (ahmed slant angle is an int).

    Returns a list of CaseFile named <study_name>_<index>.
    """

    spec = next((spec for spec in case_file.geometry if spec.name == part_name), None)
    if spec is None:
        raise ValueError(f"no part named {part_name}")

    if parameter not in sweep_choices(spec):
        raise ValueError(f"{parameter} cannot be swept on a {spec.kind} part")

    if len(values) == 0:
        raise ValueError("no values to sweep")

    field_name, index = sweep_parameters[parameter][spec.kind]
    variants = []

    for number, value in enumerate(values):
        variant = copy.deepcopy(case_file)
        variant.name = f"{study_name}_{number:02d}"
        variant_spec = next(spec for spec in variant.geometry if spec.name == part_name)
        if spec.kind == "ahmed":
            value = int(round(value))

        if index is None:
            setattr(variant_spec, field_name, value)
        else:
            getattr(variant_spec, field_name)[index] = value

        variants.append(variant)

    return variants

def values_from_range(start, end, count):
    """
    Evenly spaced values from start to end, both included.

    Returns a list of floats.
    """

    if count < 2:
        raise ValueError("a range needs at least 2 values")

    return [float(value) for value in np.linspace(start, end, count)]

def unique_study_name(studies_directory, base_name):
    """
    The base name if no study folder has it yet, else base_name_2, base_name_3 and so on.

    Returns the first name whose folder does not exist.
    """

    name = base_name
    number = 2
    while (Path(studies_directory) / name).exists():
        name = f"{base_name}_{number}"
        number += 1

    return name

def write_study(folder, base_case_file, study):
    """
    Create the study folder with study.json (schema 2) and one case file per swept value, or one file without a sweep.
    Every case file is the base case with the study's solver and timing applied (and the sweep variant when there is one).
    The study dict holds name, type, solver and timing as dicts, and sweep (None or part, parameter, values).

    Returns (study dict as written, list of (CaseFile, path)).
    """

    name = study["name"]
    check_study_name(name)
    check_choice("study type", study["type"], tuple(study_types))
    applied = apply_study_settings(base_case_file, study["solver"], study["timing"])
    sweep = study["sweep"]
    if sweep is None:
        applied.name = name
        variants = [applied]
    else:
        variants = make_variants(applied, sweep["part"], sweep["parameter"], sweep["values"], name)

    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"a study named {name} already exists in this project")

    folder.mkdir(parents=True, exist_ok=False)

    written = []
    for variant in variants:
        path = folder / f"{variant.name}.json"
        save_case_file(variant, path)
        written.append((variant, path))

    record_sweep = None
    if sweep is not None:
        record_sweep = {"part": sweep["part"], "parameter": sweep["parameter"], "values": [float(value) for value in sweep["values"]]}

    record = {"schema_version": study_schema_version, "name": name, "type": study["type"], "solver": dict(study["solver"]),
              "timing": dict(study["timing"]), "sweep": record_sweep, "base_case_hash": case_hash(base_case_file),
              "cases": [path.name for _, path in written]}
    write_json_atomic(folder / "study.json", record)

    return record, written

def load_study(folder):
    """
    Read study.json from a study folder. A schema 1 folder (a sweep only) is read as a steady study whose
    solver and timing come from its first case file.

    Returns the study dict in the schema 2 shape.
    """

    folder = Path(folder)
    record = json.loads((folder / "study.json").read_text())
    version = record.get("schema_version")
    if version == study_schema_version:
        return record

    if version != 1:
        raise ValueError(f"study schema {version}, this build reads 1 and 2")

    solver, timing = study_defaults(load_case_file(folder / record["cases"][0]))
    sweep = {"part": record["part"], "parameter": record["parameter"], "values": record["values"]}

    return {"schema_version": study_schema_version, "name": record["name"], "type": "steady", "solver": asdict(solver),
            "timing": asdict(timing), "sweep": sweep, "base_case_hash": record["base_case_hash"], "cases": record["cases"]}
