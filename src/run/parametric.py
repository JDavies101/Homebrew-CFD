# parametric studies: one swept parameter, one case file per value
import copy
import json
from pathlib import Path
import numpy as np
from src.run.case_file import save_case_file
from src.run.run_folder import case_hash, write_json_atomic

# parameter -> {geometry kind: (field, index into a list field or None)}
sweep_parameters = {"position_y": {"naca": ("leading_edge", 1), "stl": ("offset", 1), "sphere": ("center", 1), "cylinder": ("center", 1)},
                    "angle": {"naca": ("angle_degrees", None), "ahmed": ("slant_angle", None)}}
sweep_labels = {"position_y": "Position y (cells)", "angle": "Angle (degrees)"}

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

def write_study(folder, base_case_file, part_name, parameter, values, study_name):
    """
    Create the study folder with one case file per value and study.json describing the sweep.

    Returns (study dict, list of (variant CaseFile, path)).
    """

    variants = make_variants(base_case_file, part_name, parameter, values, study_name)
    folder = Path(folder)
    if folder.exists():
        raise ValueError(f"a study named {study_name} already exists in this project")

    folder.mkdir(parents=True, exist_ok=False)

    written = []
    for variant in variants:
        path = folder / f"{variant.name}.json"
        save_case_file(variant, path)
        written.append((variant, path))

    study = {"schema_version": 1, "name": study_name, "part": part_name, "parameter": parameter, "values": [float(value) for value in values],
             "base_case_hash": case_hash(base_case_file), "cases": [path.name for _, path in written]}
    write_json_atomic(folder / "study.json", study)

    return study, written

def load_study(folder):
    """
    Read study.json from a study folder.

    Returns the study dict.
    """

    return json.loads((Path(folder) / "study.json").read_text())
