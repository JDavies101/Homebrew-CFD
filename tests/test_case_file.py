# case files: JSON load / save, strict schema, geometry build
import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
from src.run.case import Flow, Domain, Turbulence, Timing
from src.run.case_file import GeometrySpec, CaseFile, save_case_file, load_case_file, build_part

template_path = Path(__file__).resolve().parent.parent / "cases" / "templates" / "wing_ground.json"

def stl_case_file():
    """
    A small case with one STL part (path relative, as a user would write it).

    Returns the CaseFile.
    """

    part = GeometrySpec(name="body", kind="stl", reference_area=100.0, path="part.stl", cells_per_unit=20.0, offset=[40.0, 30.0, 30.0])

    return CaseFile(name="stl_test", flow=Flow(), domain=Domain(nx=120, ny=60, nz=60), turbulence=Turbulence(), timing=Timing(), geometry=[part])

# test 1: the wing template loads with run 74's flow arithmetic bit for bit
def test_wing_template_loads():

    case_file = load_case_file(template_path)
    expected_relaxation_time = 3 * (0.05 * 80.0 / 5000.0) + 0.5

    assert case_file.flow.relaxation_time == expected_relaxation_time
    assert case_file.domain.floor == "moving"
    assert case_file.geometry[0].kind == "naca"
    assert case_file.geometry[0].leading_edge == [240.0, 24.5]

# test 2: save then load gives the same case, with the STL path made absolute next to the file
def test_save_load_round_trip(tmp_path):

    case_file = stl_case_file()
    path = tmp_path / "case.json"
    save_case_file(case_file, path)
    loaded = load_case_file(path)
    expected_part = dataclasses.replace(case_file.geometry[0], path=str((tmp_path / "part.stl").resolve()))
    expected = dataclasses.replace(case_file, geometry=[expected_part])

    assert loaded == expected

# test 3: an unknown key is rejected, not silently ignored
def test_unknown_key_rejected(tmp_path):

    data = json.loads(template_path.read_text())
    data["flow"]["reynolds"] = 5000.0  # typo for reynolds_number
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(data))

    with pytest.raises(TypeError, match="reynolds"):
        load_case_file(path)

# test 4: a file from another schema version is rejected
def test_wrong_schema_rejected(tmp_path):

    data = json.loads(template_path.read_text())
    data["schema_version"] = 99
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="schema 99"):
        load_case_file(path)

# test 5: a NACA spec builds a non-empty mask and Bouzidi fractions on the grid, none for staircase
def test_build_naca_part():

    domain = Domain(nx=160, ny=80, nz=4)
    spec = GeometrySpec(name="wing", kind="naca", reference_area=160.0, chord=40.0, angle_degrees=4.0, leading_edge=[60.0, 20.0])
    bouzidi_part = build_part(spec, domain, "cpu")
    staircase_part = build_part(dataclasses.replace(spec, wall="staircase"), domain, "cpu")

    assert bouzidi_part.solid.shape == (160, 80, 4)
    assert bouzidi_part.solid.sum() > 0
    assert bouzidi_part.wall_fractions.shape == (19, 160, 80, 4)
    assert np.count_nonzero(bouzidi_part.wall_fractions) > 0
    assert staircase_part.wall_fractions is None
    assert np.array_equal(staircase_part.solid, bouzidi_part.solid)