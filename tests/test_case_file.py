# case files: JSON load / save, strict schema, geometry build
import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
from src.run.case import Flow, Domain, Turbulence, Timing
from src.run.case_file import GeometrySpec, CaseFile, save_case_file, load_case_file, build_part, stl_spec, part_free_case
from src.geometry.mesh import box_mesh, inspect_mesh

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

# test 6: an imported STL is scaled to fit a quarter of the length and placed front at nx / 4, centred in y and z
def test_stl_spec_placement():

    triangles = box_mesh((-1.0, 0.0, -0.5), (3.0, 1.0, 0.5))
    domain = Domain(nx=200, ny=80, nz=80)
    spec = stl_spec("body.stl", inspect_mesh(triangles, 1.0), domain, "body")
    placed = triangles * spec.cells_per_unit + np.asarray(spec.offset)
    placed_lower = placed.reshape(-1, 3).min(axis=0)
    placed_upper = placed.reshape(-1, 3).max(axis=0)

    assert spec.cells_per_unit == 12.5
    assert np.allclose(placed_lower, [50.0, 33.75, 33.75])
    assert np.allclose(placed_upper, [100.0, 46.25, 46.25])
    assert abs(spec.reference_area - 156.25) < 0.1
    assert spec.kind == "stl"
    assert spec.name == "body"

# test 7: the part-free Case carries the file's settings and no parts
def test_part_free_case():

    case_file = load_case_file(template_path)
    case = part_free_case(case_file)

    assert case.name == case_file.name
    assert case.flow is case_file.flow
    assert case.domain is case_file.domain
    assert case.timing is case_file.timing
    assert case.allow_below_floor == case_file.allow_below_floor
    assert case.parts == []