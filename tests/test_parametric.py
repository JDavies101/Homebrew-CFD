# tests for studies: variants, ranges, study settings and the study folder
import json
from dataclasses import asdict
import pytest
from src.run.case_file import load_case_file, save_case_file
from src.run.parametric import (make_variants, values_from_range, write_study, load_study, sweep_choices, current_value, unique_study_name,
                                apply_study_settings, study_defaults, default_study_name)
from src.run.run_folder import case_hash

wing_file = load_case_file("cases/templates/wing_ground.json")
ahmed_file = load_case_file("cases/templates/ahmed.json")
wing_name = wing_file.geometry[0].name
ahmed_name = ahmed_file.geometry[0].name

def study_for(name, sweep=None, total_flow_throughs=3.0):
    """
    A study dict on the wing case's defaults, with a shorter run so the study's timing differs from the case's.

    Returns the dict write_study takes.
    """

    solver, timing = study_defaults(wing_file)
    timing.total_flow_throughs = total_flow_throughs

    return {"name": name, "type": "steady", "solver": asdict(solver), "timing": asdict(timing), "sweep": sweep}

# test 1: a position_y variant changes only leading_edge[1]
def test_variant_changes_only_position_y():

    variant = make_variants(wing_file, wing_name, "position_y", [30.5], "sweep")[0]
    variant_data = asdict(variant)
    variant_data["name"] = wing_file.name
    variant_data["geometry"][0]["leading_edge"][1] = wing_file.geometry[0].leading_edge[1]

    assert variant.geometry[0].leading_edge[1] == 30.5
    assert variant_data == asdict(wing_file)

# test 2: an angle variant changes only angle_degrees
def test_variant_changes_only_angle():

    variant = make_variants(wing_file, wing_name, "angle", [7.0], "sweep")[0]
    variant_data = asdict(variant)
    variant_data["name"] = wing_file.name
    variant_data["geometry"][0]["angle_degrees"] = wing_file.geometry[0].angle_degrees

    assert variant.geometry[0].angle_degrees == 7.0
    assert variant_data == asdict(wing_file)

# test 3: variant names are unique and ordered
def test_variant_names_ordered():

    variants = make_variants(wing_file, wing_name, "angle", [2.0, 4.0, 6.0], "sweep")
    names = [variant.name for variant in variants]

    assert names == ["sweep_00", "sweep_01", "sweep_02"]

# test 4: ahmed slant angle is an int
def test_ahmed_slant_angle_is_int():

    variant = make_variants(ahmed_file, ahmed_name, "angle", [20.0], "sweep")[0]

    assert isinstance(variant.geometry[0].slant_angle, int)
    assert variant.geometry[0].slant_angle == 20

# test 5: unknown part, invalid parameter for the kind, and empty values raise ValueError
def test_invalid_sweeps_raise():

    with pytest.raises(ValueError):
        make_variants(wing_file, "nothing", "angle", [1.0], "sweep")

    with pytest.raises(ValueError):
        make_variants(ahmed_file, ahmed_name, "position_y", [1.0], "sweep")

    with pytest.raises(ValueError):
        make_variants(wing_file, wing_name, "angle", [], "sweep")

# test 6: values_from_range includes both endpoints and returns count values
def test_values_from_range():

    values = values_from_range(2.0, 10.0, 5)

    with pytest.raises(ValueError):
        values_from_range(0.0, 1.0, 1)

    assert values == [2.0, 4.0, 6.0, 8.0, 10.0]

# test 7: sweep_choices follows the kind and current_value reads the field
def test_choices_and_current_value():

    wing_spec = wing_file.geometry[0]
    ahmed_spec = ahmed_file.geometry[0]

    assert sweep_choices(wing_spec) == ["position_y", "angle"]
    assert sweep_choices(ahmed_spec) == ["angle"]
    assert current_value(wing_spec, "position_y") == wing_spec.leading_edge[1]
    assert current_value(ahmed_spec, "angle") == ahmed_spec.slant_angle

# test 8: write_study then load_study round-trips, files exist and the base hash matches
def test_write_and_load_study(tmp_path):

    folder = tmp_path / "sweep"
    sweep = {"part": wing_name, "parameter": "angle", "values": [2.0, 4.0]}
    study, written = write_study(folder, wing_file, study_for("sweep", sweep))
    loaded = load_study(folder)

    assert loaded == study
    assert loaded["base_case_hash"] == case_hash(wing_file)
    assert loaded["cases"] == ["sweep_00.json", "sweep_01.json"]
    assert all(path.exists() for _, path in written)
    assert load_case_file(written[1][1]).geometry[0].angle_degrees == 4.0

# test 9: unique_study_name returns the base when free, then _2, _3 as folders appear
def test_unique_study_name(tmp_path):

    free = unique_study_name(tmp_path, "sweep")
    (tmp_path / "sweep").mkdir()
    second = unique_study_name(tmp_path, "sweep")
    (tmp_path / "sweep_2").mkdir()
    third = unique_study_name(tmp_path, "sweep")

    assert free == "sweep"
    assert second == "sweep_2"
    assert third == "sweep_3"

# test 10: write_study on an existing folder raises ValueError and leaves it untouched
def test_write_study_existing_folder_raises(tmp_path):

    folder = tmp_path / "sweep"
    folder.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        write_study(folder, wing_file, study_for("sweep", {"part": wing_name, "parameter": "angle", "values": [2.0, 4.0]}))

    assert list(folder.iterdir()) == []

# test 11: apply_study_settings changes only the given solver and timing fields and leaves the base case alone
def test_apply_study_settings_changes_only_solver_and_timing():

    collision = "bgk" if wing_file.collision != "bgk" else "trt"
    total_flow_throughs = wing_file.timing.total_flow_throughs + 1.0
    applied = apply_study_settings(wing_file, {"collision": collision, "allow_below_floor": not wing_file.allow_below_floor},
                                   {"total_flow_throughs": total_flow_throughs, "sample_every": 10})
    applied_data = asdict(applied)
    applied_data["collision"] = wing_file.collision
    applied_data["allow_below_floor"] = wing_file.allow_below_floor
    applied_data["timing"]["total_flow_throughs"] = wing_file.timing.total_flow_throughs
    applied_data["timing"]["sample_every"] = wing_file.timing.sample_every

    with pytest.raises(ValueError):
        apply_study_settings(wing_file, {"name": "other"}, {})

    assert applied.collision == collision
    assert applied.allow_below_floor != wing_file.allow_below_floor
    assert applied.timing.total_flow_throughs == total_flow_throughs
    assert applied.timing.sample_every == 10
    assert applied_data == asdict(wing_file)
    assert wing_file.collision != collision

# test 12: write_study without a sweep writes one case file with the study's timing, and study.json schema 2
def test_write_study_without_sweep(tmp_path):

    folder = tmp_path / "Study 1"
    study, written = write_study(folder, wing_file, study_for("Study 1", total_flow_throughs=3.0))
    recorded = json.loads((folder / "study.json").read_text())
    case_file = load_case_file(folder / "Study 1.json")

    assert recorded["schema_version"] == 2
    assert recorded["sweep"] is None
    assert recorded["type"] == "steady"
    assert recorded["cases"] == ["Study 1.json"]
    assert recorded["timing"]["total_flow_throughs"] == 3.0
    assert sorted(path.name for path in folder.iterdir()) == ["Study 1.json", "study.json"]
    assert len(written) == 1
    assert case_file.name == "Study 1"
    assert case_file.timing.total_flow_throughs == 3.0
    assert case_file.geometry == wing_file.geometry
    assert wing_file.timing.total_flow_throughs != 3.0

# test 13: write_study with a sweep writes one case file per value, each with the study's timing and its own angle
def test_write_study_with_sweep(tmp_path):

    folder = tmp_path / "Study 2"
    sweep = {"part": wing_name, "parameter": "angle", "values": [2.0, 4.0, 6.0]}
    study, written = write_study(folder, wing_file, study_for("Study 2", sweep, total_flow_throughs=3.0))
    case_files = [load_case_file(folder / name) for name in study["cases"]]

    assert study["sweep"] == sweep
    assert study["cases"] == ["Study 2_00.json", "Study 2_01.json", "Study 2_02.json"]
    assert [case_file.geometry[0].angle_degrees for case_file in case_files] == [2.0, 4.0, 6.0]
    assert all(case_file.timing.total_flow_throughs == 3.0 for case_file in case_files)

# test 14: load_study reads a schema 1 folder (a sweep only) as a steady study with the case's solver and timing
def test_load_study_schema_1(tmp_path):

    folder = tmp_path / "old"
    folder.mkdir()
    variants = make_variants(wing_file, wing_name, "angle", [2.0, 4.0], "old")
    for variant in variants:
        save_case_file(variant, folder / f"{variant.name}.json")

    (folder / "study.json").write_text(json.dumps({"schema_version": 1, "name": "old", "part": wing_name, "parameter": "angle",
                                                   "values": [2.0, 4.0], "base_case_hash": "abc", "cases": ["old_00.json", "old_01.json"]}))
    loaded = load_study(folder)
    solver, timing = study_defaults(wing_file)

    assert loaded == {"schema_version": 2, "name": "old", "type": "steady", "solver": asdict(solver), "timing": asdict(timing),
                      "sweep": {"part": wing_name, "parameter": "angle", "values": [2.0, 4.0]}, "base_case_hash": "abc",
                      "cases": ["old_00.json", "old_01.json"]}

# test 15: default_study_name skips names taken by a folder on disk or by an unwritten study
def test_default_study_name(tmp_path):

    free = default_study_name(tmp_path, set())
    (tmp_path / "Study 1").mkdir()
    after_folder = default_study_name(tmp_path, set())
    after_draft = default_study_name(tmp_path, {"Study 2"})
    both = default_study_name(tmp_path, {"Study 2", "Study 3"})

    assert free == "Study 1"
    assert after_folder == "Study 2"
    assert after_draft == "Study 3"
    assert both == "Study 4"

# test 16: write_study refuses a study name that is not a safe folder name, before writing anything
def test_write_study_bad_name_raises(tmp_path):

    with pytest.raises(ValueError, match="study name"):
        write_study(tmp_path / "bad", wing_file, study_for("bad/name"))

    assert list(tmp_path.iterdir()) == []
