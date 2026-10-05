# tests for parametric studies: variants, ranges, and the study folder
from dataclasses import asdict
import pytest
from src.run.case_file import load_case_file
from src.run.parametric import make_variants, values_from_range, write_study, load_study, sweep_choices, current_value, unique_study_name
from src.run.run_folder import case_hash

wing_file = load_case_file("cases/templates/wing_ground.json")
ahmed_file = load_case_file("cases/templates/ahmed.json")
wing_name = wing_file.geometry[0].name
ahmed_name = ahmed_file.geometry[0].name

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
    study, written = write_study(folder, wing_file, wing_name, "angle", [2.0, 4.0], "sweep")
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
        write_study(folder, wing_file, wing_name, "angle", [2.0, 4.0], "sweep")

    assert list(folder.iterdir()) == []
