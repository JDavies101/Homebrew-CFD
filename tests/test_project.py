# projects: folders, copied geometry, portable case files, runs beside the case
import shutil
import pytest
from src.geometry.mesh import box_mesh, write_stl
from src.run.case import Flow, Domain, Turbulence, Timing
from src.run.case_file import GeometrySpec, CaseFile, save_case_file, load_case_file
from src.run.project import create_project, is_project, runs_directory_for, adopt_geometry

def _case_with_stl(path):
    """
    A part-free case plus one STL part at path.

    Returns a CaseFile.
    """

    return CaseFile(name="body", flow=Flow(), domain=Domain(nx=40, ny=30, nz=30), turbulence=Turbulence(), timing=Timing(),
                    geometry=[GeometrySpec(name="body", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=10.0)])

# test 1: a project gets its folders and case path; a second project of the same name or a bad name is refused
def test_create_project(tmp_path):

    case_path = create_project(tmp_path, "wing study")
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "wing study")
    with pytest.raises(ValueError):
        create_project(tmp_path, "bad/name")

    assert case_path == tmp_path / "wing study" / "wing study.json"
    assert all((tmp_path / "wing study" / subfolder).is_dir() for subfolder in ("geometry", "runs", "exports"))
    assert is_project(case_path)

# test 2: an outside STL is copied in; an identical one is reused, a different one with the same name is numbered
def test_adopt_geometry(tmp_path):

    outside = tmp_path / "outside"
    outside.mkdir()
    write_stl(outside / "body.stl", box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    case_path = create_project(tmp_path, "project")
    case_file = _case_with_stl(outside / "body.stl")
    first_copies = adopt_geometry(case_file, case_path)
    second_copies = adopt_geometry(_case_with_stl(outside / "body.stl"), case_path)
    write_stl(outside / "body.stl", box_mesh((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)))
    changed = _case_with_stl(outside / "body.stl")
    adopt_geometry(changed, case_path)

    assert len(first_copies) == 1
    assert case_file.geometry[0].path == str((tmp_path / "project" / "geometry" / "body.stl").resolve())
    assert second_copies == []
    assert changed.geometry[0].path.endswith("body_2.stl")

# test 3: a saved project can be moved whole and still finds its STL
def test_project_moves_whole(tmp_path):

    outside = tmp_path / "outside.stl"
    write_stl(outside, box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    case_path = create_project(tmp_path / "workspace", "project")
    case_file = _case_with_stl(outside)
    adopt_geometry(case_file, case_path)
    save_case_file(case_file, case_path)
    stored_path = case_path.read_text()
    shutil.move(str(tmp_path / "workspace" / "project"), str(tmp_path / "moved"))
    outside.unlink()
    reloaded = load_case_file(tmp_path / "moved" / "project.json")

    assert '"geometry/outside.stl"' in stored_path
    assert reloaded.geometry[0].path == str((tmp_path / "moved" / "geometry" / "outside.stl").resolve())
    assert (tmp_path / "moved" / "geometry" / "outside.stl").exists()

# test 4: an STL outside the case's folder stays absolute in the file
def test_outside_path_stays_absolute(tmp_path):

    outside = tmp_path / "elsewhere" / "body.stl"
    outside.parent.mkdir()
    write_stl(outside, box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    (tmp_path / "loose").mkdir()
    save_case_file(_case_with_stl(outside), tmp_path / "loose" / "body.json")
    reloaded = load_case_file(tmp_path / "loose" / "body.json")

    assert reloaded.geometry[0].path == str(outside.resolve())

# test 5: runs go to the project's runs folder, or the workspace Runs folder for a loose or unsaved case
def test_runs_directory(tmp_path):

    case_path = create_project(tmp_path, "project")
    loose_path = tmp_path / "loose.json"

    assert runs_directory_for(case_path, tmp_path) == tmp_path / "project" / "runs"
    assert runs_directory_for(loose_path, tmp_path) == tmp_path / "Runs"
    assert runs_directory_for(None, tmp_path) == tmp_path / "Runs"