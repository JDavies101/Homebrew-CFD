# geometry preview: links put the wall where q says, the saved file round-trips, the report flags a silent wall
import sys
import numpy as np
from src.engine import lattice_d3q19 as d3q19
from src.geometry.preview import boundary_links, wall_points, save_preview, load_preview
from src.geometry.sdf import sdf_sphere, q_from_sdf, solid_from_sdf
from src.run.case import Flow, Domain, Turbulence, Timing, Part
from src.run.case_file import GeometrySpec, CaseFile, save_case_file
from app.viewport import preview_report

grid_size = 20
center = (9.3, 10.1, 9.7)  # off-lattice so q takes many values
radius = 5.4

# test 1: every boundary link's wall point lies on the sphere, and the link count is the q > 0 count
def test_wall_points_on_sphere():

    phi = sdf_sphere(*center, radius)
    q = q_from_sdf(phi, grid_size, grid_size, grid_size)
    nodes, directions, fractions = boundary_links(q)
    offsets = wall_points(nodes, directions, fractions) - np.array(center)
    distance = np.sqrt((offsets * offsets).sum(axis=1))

    assert len(fractions) == np.count_nonzero(q > 0.0)
    assert len(fractions) > 0
    assert np.abs(distance - radius).max() < 1e-4
    assert np.all((fractions > 0.0) & (fractions <= 1.0))
    assert np.all(d3q19.lattice_velocities[directions].any(axis=1))

# test 2: two parts (staircase, Bouzidi) round-trip with their ids, names and only the Bouzidi part's links
def test_save_load_round_trip(tmp_path):

    shape = (grid_size, grid_size, grid_size)
    phi = sdf_sphere(*center, radius)
    sphere = Part(name="sphere", solid=solid_from_sdf(phi, *shape), reference_area=1.0, wall_fractions=q_from_sdf(phi, *shape))
    block_solid = np.zeros(shape, np.int32)
    block_solid[1:3, 1:3, 1:3] = 1
    block = Part(name="block", solid=block_solid, reference_area=1.0)
    save_preview(tmp_path / "preview.npz", [block, sphere], shape)
    preview = load_preview(tmp_path / "preview.npz")

    assert list(preview["names"]) == ["block", "sphere"]
    assert list(preview["bouzidi"]) == [False, True]
    assert np.count_nonzero(preview["part_id"] == 1) == 8
    assert np.count_nonzero(preview["part_id"] == 2) == np.count_nonzero(sphere.solid)
    assert set(np.unique(preview["link_part"])) == {2}
    assert len(preview["fractions"]) == np.count_nonzero(sphere.wall_fractions > 0.0)

# test 3: the report flags a Bouzidi part with no links and a part with no solid cells
def test_report_flags_silent_wall(tmp_path):

    shape = (8, 8, 8)
    solid = np.zeros(shape, np.int32)
    solid[3:5, 3:5, 3:5] = 1
    unset = Part(name="unset", solid=solid, reference_area=1.0, wall_fractions=np.zeros((d3q19.direction_count, *shape), np.float32))
    empty = Part(name="empty", solid=np.zeros(shape, np.int32), reference_area=1.0)
    save_preview(tmp_path / "preview.npz", [unset, empty], shape)
    lines = preview_report(load_preview(tmp_path / "preview.npz"))

    assert "no boundary links" in lines[0]
    assert "no solid cells" in lines[1]
    assert "staircase walls" in lines[1]

# test 4: --preview builds a case's geometry and writes the file without making a run folder
def test_preview_command(tmp_path, monkeypatch):

    from src.run.__main__ import main
    case_file = CaseFile(name="tiny_wing", flow=Flow(), domain=Domain(nx=48, ny=32, nz=4), turbulence=Turbulence(), timing=Timing(),
                         geometry=[GeometrySpec(name="wing", kind="naca", reference_area=16.0, chord=16.0, angle_degrees=4.0,
                                                leading_edge=[12.0, 8.0])])
    save_case_file(case_file, tmp_path / "case.json")
    monkeypatch.setattr(sys, "argv", ["src.run", str(tmp_path / "case.json"), "--preview", str(tmp_path / "preview.npz"),
                                      "--backend", "cpu", "--runs", str(tmp_path / "runs")])
    main()
    preview = load_preview(tmp_path / "preview.npz")

    assert np.count_nonzero(preview["part_id"] == 1) > 0
    assert len(preview["fractions"]) > 0
    assert not (tmp_path / "runs").exists()
