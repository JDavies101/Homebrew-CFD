# viewport geometry: display surfaces in cells and part size reports
from pathlib import Path
import pyvista
from src.geometry.mesh import icosphere, write_stl, box_mesh
from src.run.case import Domain
from src.run.case_file import GeometrySpec, load_case_file
from app.viewport import part_surface, part_report, read_part_mesh, estimate_lines

pyvista.OFF_SCREEN = True
domain = Domain(nx=160, ny=80, nz=4)
template_path = Path(__file__).resolve().parent.parent / "cases" / "templates" / "wing_ground.json"

# test 1: a NACA spec spans one chord in x (rotated by the incidence) and the full span in z
def test_naca_surface_extent():

    spec = GeometrySpec(name="wing", kind="naca", reference_area=160.0, chord=40.0, angle_degrees=4.0, leading_edge=[60.0, 20.0])
    surface = part_surface(spec, domain)
    x_min, x_max, y_min, y_max, z_min, z_max = surface.bounds

    assert 39.0 < x_max - x_min < 40.5
    assert abs(y_min - 20.0) < 0.5
    assert z_min == 0.0
    assert z_max == 4.0

# test 2: an STL is scaled by cells_per_unit and moved by the offset
def test_stl_scale_and_offset(tmp_path):

    path = tmp_path / "ball.stl"
    write_stl(path, icosphere((0.0, 0.0, 0.0), 1.0, subdivisions=2))
    spec = GeometrySpec(name="ball", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=10.0, offset=[50.0, 40.0, 2.0])
    x_min, x_max, y_min, y_max, z_min, z_max = part_surface(spec, Domain(nx=100, ny=80, nz=80)).bounds

    assert abs((x_max - x_min) - 20.0) < 0.5
    assert abs((x_min + x_max) / 2 - 50.0) < 0.5
    assert abs((y_min + y_max) / 2 - 40.0) < 0.5

# test 3: a part too small in cells is flagged (the mm-vs-m import mistake)
def test_report_flags_coarse_part(tmp_path):

    path = tmp_path / "tiny.stl"
    write_stl(path, icosphere((0.0, 0.0, 0.0), 1.0, subdivisions=2))
    spec = GeometrySpec(name="tiny", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=1.0, offset=[50.0, 40.0, 40.0])
    report = part_report("tiny", part_surface(spec, Domain(nx=100, ny=80, nz=80)), Domain(nx=100, ny=80, nz=80))

    assert "under 10 cells across" in report

# test 4: a part outside the grid is flagged
def test_report_flags_outside(tmp_path):

    path = tmp_path / "ball.stl"
    write_stl(path, icosphere((0.0, 0.0, 0.0), 1.0, subdivisions=2))
    spec = GeometrySpec(name="ball", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=20.0, offset=[5.0, 40.0, 40.0])
    report = part_report("ball", part_surface(spec, Domain(nx=100, ny=80, nz=80)), Domain(nx=100, ny=80, nz=80))

    assert "outside the domain" in report

# test 5: an STL report shows its size in STL units and the scale, and no watertight warning when closed
def test_report_shows_units_and_scale(tmp_path):

    path = tmp_path / "ball.stl"
    write_stl(path, icosphere((0.0, 0.0, 0.0), 1.0, subdivisions=2))
    spec = GeometrySpec(name="ball", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=10.0, offset=[50.0, 40.0, 40.0])
    big_domain = Domain(nx=100, ny=80, nz=80)
    _, inspection = read_part_mesh(str(path))
    report = part_report("ball", part_surface(spec, big_domain), big_domain, inspection, spec.cells_per_unit)

    assert "units at 10 cells/unit" in report
    assert "not watertight" not in report

# test 6: an STL with a hole is flagged with its open edge count
def test_report_flags_open_stl(tmp_path):

    path = tmp_path / "open_box.stl"
    write_stl(path, box_mesh((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))[1:])
    spec = GeometrySpec(name="box", kind="stl", reference_area=1.0, path=str(path), cells_per_unit=20.0, offset=[40.0, 30.0, 30.0])
    big_domain = Domain(nx=100, ny=80, nz=80)
    _, inspection = read_part_mesh(str(path))
    report = part_report("box", part_surface(spec, big_domain), big_domain, inspection, spec.cells_per_unit)

    assert "not watertight (3 open, 0 flipped, 0 non-manifold edges)" in report

# test 7: the cache returns the same triangles and inspection objects on the second read
def test_read_part_mesh_cache(tmp_path):

    path = tmp_path / "ball.stl"
    write_stl(path, icosphere((0.0, 0.0, 0.0), 1.0, subdivisions=1))
    cache = {}
    first = read_part_mesh(str(path), cache)
    second = read_part_mesh(str(path), cache)

    assert second[0] is first[0]
    assert second[1] is first[1]
    assert list(cache) == [str(path)]

# test 8: the memory line turns red with "will not fit" when the fields exceed the GPU, and drops the total when unknown
def test_estimate_lines_fit():

    case_file = load_case_file(template_path)
    fits = estimate_lines(case_file, 600.0, False, 100.0)
    too_big = estimate_lines(case_file, 600.0, True, 0.1)
    unknown = estimate_lines(case_file, 600.0, False, None)

    assert "will not fit" not in fits[0]
    assert "will not fit" in too_big[0]
    assert "measured on this machine" in too_big[1]
    assert " of " not in unknown[0]

# test 9: analytic kinds draw at their size: sphere 2R across, cylinder 2R across and the full span
def test_analytic_surfaces():

    big_domain = Domain(nx=100, ny=80, nz=12)
    sphere_bounds = part_surface(GeometrySpec(name="s", kind="sphere", reference_area=1.0, center=[50.0, 40.0, 6.0], radius=5.0), big_domain).bounds
    cylinder_bounds = part_surface(GeometrySpec(name="c", kind="cylinder", reference_area=1.0, center=[50.0, 40.0, 0.0], radius=5.0), big_domain).bounds

    assert abs((sphere_bounds[1] - sphere_bounds[0]) - 10.0) < 0.1
    assert abs((cylinder_bounds[1] - cylinder_bounds[0]) - 10.0) < 0.1
    assert cylinder_bounds[4] == 0.0
    assert cylinder_bounds[5] == 12.0