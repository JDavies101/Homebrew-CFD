# viewport geometry: display surfaces in cells and part size reports
import pyvista
from src.geometry.mesh import icosphere, write_stl
from src.run.case import Domain
from src.run.case_file import GeometrySpec
from app.viewport import part_surface, part_report

pyvista.OFF_SCREEN = True
domain = Domain(nx=160, ny=80, nz=4)

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