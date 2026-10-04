# pre-run estimate: field memory matches the solver exactly, run time and durations from hand values
from src.engine.simulation3d import Simulation3D
from src.post.run_log import fields_gb
from src.run.case import Flow, Domain, Turbulence, Timing
from src.run.case_file import CaseFile
from src.run.estimate import device_bytes, run_seconds, format_duration

# test 1: the estimate equals what a real Simulation3D allocates, with and without the Bouzidi fields
def test_device_bytes_match_solver():

    staircase = Simulation3D(12, 10, 8, "cpu")
    bouzidi = Simulation3D(12, 10, 8, "cpu", interp=True)

    assert abs(device_bytes(12, 10, 8, False) / 1e9 - fields_gb(staircase)) < 1e-12
    assert abs(device_bytes(12, 10, 8, True) / 1e9 - fields_gb(bouzidi)) < 1e-12

# test 2: run 170's logged field memory (800 x 400 x 4, Bouzidi wing) is reproduced
def test_device_bytes_run_170():

    gigabytes = device_bytes(800, 400, 4, True) / 1e9

    assert round(gigabytes, 3) == 0.471

# test 3: 1000 cells for 1000 steps at 1 MLUPS take one second
def test_run_seconds():

    case_file = CaseFile(name="tiny", flow=Flow(), domain=Domain(nx=10, ny=10, nz=10), turbulence=Turbulence(),
                         timing=Timing(steps_override=1000))

    assert abs(run_seconds(case_file, 1.0) - 1.0) < 1e-12
    assert abs(run_seconds(case_file, 2.0) - 0.5) < 1e-12

# test 4: durations read as seconds, minutes, then hours and minutes
def test_format_duration():

    assert format_duration(45) == "45 s"
    assert format_duration(480) == "8 min"
    assert format_duration(8040) == "2 h 14 min"
