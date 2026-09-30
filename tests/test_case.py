# run case description: derived flow quantities, step counts, validation
import numpy as np
import pytest
from src.run.case import Flow, Domain, Turbulence, Timing, Part, Case

grid_shape = (8, 4, 2)

def small_case(**overrides):
    """
    A valid regularized inflow case on a tiny grid; keyword overrides replace Case fields.

    Returns the Case.
    """

    settings = dict(name="test", tag="t", flow=Flow(), domain=Domain(nx=8, ny=4, nz=2), turbulence=Turbulence(), timing=Timing())
    settings.update(overrides)

    return Case(**settings)

# test 1: default flow reproduces the wing run 74 arithmetic bit for bit
def test_flow_default_matches_wing_run():

    flow = Flow()
    expected_viscosity = 0.05 * 80.0 / 5000.0
    expected_relaxation_time = 3 * expected_viscosity + 0.5

    assert flow.viscosity == expected_viscosity
    assert flow.relaxation_time == expected_relaxation_time
    assert np.isclose(flow.relaxation_time, 0.5024)

# test 2: a given relaxation time overrides Re and sets the viscosity from it
def test_flow_relaxation_time_override():

    flow = Flow(relaxation_time_override=0.8)

    assert flow.relaxation_time == 0.8
    assert np.isclose(flow.viscosity, 0.1)

# test 3: tau below the floor is rejected unless allowed on purpose
def test_validate_relaxation_time_floor():

    below_floor = small_case(flow=Flow(reynolds_number=20000.0))
    allowed = small_case(flow=Flow(reynolds_number=20000.0), allow_below_floor=True)
    allowed.validate()

    assert np.isclose(below_floor.flow.relaxation_time, 0.5006)
    with pytest.raises(ValueError, match="below 0.501"):
        below_floor.validate()

# test 4: an unknown collision name is rejected
def test_validate_unknown_collision():

    case = small_case(collision="regularised")

    with pytest.raises(ValueError, match="collision = 'regularised'"):
        case.validate()

# test 5: a part mask that does not match the grid is rejected
def test_validate_part_shape():

    part = Part(name="box", solid=np.zeros((8, 4, 3), np.int32), reference_area=1.0)
    case = small_case(parts=[part])

    with pytest.raises(ValueError, match="part box mask"):
        case.validate()

# test 6: Bouzidi and staircase parts in one case are rejected
def test_validate_mixed_wall_types():

    bouzidi_part = Part(name="wing", solid=np.zeros(grid_shape, np.int32), reference_area=1.0, wall_fractions=np.zeros((19,) + grid_shape, np.float32))
    staircase_part = Part(name="box", solid=np.zeros(grid_shape, np.int32), reference_area=1.0)
    case = small_case(parts=[bouzidi_part, staircase_part])

    with pytest.raises(ValueError, match="mixed Bouzidi and staircase"):
        case.validate()

# test 7: periodic x with an inlet is rejected
def test_validate_periodic_x_with_inlet():

    case = small_case(domain=Domain(nx=8, ny=4, nz=2, x_boundary="periodic"))

    with pytest.raises(ValueError, match="periodic x needs"):
        case.validate()

# test 8: fused layers with a non-regularized collision are rejected
def test_validate_fused_layers_need_regularized():

    case = small_case(collision="trt")

    with pytest.raises(ValueError, match="fused layers"):
        case.validate()

# test 9: wing run 74 settings give its step counts and pass validation
def test_wing_case_step_counts():

    wing_solid = np.zeros((800, 400, 4), np.int32)
    wing = Part(name="wing", solid=wing_solid, reference_area=80 * 4, wall_fractions=np.zeros((19, 800, 400, 4), np.float32))
    domain = Domain(nx=800, ny=400, nz=4, floor="moving", ceiling="moving")
    case = Case(name="wing_ground", tag="h0.3_a4", flow=Flow(), domain=domain, turbulence=Turbulence(), timing=Timing(), parts=[wing])
    case.validate()

    assert case.total_steps() == 160000
    assert case.warmup_steps() == 64000
    assert case.ramp_steps() == 16000

# test 10: a custom start without an initial velocity field is rejected
def test_validate_custom_start_needs_velocity():

    case = small_case(start="custom")

    with pytest.raises(ValueError, match="start = 'custom' needs initial_velocity"):
        case.validate()

# test 11: the wall model with Bouzidi walls is rejected
def test_validate_wall_model_needs_staircase():

    part = Part(name="wing", solid=np.zeros(grid_shape, np.int32), reference_area=1.0, wall_fractions=np.zeros((19,) + grid_shape, np.float32))
    case = small_case(parts=[part], turbulence=Turbulence(wall_model=True))

    with pytest.raises(ValueError, match="wall model needs staircase"):
        case.validate()

# test 12: a lid velocity is rejected until the 2D engine port
def test_validate_lid_velocity_reserved():

    case = small_case(domain=Domain(nx=8, ny=4, nz=2, lid_velocity=0.1))

    with pytest.raises(ValueError, match="lid_velocity is reserved"):
        case.validate()

# test 13: wall fractions that do not match the grid are rejected
def test_validate_wall_fraction_shape():

    part = Part(name="wing", solid=np.zeros(grid_shape, np.int32), reference_area=1.0, wall_fractions=np.zeros((19, 8, 4, 3), np.float32))
    case = small_case(parts=[part])

    with pytest.raises(ValueError, match="wall fractions"):
        case.validate()