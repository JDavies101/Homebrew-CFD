# new-study answers: physical to lattice units, Re cap at the tau floor, domain sizing per problem
import numpy as np
from src.geometry.mesh import box_mesh, inspect_mesh
from src.run.case import minimum_relaxation_time
from src.run.case_file import part_free_case
from src.run.study import StudyAnswers, physical_reynolds_number, case_from_answers, add_stl_part

# test 1: the defaults alone give a case the engine accepts
def test_defaults_validate():

    case = part_free_case(case_from_answers(StudyAnswers()))
    case.validate()

    assert case.domain.nx == 384
    assert case.flow.reference_length == 64.0

# test 2: physical Re from speed, length and fluid against the hand value
def test_physical_reynolds_number():

    air = physical_reynolds_number(StudyAnswers(speed=50.0, body_length=0.5, fluid="air"))
    water = physical_reynolds_number(StudyAnswers(speed=2.0, body_length=0.1, fluid="water"))

    assert abs(air - 50.0 * 0.5 / 1.516e-5) < 1e-6
    assert abs(water - 2.0 * 0.1 / 1.004e-6) < 1e-6

# test 3: a high physical Re is capped so tau sits on the floor, not under it
def test_reynolds_number_capped_at_floor():

    flow = case_from_answers(StudyAnswers()).flow

    assert flow.relaxation_time >= minimum_relaxation_time
    assert flow.relaxation_time < minimum_relaxation_time + 1e-6
    assert flow.reynolds_number < physical_reynolds_number(StudyAnswers())

# test 4: allow_below_floor keeps the physical Re and the case still validates
def test_below_floor_keeps_physical_reynolds_number():

    answers = StudyAnswers(allow_below_floor=True)
    case = part_free_case(case_from_answers(answers))
    case.validate()

    assert case.flow.reynolds_number == physical_reynolds_number(answers)
    assert case.flow.relaxation_time < minimum_relaxation_time

# test 5: a low physical Re is used as it is
def test_low_reynolds_number_not_capped():

    answers = StudyAnswers(speed=0.02, body_length=0.1, fluid="water")
    flow = case_from_answers(answers).flow

    assert flow.reynolds_number == physical_reynolds_number(answers)
    assert flow.relaxation_time > minimum_relaxation_time

# test 6: free stream is 6 x 3 x 3 body lengths with free-slip y and z
def test_free_stream_domain():

    domain = case_from_answers(StudyAnswers(problem="free_stream", resolution="coarse")).domain

    assert (domain.nx, domain.ny, domain.nz) == (240, 120, 120)
    assert domain.y_boundary == "free_slip"
    assert domain.side_walls == "free_slip"

# test 7: ground problems are 2 body lengths high with y walls, the floor static or moving, the ceiling moving
def test_ground_domain():

    static = case_from_answers(StudyAnswers(problem="ground_static")).domain
    moving = case_from_answers(StudyAnswers(problem="ground_moving")).domain

    assert static.ny == 128
    assert static.y_boundary == "walls"
    assert static.floor == "static"
    assert static.ceiling == "moving"
    assert moving.floor == "moving"

# test 8: thin span is the quasi-2D setup, nz = 4 with periodic sides
def test_thin_span():

    domain = case_from_answers(StudyAnswers(thin_span=True)).domain

    assert domain.nz == 4
    assert domain.side_walls == "periodic"

# test 9: an unsteady study runs twice as long and samples more often
def test_unsteady_timing():

    averaged = case_from_answers(StudyAnswers(study="averaged")).timing
    unsteady = case_from_answers(StudyAnswers(study="unsteady")).timing

    assert unsteady.total_flow_throughs == 2 * averaged.total_flow_throughs
    assert unsteady.sample_every < averaged.sample_every

# test 10: an unknown answer is rejected, not silently defaulted
def test_unknown_problem_rejected():

    try:
        case_from_answers(StudyAnswers(problem="rocket"))
        rejected = False
    except ValueError:
        rejected = True

    assert rejected

# test 11: an STL part is scaled so its x-length is the body length in cells, front at nx / 4
def test_add_stl_part_scale():

    triangles = box_mesh((-1.0, 0.0, -0.5), (3.0, 1.0, 0.5))
    case_file = case_from_answers(StudyAnswers())
    add_stl_part(case_file, "body.stl", inspect_mesh(triangles, 1.0), "body")
    spec = case_file.geometry[0]
    placed = (triangles * spec.cells_per_unit + np.asarray(spec.offset)).reshape(-1, 3)

    assert spec.cells_per_unit == 16.0
    assert abs((placed[:, 0].max() - placed[:, 0].min()) - 64.0) < 1e-9
    assert abs(placed[:, 0].min() - 96.0) < 1e-9
    assert len(case_file.geometry) == 1