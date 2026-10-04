# new-study answers: physical to lattice units, Re cap at the tau floor, domain sizing per problem
from src.run.case import Case, minimum_relaxation_time
from src.run.study import StudyAnswers, physical_reynolds_number, case_from_answers

def _case(case_file):
    """
    The runnable Case of a part-free case file, as the app's summary builds it.

    Returns a Case.
    """

    return Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                timing=case_file.timing, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                allow_below_floor=case_file.allow_below_floor)

# test 1: the defaults alone give a case the engine accepts
def test_defaults_validate():

    case = _case(case_from_answers(StudyAnswers()))
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
    case = _case(case_from_answers(answers))
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