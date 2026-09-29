# wall-function log-law inversion: recover u_tau from a point that exactly satisfies the log law
# isolates the Newton solver from any "is the profile actually log?" question
import numpy as np
import pytest
from src.turbulence.wall_function import friction_velocity

von_karman = 0.41  # kappa
log_law_constant = 5.2  # B
# test 1: exact round trip; build the first-node speed from the log law, invert, must recover u_tau
@pytest.mark.parametrize("u_tau, y1, viscosity", [
    (0.0045, 20.0, 0.0016),  # y1+ ~ 56
    (0.05, 10.0, 0.01),  # y1+ ~ 50
    (0.002, 50.0, 0.0004),  # y1+ ~ 250
])
def test_log_law_round_trip(u_tau, y1, viscosity):

    first_node_speed = u_tau * ((1 / von_karman) * np.log(y1 * u_tau / viscosity) + log_law_constant)  # exact log-law velocity

    assert friction_velocity(first_node_speed, y1, viscosity) == pytest.approx(u_tau, rel=1e-5)

# test 2: non-positive wall-parallel speed gives zero stress, no NaN from the log
def test_non_positive_speed_gives_zero():

    assert friction_velocity(0.0, 20.0, 0.0016) == 0.0
    assert friction_velocity(-0.1, 20.0, 0.0016) == 0.0
