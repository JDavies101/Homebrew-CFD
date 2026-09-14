# wall-function log-law inversion: recover u_tau from a point that exactly satisfies
# the log law. Isolates the Newton solver from any "is the profile actually log?" question.
import numpy as np
import pytest
from src.turbulence.wall_function import friction_velocity

k, B = 0.41, 5.2

# test 1: exact round-trip; build u1 from the log law, invert, must recover u_tau
@pytest.mark.parametrize("u_tau, y1, nu", [
    (0.0045, 20.0, 0.0016),   # y1+ ~ 56
    (0.05,   10.0, 0.01),     # y1+ ~ 50
    (0.002,  50.0, 0.0004),   # y1+ ~ 250
])
def test_roundtrip(u_tau, y1, nu):
    u1 = u_tau * ((1 / k) * np.log(y1 * u_tau / nu) + B)   # exact log-law velocity
    assert friction_velocity(u1, y1, nu) == pytest.approx(u_tau, rel=1e-5)

# test 2: non-positive wall-parallel velocity -> zero stress, no NaN from log
def test_nonpositive():
    assert friction_velocity(0.0, 20.0, 0.0016) == 0.0
    assert friction_velocity(-0.1, 20.0, 0.0016) == 0.0
