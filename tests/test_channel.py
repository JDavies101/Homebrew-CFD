# forced-channel golden oracle + TRT-forcing regression
# the composed collide_full(tau, cs, gx, trt) must inject momentum correctly for both collision operators.
# the parity test (omega- = omega+ -> BGK) is blind to the Guo antisymmetric-source relaxation rate, so a bug
# there only shows up when TRT and BGK are driven by the same force and their peaks are compared. They must
# agree: forced Poiseuille is operator-independent up to BGK's tau-dependent wall location.
# TRT (Lambda = 3/16) puts the wall exactly halfway, so its peak must hit g delta^2 / (2 nu)
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D
from src.engine import lattice_d3q19 as d3q19

pytestmark = pytest.mark.slow
nx, ny, nz = 8, 66, 8  # delta = 32: the validated regime; BGK's wall shift is a small fraction of the peak here
relaxation_time = 0.8
viscosity = (relaxation_time - 0.5) / 3
half_height = (ny - 2) / 2  # delta, halfway walls: solid at j = 0, j = ny - 1
body_force_x = 1e-6  # g
steps = 25000  # ~6 viscous decay times at this delta
analytic_peak = body_force_x * half_height * half_height / (2 * viscosity)

def _run(trt):
    """
    Run the forced channel with BGK (trt = 0) or TRT (trt = 1).

    Returns the u_x profile averaged over x and z, shape (ny,).
    """

    sim = Simulation3D(nx, ny, nz, "cpu")
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, nx, ny, nz)).astype(np.float64))
    for _ in range(steps):
        sim.collide_full(relaxation_time, 0.0, body_force_x, trt)
        sim.stream()
        sim.bounce_back()
    sim.macroscopic()

    return sim.u.to_numpy()[0].mean(axis=(0, 2))

def _fit(profile):
    """
    Fit the fluid nodes to u = c1 y^2 + c0, y from the centerline.

    Returns (R^2, peak = c0).
    """

    j = np.arange(1, len(profile) - 1)  # fluid nodes only
    y = j - (len(profile) - 1) / 2.0  # centerline coordinates
    velocity = profile[1:-1]
    y_squared = y * y
    c1, c0 = np.polyfit(y_squared, velocity, 1)
    r_squared = 1.0 - (velocity - (c1 * y_squared + c0)).var() / velocity.var()

    return r_squared, c0

@pytest.fixture(scope="module")
def fits():

    return {"bgk": _fit(_run(0)), "trt": _fit(_run(1))}

# test 1: both operators give a clean parabola (anything below 1 is a real defect)
def test_profile_is_parabola(fits):

    assert fits["bgk"][0] >= 0.9999
    assert fits["trt"][0] >= 0.9999

# test 2: the regression: same g, nu, delta -> same physical peak, operator-independent
# (pre-fix, with the wrong antisymmetric-source rate, these were ~18% apart)
def test_trt_bgk_peaks_agree(fits):

    peak_bgk, peak_trt = fits["bgk"][1], fits["trt"][1]

    assert abs(peak_trt / peak_bgk - 1) < 0.01

# test 3: TRT peak vs g delta^2 / (2 nu); was +1.95% while wall nodes were being collided
def test_peak_matches_analytic(fits):

    assert abs(fits["trt"][1] / analytic_peak - 1) < 0.005
