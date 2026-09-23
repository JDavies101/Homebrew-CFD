# Forced-channel golden oracle + TRT-forcing regression.
# The composed collide_full(tau, cs, gx, trt) must inject momentum correctly for
# BOTH collision operators. The parity test (s_minus=s_plus->BGK) is blind to the
# Guo antisymmetric-source relaxation rate, so a bug there only shows up when TRT
# and BGK are driven by the same force and their peaks are compared. They must
# agree: forced Poiseuille is operator-independent up to BGK's tau-dependent wall location.
# TRT (Lambda=3/16) puts the wall exactly halfway, so its peak must hit g*delta^2/(2 nu).
import numpy as np
import pytest
from src.engine.simulation3d import Simulation3D as S
from src.engine import lattice3d as L

pytestmark = pytest.mark.slow

nx, ny, nz = 8, 66, 8              # delta=32: the validated regime; BGK's tau-dependent
tau = 0.8                          # wall shift is a smaller fraction of the peak here
nu = (tau - 0.5) / 3
delta = (ny - 2) / 2               # halfway walls: solid at j=0, j=ny-1
g = 1e-6
steps = 25000                      # ~6 viscous decay times at this delta
U_analytic = g * delta**2 / (2 * nu)


def _run(trt):
    sim = S(nx, ny, nz, "cpu")
    solid = np.zeros((nx, ny, nz), np.int32)
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1
    sim.solid.from_numpy(solid)
    sim.f.from_numpy(np.tile(L.W[:, None, None, None], (1, nx, ny, nz)).astype(np.float64))
    for _ in range(steps):
        sim.collide_full(tau, 0.0, g, trt)
        sim.stream()
        sim.bounce_back()
    sim.macroscopic()
    return sim.u.to_numpy()[0].mean(axis=(0, 2))   # u_x averaged over x,z -> (ny,)


def _fit(prof):
    j = np.arange(1, len(prof) - 1)                # fluid nodes only
    y = j - (len(prof) - 1) / 2.0                  # centerline coords
    u = prof[1:-1]
    c1, c0 = np.polyfit(y**2, u, 1)                # u ~ c1*y^2 + c0
    r2 = 1.0 - (u - (c1 * y**2 + c0)).var() / u.var()
    return r2, c0                                  # c0 = peak at y=0


@pytest.fixture(scope="module")
def fits():
    return {"bgk": _fit(_run(0)), "trt": _fit(_run(1))}


def test_shape_is_parabola(fits):
    # a clean parabola is the physics proof; anything below 1 is a real defect
    assert fits["bgk"][0] >= 0.9999
    assert fits["trt"][0] >= 0.9999


def test_trt_bgk_peaks_agree(fits):
    # THE regression: same g, nu, delta -> same physical peak, operator-independent.
    # Pre-fix (wrong antisymmetric-source rate) these were ~18% apart.
    peak_bgk, peak_trt = fits["bgk"][1], fits["trt"][1]
    assert abs(peak_trt / peak_bgk - 1) < 0.01


def test_peak_matches_analytic(fits):
    # magnitude vs g*delta^2/(2 nu). TRT is exact at halfway walls; this was +1.95% while
    # wall nodes were being collided, so it catches that bug if it ever comes back
    assert abs(fits["trt"][1] / U_analytic - 1) < 0.005
