# geometry: solid masks and sub-cell wall fractions
import numpy as np
from src.engine import lattice3d as L
from src.geometry.cylinder import cylinder
from src.geometry.sphere import sphere
from src.geometry.step import step
from src.geometry.wall_fraction import wall_fraction_cylinder, wall_fraction_sphere
import pytest
from src.geometry.ahmed_body import ahmed_body

# test 1: cylinder mask: right count, spans z, centered
def test_cylinder_mask():
    s = cylinder(60, 60, 4, 30, 30, 10)
    
    assert s[30, 30, 0] == 1 and s[0, 0, 0] == 0
    assert np.array_equal(s[:, :, 0], s[:, :, 3])          # uniform in z
    assert abs(s[:, :, 0].sum() - np.pi*10**2) / (np.pi*10**2) < 0.05

# test 2: sphere mask: right volume, centered
def test_sphere_mask():
    s = sphere(40, 40, 40, 20, 20, 20, 7)                  # r=7
    
    assert s[20, 20, 20] == 1 and s[0, 0, 0] == 0
    assert abs(s.sum() - 4/3*np.pi*7**3) / (4/3*np.pi*7**3) < 0.05

# test 3:step mask: two walls plus the block, correct expansion
def test_step_mask():
    nx, ny, nz, xs, S = 200, 64, 4, 60, 30
    s = step(nx, ny, nz, xs, S)

    assert s[:, 0, :].all() and s[:, -1, :].all()          # both walls solid
    assert s[10, 1:S+1, 0].all()                           # step block upstream
    assert s[xs+10, 1, 0] == 0                             # floor open downstream

# test 4: wall fraction q: on-surface, in (0,1], zero off boundary links
def _check_on_surface(q, E, center, R):
    maxerr = 0.0
    for d in range(L.Q):
        for (i, j, k) in np.argwhere(q[d] > 0):
            p = np.array([i, j, k]) + q[d, i, j, k]*E[d]
            maxerr = max(maxerr, abs(np.linalg.norm(p - center) - R))
    return maxerr

def test_wall_fraction_cylinder():
    q = wall_fraction_cylinder(40, 40, 4, 20, 20, 6)
    bl = q[q > 0]
    assert (bl > 0).all() and (bl <= 1).all()
    # crossing lands on the circle (z ignored -> use xy distance)
    for d in range(L.Q):
        for (i, j, k) in np.argwhere(q[d] > 0):
            p = np.array([i, j]) + q[d, i, j, k]*L.E[d, :2]
            
            assert abs(np.hypot(p[0]-20, p[1]-20) - 6) < 1e-4

def test_wall_fraction_sphere():
    q = wall_fraction_sphere(40, 40, 40, 20, 20, 20, 7)
    bl = q[q > 0]
    
    assert (bl > 0).all() and (bl <= 1).all()
    assert _check_on_surface(q, L.E, np.array([20, 20, 20]), 7) < 1e-4

# test 5: ahmed mask, measured back from the array against the real body at H=48
# (1044 x 288 x 389 mm -> 174 x 48 x 65 cells, 50 mm ground gap -> 8), slant matches phi
@pytest.mark.parametrize("phi", [25, 30, 35])
def test_ahmed_mask(phi):
    nx, ny, nz, x0 = 300, 80, 100, 50
    s = ahmed_body(nx, ny, nz, x0, H=48, phi=phi)
    xs = np.where(s.any(axis=(1, 2)))[0]
    js = np.where(s.any(axis=(0, 2)))[0]
    ks = np.where(s.any(axis=(0, 1)))[0]

    assert not s[:, 0, :].any()                                   # body only, the run adds the floor
    assert xs.min() == x0 and xs.max() - xs.min() + 1 == 174      # length
    assert js.min() == 9 and js.max() - js.min() + 1 == 48        # 8-cell gap, then 48 tall
    assert ks.max() - ks.min() + 1 == 65                          # width
    assert abs((ks.min() + ks.max()) / 2 - (nz - 1) / 2) <= 0.5   # centered spanwise

    # nose: the leading bottom-side corner is carved, the middle of the front face isn't
    assert s[xs.min(), js.min(), ks.min()] == 0
    assert s[xs.min(), (js.min() + js.max()) // 2, nz // 2] == 1

    # slant: over the rear half, mid-span column heights fall at tan(phi)
    h = s[:, :, nz // 2].sum(axis=1)
    rear = np.arange(xs.min() + 174 // 2, xs.max() + 1)
    slant = rear[h[rear] < 48]
    slope = np.polyfit(slant, h[slant], 1)[0]
    assert abs(np.degrees(np.arctan(-slope)) - phi) < 1.5