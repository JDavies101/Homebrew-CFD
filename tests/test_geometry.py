# geometry: solid masks and sub-cell wall fractions
import numpy as np
import pytest
from src.engine import lattice_d3q19 as d3q19
from src.geometry.cylinder_body import cylinder
from src.geometry.sphere_body import sphere
from src.geometry.step_body import step
from src.geometry.ahmed_body import ahmed_body
from src.geometry.wall_fraction import wall_fraction_cylinder, wall_fraction_sphere

def _max_distance_from_sphere(q, center, radius):
    """
    Largest distance between a q crossing point and the sphere surface.

    Returns the max error in lattice units.
    """

    max_error = 0.0
    for d in range(d3q19.direction_count):
        for (i, j, k) in np.argwhere(q[d] > 0):
            crossing = np.array([i, j, k]) + q[d, i, j, k] * d3q19.lattice_velocities[d]
            max_error = max(max_error, abs(np.linalg.norm(crossing - center) - radius))

    return max_error

# test 1: cylinder mask: right area, spans z, centered
def test_cylinder_mask():

    solid = cylinder(60, 60, 4, 30, 30, 10)
    disc_area = np.pi * 10 * 10

    assert solid[30, 30, 0] == 1 and solid[0, 0, 0] == 0
    assert np.array_equal(solid[:, :, 0], solid[:, :, 3])  # uniform in z
    assert abs(solid[:, :, 0].sum() - disc_area) / disc_area < 0.05

# test 2: sphere mask: right volume, centered
def test_sphere_mask():

    solid = sphere(40, 40, 40, 20, 20, 20, 7)
    ball_volume = 4 / 3 * np.pi * 7 * 7 * 7

    assert solid[20, 20, 20] == 1 and solid[0, 0, 0] == 0
    assert abs(solid.sum() - ball_volume) / ball_volume < 0.05

# test 3: step mask: two walls plus the block, open floor downstream
def test_step_mask():

    nx, ny, nz, x_step, step_height = 200, 64, 4, 60, 30
    solid = step(nx, ny, nz, x_step, step_height)

    assert solid[:, 0, :].all() and solid[:, -1, :].all()  # both walls solid
    assert solid[10, 1 : step_height + 1, 0].all()  # step block upstream
    assert solid[x_step + 10, 1, 0] == 0  # floor open downstream

# test 4: cylinder wall fraction: q in (0, 1], crossing lands on the circle
def test_wall_fraction_cylinder():

    q = wall_fraction_cylinder(40, 40, 4, 20, 20, 6)
    boundary_q = q[q > 0]
    # crossing distance from the axis (z ignored, xy distance only)
    max_error = 0.0
    for d in range(d3q19.direction_count):
        for (i, j, k) in np.argwhere(q[d] > 0):
            crossing = np.array([i, j]) + q[d, i, j, k] * d3q19.lattice_velocities[d, :2]
            max_error = max(max_error, abs(np.hypot(crossing[0] - 20, crossing[1] - 20) - 6))

    assert (boundary_q > 0).all() and (boundary_q <= 1).all()
    assert max_error < 1e-4

# test 5: sphere wall fraction: q in (0, 1], crossing lands on the sphere
def test_wall_fraction_sphere():

    q = wall_fraction_sphere(40, 40, 40, 20, 20, 20, 7)
    boundary_q = q[q > 0]

    assert (boundary_q > 0).all() and (boundary_q <= 1).all()
    assert _max_distance_from_sphere(q, np.array([20, 20, 20]), 7) < 1e-4

# test 6: ahmed mask, measured back from the array against the real body at height 48
# (1044 x 288 x 389 mm -> 174 x 48 x 65 cells, 50 mm ground gap -> 8), slant matches the angle
@pytest.mark.parametrize("slant_angle", [25, 30, 35])
def test_ahmed_mask(slant_angle):

    nx, ny, nz, x_start = 300, 80, 100, 50
    solid = ahmed_body(nx, ny, nz, x_start, body_height=48, slant_angle=slant_angle)
    x_cells = np.where(solid.any(axis=(1, 2)))[0]
    y_cells = np.where(solid.any(axis=(0, 2)))[0]
    z_cells = np.where(solid.any(axis=(0, 1)))[0]
    # slant: over the rear half, mid-span column heights fall at tan(slant_angle)
    column_height = solid[:, :, nz // 2].sum(axis=1)
    rear = np.arange(x_cells.min() + 174 // 2, x_cells.max() + 1)
    slant_columns = rear[column_height[rear] < 48]
    slope = np.polyfit(slant_columns, column_height[slant_columns], 1)[0]

    assert not solid[:, 0, :].any()  # body only, the run adds the floor
    assert x_cells.min() == x_start and x_cells.max() - x_cells.min() + 1 == 174  # length
    assert y_cells.min() == 9 and y_cells.max() - y_cells.min() + 1 == 48  # 8-cell gap, then 48 tall
    assert z_cells.max() - z_cells.min() + 1 == 65  # width
    assert abs((z_cells.min() + z_cells.max()) / 2 - (nz - 1) / 2) <= 0.5  # centered spanwise
    assert solid[x_cells.min(), y_cells.min(), z_cells.min()] == 0  # nose: leading bottom-side corner carved
    assert solid[x_cells.min(), (y_cells.min() + y_cells.max()) // 2, nz // 2] == 1  # middle of the front face kept
    assert abs(np.degrees(np.arctan(-slope)) - slant_angle) < 1.5
