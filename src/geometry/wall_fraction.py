# analytic sub-cell wall fractions q for interpolated bounce-back
import numpy as np
from src.engine import lattice_d3q19 as d3q19

def wall_fraction_cylinder(nx, ny, nz, center_x, center_y, radius):
    """
    Exact q for a z-spanning cylinder: first link-circle crossing from the fluid side.

    Returns float32 q (19, nx, ny, nz), 0 on non-boundary links.
    """

    q = np.zeros((d3q19.direction_count, nx, ny, nz), np.float32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    offset_x = X - center_x
    offset_y = Y - center_y
    squared_distance = offset_x * offset_x + offset_y * offset_y
    solid = squared_distance < radius * radius  # (nx, ny) bool
    # quadratic a t^2 + b t + c = 0 for |p + t e| = radius; c is the same for all directions
    quadratic_c = squared_distance - radius * radius

    for d in range(d3q19.direction_count):
        ex = int(d3q19.lattice_velocities[d, 0])
        ey = int(d3q19.lattice_velocities[d, 1])
        quadratic_a = ex * ex + ey * ey
        # pure-z link can't hit the cylinder
        if quadratic_a == 0:
            continue

        # neighbour_solid[i, j] = solid[i + ex, j + ey], False out of bounds (no wrap)
        neighbour_solid = np.zeros_like(solid)
        neighbour_solid[max(-ex, 0) : nx - max(ex, 0), max(-ey, 0) : ny - max(ey, 0)] = solid[max(ex, 0) : nx + min(ex, 0), max(ey, 0) : ny + min(ey, 0)]
        link = (~solid) & neighbour_solid

        quadratic_b = 2 * (offset_x * ex + offset_y * ey)
        discriminant = quadratic_b * quadratic_b - 4 * quadratic_a * quadratic_c
        crossing_links = link & (discriminant >= 0)
        crossing = (-quadratic_b[crossing_links] - np.sqrt(discriminant[crossing_links])) / (2 * quadratic_a)  # first crossing from fluid side
        q[d][crossing_links] = crossing[:, None]  # same for every z (cylinder spans z)

    return q

def wall_fraction_sphere(nx, ny, nz, center_x, center_y, center_z, radius):
    """
    Exact q for a sphere: first link-sphere crossing from the fluid side.

    Returns float32 q (19, nx, ny, nz), 0 on non-boundary links.
    """

    q = np.zeros((d3q19.direction_count, nx, ny, nz), np.float32)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    offset_x, offset_y, offset_z = X - center_x, Y - center_y, Z - center_z
    solid = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z < radius * radius
    # quadratic a t^2 + b t + c = 0 for |p + t e| = radius; c is the same for all directions
    quadratic_c = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z - radius * radius

    # 19 iterations, each fully vectorized
    for d in range(d3q19.direction_count):
        ex, ey, ez = (int(v) for v in d3q19.lattice_velocities[d])
        quadratic_a = ex * ex + ey * ey + ez * ez
        if quadratic_a == 0:
            continue
        neighbour_solid = np.roll(solid, (-ex, -ey, -ez), axis=(0, 1, 2))
        link = (~solid) & neighbour_solid
        quadratic_b = 2 * (offset_x * ex + offset_y * ey + offset_z * ez)
        discriminant = quadratic_b * quadratic_b - 4 * quadratic_a * quadratic_c
        crossing_links = link & (discriminant >= 0)
        q[d][crossing_links] = ((-quadratic_b[crossing_links] - np.sqrt(discriminant[crossing_links])) / (2 * quadratic_a)).astype(np.float32)

    return q
