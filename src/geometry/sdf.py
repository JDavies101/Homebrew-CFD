# signed distance fields: phi > 0 fluid, phi < 0 solid, phi = 0 on the surface
# one phi gives the solid mask, per-link Bouzidi fractions q, wall normals, and wall distance
import numpy as np
from src.engine import lattice_d3q19 as L3

def sdf_sphere(center_x, center_y, center_z, radius):
    """
    Signed distance to a sphere.

    Returns a function phi(x, y, z) that works on scalars or whole arrays.
    """

    def phi(x, y, z):
        return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2) - radius

    return phi

def node_grid(nx, ny, nz):
    """
    Float coordinates of every lattice node.

    Returns X, Y, Z, each shaped (nx, ny, nz).
    """

    return np.meshgrid(np.arange(nx, dtype=np.float64),
                       np.arange(ny, dtype=np.float64),
                       np.arange(nz, dtype=np.float64),
                       indexing="ij")

def solid_from_sdf(phi, nx, ny, nz):
    """
    Solid mask from the sign of phi at the nodes.

    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    X, Y, Z = node_grid(nx, ny, nz)

    return (phi(X, Y, Z) < 0.0).astype(np.int32)

def q_from_sdf(phi, nx, ny, nz, iterations=40):
    """
    Bouzidi wall fraction q on every fluid-to-solid link, by bisection of phi along the link.

    Returns float32 q (19, nx, ny, nz), 0 on non-boundary links.
    """

    X, Y, Z = node_grid(nx, ny, nz)
    solid = phi(X, Y, Z) < 0.0
    q = np.zeros((L3.Q, nx, ny, nz), np.float32)

    for d in range(L3.Q):
        ex, ey, ez = (int(v) for v in L3.E[d])
        if ex == 0 and ey == 0 and ez == 0:
            continue

        in_bounds = ((X + ex >= 0) & (X + ex < nx) & (Y + ey >= 0) & (Y + ey < ny)
                     & (Z + ez >= 0) & (Z + ez < nz))
        neighbour_solid = np.roll(solid, (-ex, -ey, -ez), axis=(0, 1, 2))
        link = (~solid) & neighbour_solid & in_bounds
        link_index = np.nonzero(link)
        x_start = X[link_index]
        y_start = Y[link_index]
        z_start = Z[link_index]
        lower = np.zeros_like(x_start)
        upper = np.ones_like(x_start)

        # bisection: keep the crossing between lower (fluid) and upper (solid)
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            inside = phi(x_start + middle * ex, y_start + middle * ey, z_start + middle * ez) < 0.0
            upper = np.where(inside, middle, upper)
            lower = np.where(inside, lower, middle)

        q[d][link_index] = (0.5 * (lower + upper)).astype(np.float32)

    return q

def normals_from_sdf(phi, x, y, z, step=1e-4):
    """
    Outward unit normal grad(phi) / |grad(phi)| by central differences; x, y, z are 1D arrays of points.

    Returns an array (points, 3).
    """

    gradient_x = (phi(x + step, y, z) - phi(x - step, y, z)) / (2.0 * step)
    gradient_y = (phi(x, y + step, z) - phi(x, y - step, z)) / (2.0 * step)
    gradient_z = (phi(x, y, z + step) - phi(x, y, z - step)) / (2.0 * step)
    gradient_magnitude = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + gradient_z * gradient_z)

    return np.stack([gradient_x / gradient_magnitude, gradient_y / gradient_magnitude, gradient_z / gradient_magnitude], axis=-1)
