# signed distance fields: phi > 0 fluid, phi < 0 solid, phi = 0 on the surface
# one phi gives the solid mask, per-link Bouzidi fractions q, wall normals, and wall distance
import numpy as np
from src.engine import lattice_d3q19 as d3q19

def sdf_sphere(center_x, center_y, center_z, radius):
    """
    Signed distance to a sphere.

    Returns a function phi(x, y, z) that works on scalars or whole arrays.
    """

    def phi(x, y, z):
        offset_x = x - center_x
        offset_y = y - center_y
        offset_z = z - center_z

        return np.sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z) - radius

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

def node_values(phi, nx, ny, nz, extruded=False):
    """
    phi at every lattice node, evaluated once and shared by solid_from_sdf and q_from_sdf.
    extruded=True for an SDF that ignores z: one xy-plane is evaluated and repeated along z (nz times cheaper).

    Returns float64 (nx, ny, nz); a read-only broadcast view when extruded.
    """

    if extruded:
        plane_x, plane_y = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64), indexing="ij")
        plane = phi(plane_x, plane_y, 0.0)

        return np.broadcast_to(plane[:, :, None], (nx, ny, nz))

    X, Y, Z = node_grid(nx, ny, nz)

    return phi(X, Y, Z)

def solid_from_sdf(phi, nx, ny, nz, node_phi=None):
    """
    Solid mask from the sign of phi at the nodes; node_phi (from node_values) skips re-evaluating phi.

    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    if node_phi is None:
        node_phi = node_values(phi, nx, ny, nz)

    return (node_phi < 0.0).astype(np.int32)

def q_from_sdf(phi, nx, ny, nz, iterations=40, node_phi=None, periodic=(False, False, False), thin_samples=16, thin_walls=True):
    """
    Bouzidi wall fraction q on every link the surface crosses, by bisection of phi along the link: fluid-to-solid
    links, and thin-wall links whose ends are both fluid but whose interior dips inside (a sub-cell trailing edge);
    links cross a periodic face to the wrapped neighbour, as streaming does.
    thin_walls=False keeps only fluid-to-solid links (the analytic-q comparisons).

    Returns float32 q (19, nx, ny, nz), 0 on non-boundary links.
    """

    if node_phi is None:
        node_phi = node_values(phi, nx, ny, nz)
    solid = node_phi < 0.0
    q = np.zeros((d3q19.direction_count, nx, ny, nz), np.float32)
    sizes = (nx, ny, nz)

    for d in range(d3q19.direction_count):
        ex, ey, ez = (int(v) for v in d3q19.lattice_velocities[d])
        if ex == 0 and ey == 0 and ez == 0:
            continue

        # neighbour_solid[i, j, k] = solid[i + ex, j + ey, k + ez]: wraps on periodic axes, False past a closed face
        neighbour_solid = np.roll(solid, (-ex, -ey, -ez), axis=(0, 1, 2))
        neighbour_phi = np.roll(node_phi, (-ex, -ey, -ez), axis=(0, 1, 2))
        neighbour_inside = np.ones_like(solid)
        for axis, step in enumerate((ex, ey, ez)):
            if step != 0 and not periodic[axis]:
                face = [slice(None)] * 3
                face[axis] = sizes[axis] - 1 if step > 0 else 0
                neighbour_solid[tuple(face)] = False
                neighbour_inside[tuple(face)] = False
        link = (~solid) & neighbour_solid
        link_index = np.nonzero(link)

        # thin walls: both ends fluid, but phi_A + phi_B < |c| leaves room for a surface between them;
        # sample phi inside the link and bracket the first crossing
        link_length = np.sqrt(ex * ex + ey * ey + ez * ez)
        candidate = (~solid) & (~neighbour_solid) & neighbour_inside & (node_phi + neighbour_phi < link_length) & thin_walls
        candidate_index = np.nonzero(candidate)
        samples = (np.arange(thin_samples) + 1.0) / (thin_samples + 1.0)
        sample_x = candidate_index[0][:, None] + samples[None, :] * ex
        sample_y = candidate_index[1][:, None] + samples[None, :] * ey
        sample_z = candidate_index[2][:, None] + samples[None, :] * ez
        sample_x = np.mod(sample_x, nx) if periodic[0] else sample_x
        sample_y = np.mod(sample_y, ny) if periodic[1] else sample_y
        sample_z = np.mod(sample_z, nz) if periodic[2] else sample_z
        sample_inside = phi(sample_x.ravel(), sample_y.ravel(), sample_z.ravel()).reshape(sample_x.shape) < 0.0
        crossed = sample_inside.any(axis=1)
        first = np.argmax(sample_inside, axis=1)[crossed]
        thin_index = tuple(axis_index[crossed] for axis_index in candidate_index)
        thin_lower = np.where(first > 0, samples[first - 1], 0.0)
        thin_upper = samples[first]

        # one bisection for both kinds: keep the crossing between lower (fluid) and upper (inside)
        index = tuple(np.concatenate([solid_link, thin_link]) for solid_link, thin_link in zip(link_index, thin_index))
        lower = np.concatenate([np.zeros(len(link_index[0])), thin_lower])
        upper = np.concatenate([np.ones(len(link_index[0])), thin_upper])
        x_start = index[0].astype(np.float64)
        y_start = index[1].astype(np.float64)
        z_start = index[2].astype(np.float64)

        # periodic axes read phi wrapped
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            x = x_start + middle * ex
            y = y_start + middle * ey
            z = z_start + middle * ez
            x = np.mod(x, nx) if periodic[0] else x
            y = np.mod(y, ny) if periodic[1] else y
            z = np.mod(z, nz) if periodic[2] else z
            inside = phi(x, y, z) < 0.0
            upper = np.where(inside, middle, upper)
            lower = np.where(inside, lower, middle)

        q[d][index] = (0.5 * (lower + upper)).astype(np.float32)

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
