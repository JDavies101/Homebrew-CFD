# narrow-band unsigned distance from lattice nodes to a triangle mesh, one GPU thread per triangle
import numpy as np
import taichi as ti
from src.engine import runtime
from src.engine import lattice_d3q19 as d3q19
from src.geometry.sdf import q_from_sdf

@ti.func
def _closest_on_tri(p, a, b, c):
    """
    Closest point on triangle abc to p (Ericson, Real-Time Collision Detection 5.1.5; d1-d6, va-vc as in the book).

    Returns the closest point.
    """

    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    closest = a
    if d1 <= 0.0 and d2 <= 0.0:
        closest = a  # vertex region a
    else:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            closest = b  # vertex region b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                closest = a + (d1 / (d1 - d3)) * ab  # edge ab
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    closest = c  # vertex region c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        closest = a + (d2 / (d2 - d6)) * ac  # edge ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            closest = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)  # edge bc
                        else:
                            inverse_denominator = 1.0 / (va + vb + vc)
                            closest = a + ab * (vb * inverse_denominator) + ac * (vc * inverse_denominator)  # face interior

    return closest

@ti.kernel
def _band_distance(triangles: ti.types.ndarray(), distance: ti.template(), band: ti.f32):
    """
    Min distance from every node within band of a triangle to that triangle (atomic min over triangles).
    """

    for t in range(triangles.shape[0]):
        a = ti.Vector([triangles[t, 0, 0], triangles[t, 0, 1], triangles[t, 0, 2]])
        b = ti.Vector([triangles[t, 1, 0], triangles[t, 1, 1], triangles[t, 1, 2]])
        c = ti.Vector([triangles[t, 2, 0], triangles[t, 2, 1], triangles[t, 2, 2]])
        # node box around the triangle, grown by the band and clamped to the domain
        box_lower = ti.floor(ti.min(a, b, c) - band)
        box_upper = ti.ceil(ti.max(a, b, c) + band)
        i_start = ti.max(ti.cast(box_lower[0], ti.i32), 0)
        j_start = ti.max(ti.cast(box_lower[1], ti.i32), 0)
        k_start = ti.max(ti.cast(box_lower[2], ti.i32), 0)
        i_end = ti.min(ti.cast(box_upper[0], ti.i32), distance.shape[0] - 1)
        j_end = ti.min(ti.cast(box_upper[1], ti.i32), distance.shape[1] - 1)
        k_end = ti.min(ti.cast(box_upper[2], ti.i32), distance.shape[2] - 1)
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    node = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)])
                    offset = node - _closest_on_tri(node, a, b, c)
                    squared_distance = offset.dot(offset)
                    if squared_distance <= band * band:
                        ti.atomic_min(distance[i, j, k], ti.sqrt(squared_distance))

def band_distance(triangles, nx, ny, nz, band=3.0, backend="cpu"):
    """
    Unsigned distance to a mesh (triangles in lattice units, (T, 3, 3)) on nodes within band of it.

    Returns float32 (nx, ny, nz), inf outside the band.
    """

    runtime.init(backend)
    distance = ti.field(ti.f32, shape=(nx, ny, nz))
    distance.fill(np.inf)
    _band_distance(np.ascontiguousarray(triangles, np.float32), distance, band)

    return distance.to_numpy()

@ti.kernel
def _ray_crossings(triangles: ti.types.ndarray(), crossings: ti.template(), ray_offset_y: ti.f32, ray_offset_z: ti.f32):
    """
    Cast +x rays along grid lines (j + ray_offset_y, k + ray_offset_z); mark the first node past each triangle crossing.
    """

    for t in range(triangles.shape[0]):
        a = ti.Vector([triangles[t, 0, 0], triangles[t, 0, 1], triangles[t, 0, 2]])
        b = ti.Vector([triangles[t, 1, 0], triangles[t, 1, 1], triangles[t, 1, 2]])
        c = ti.Vector([triangles[t, 2, 0], triangles[t, 2, 1], triangles[t, 2, 2]])

        # triangle edges projected onto the y-z plane
        ab_y = b[1] - a[1]
        ab_z = b[2] - a[2]
        ac_y = c[1] - a[1]
        ac_z = c[2] - a[2]
        determinant = ab_y * ac_z - ac_y * ab_z

        # skip triangles edge-on to the rays
        if determinant != 0.0:
            j_start = ti.max(ti.cast(ti.ceil(ti.min(a[1], b[1], c[1]) - ray_offset_y), ti.i32), 0)
            j_end = ti.min(ti.cast(ti.floor(ti.max(a[1], b[1], c[1]) - ray_offset_y), ti.i32), crossings.shape[1] - 1)
            k_start = ti.max(ti.cast(ti.ceil(ti.min(a[2], b[2], c[2]) - ray_offset_z), ti.i32), 0)
            k_end = ti.min(ti.cast(ti.floor(ti.max(a[2], b[2], c[2]) - ray_offset_z), ti.i32), crossings.shape[2] - 1)
            inverse_determinant = 1.0 / determinant

            for j in range(j_start, j_end + 1):
                for k in range(k_start, k_end + 1):
                    ray_y = ti.cast(j, ti.f32) + ray_offset_y - a[1]
                    ray_z = ti.cast(k, ti.f32) + ray_offset_z - a[2]
                    weight_b = (ray_y * ac_z - ac_y * ray_z) * inverse_determinant
                    weight_c = (ab_y * ray_z - ray_y * ab_z) * inverse_determinant
                    weight_a = 1.0 - weight_b - weight_c

                    if weight_a >= 0.0 and weight_b >= 0.0 and weight_c >= 0.0:
                        crossing_x = weight_a * a[0] + weight_b * b[0] + weight_c * c[0]
                        i = ti.max(ti.cast(ti.ceil(crossing_x), ti.i32), 0)

                        if i < crossings.shape[0]:
                            ti.atomic_add(crossings[i, j, k], 1)

def trilinear(grid):
    """
    Wrap a node-sampled field for evaluation at arbitrary points (arrays or scalars).

    Returns a function phi(x, y, z).
    """

    nx, ny, nz = grid.shape

    def phi(x, y, z):
        x = np.clip(np.asarray(x, np.float64), 0.0, nx - 1.0)
        y = np.clip(np.asarray(y, np.float64), 0.0, ny - 1.0)
        z = np.clip(np.asarray(z, np.float64), 0.0, nz - 1.0)

        # lower corner of the enclosing cell
        i = np.minimum(np.floor(x).astype(np.int64), nx - 2)
        j = np.minimum(np.floor(y).astype(np.int64), ny - 2)
        k = np.minimum(np.floor(z).astype(np.int64), nz - 2)

        # weights toward the upper (fraction) and lower (remainder) corners
        fraction_x = x - i
        fraction_y = y - j
        fraction_z = z - k
        remainder_x = 1.0 - fraction_x
        remainder_y = 1.0 - fraction_y
        remainder_z = 1.0 - fraction_z

        return (grid[i, j, k] * remainder_x * remainder_y * remainder_z + grid[i + 1, j, k] * fraction_x * remainder_y * remainder_z
                + grid[i, j + 1, k] * remainder_x * fraction_y * remainder_z + grid[i, j, k + 1] * remainder_x * remainder_y * fraction_z
                + grid[i + 1, j + 1, k] * fraction_x * fraction_y * remainder_z + grid[i + 1, j, k + 1] * fraction_x * remainder_y * fraction_z
                + grid[i, j + 1, k + 1] * remainder_x * fraction_y * fraction_z + grid[i + 1, j + 1, k + 1] * fraction_x * fraction_y * fraction_z)

    return phi

def sdf_from_mesh(triangles, nx, ny, nz, band=3.0, backend="cpu"):
    """
    Signed distance for a closed triangle mesh in lattice units: phi > 0 fluid, < 0 inside the mesh.

    Returns (phi_grid, phi): the float32 node grid and its trilinear interpolant.
    """

    runtime.init(backend)
    distance = band_distance(triangles, nx, ny, nz, band, backend)
    crossings = ti.field(ti.i32, shape=(nx, ny, nz))

    # odd crossing count along +x means inside; offsets keep rays off edges and vertices
    _ray_crossings(np.ascontiguousarray(triangles, np.float32), crossings, 0.00137, 0.00271)
    inside = (np.cumsum(crossings.to_numpy(), axis=0) % 2) == 1
    magnitude = np.where(np.isfinite(distance), distance, band + 1.0)
    grid = np.where(inside, -magnitude, magnitude).astype(np.float32)

    return grid, trilinear(grid)

@ti.kernel
def _link_hits(triangles: ti.types.ndarray(), lattice_velocities: ti.types.ndarray(), q: ti.template()):
    """
    Nearest segment-triangle hit t in [0, 1] for every lattice link near each triangle (Moller-Trumbore).
    """

    for m in range(triangles.shape[0]):
        a = ti.Vector([triangles[m, 0, 0], triangles[m, 0, 1], triangles[m, 0, 2]])
        b = ti.Vector([triangles[m, 1, 0], triangles[m, 1, 1], triangles[m, 1, 2]])
        c = ti.Vector([triangles[m, 2, 0], triangles[m, 2, 1], triangles[m, 2, 2]])

        edge_1 = b - a
        edge_2 = c - a

        # node box around the triangle, grown by one link and clamped to the domain
        box_lower = ti.floor(ti.min(a, b, c)) - 1.0
        box_upper = ti.ceil(ti.max(a, b, c)) + 1.0

        i_start = ti.max(ti.cast(box_lower[0], ti.i32), 0)
        j_start = ti.max(ti.cast(box_lower[1], ti.i32), 0)
        k_start = ti.max(ti.cast(box_lower[2], ti.i32), 0)

        i_end = ti.min(ti.cast(box_upper[0], ti.i32), q.shape[1] - 1)
        j_end = ti.min(ti.cast(box_upper[1], ti.i32), q.shape[2] - 1)
        k_end = ti.min(ti.cast(box_upper[2], ti.i32), q.shape[3] - 1)

        # direction terms depend only on (triangle, direction): compute once, then sweep the nodes
        for d in range(1, lattice_velocities.shape[0]):
            link = ti.Vector([ti.cast(lattice_velocities[d, 0], ti.f32), ti.cast(lattice_velocities[d, 1], ti.f32), ti.cast(lattice_velocities[d, 2], ti.f32)])
            # Moller-Trumbore: barycentric u, v and link parameter t
            link_cross_edge_2 = link.cross(edge_2)
            determinant = edge_1.dot(link_cross_edge_2)

            if ti.abs(determinant) > 1e-12:
                inverse_determinant = 1.0 / determinant

                for i in range(i_start, i_end + 1):
                    for j in range(j_start, j_end + 1):
                        for k in range(k_start, k_end + 1):
                            node = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)])
                            node_from_a = node - a
                            u = node_from_a.dot(link_cross_edge_2) * inverse_determinant

                            if u >= 0.0 and u <= 1.0:
                                node_cross_edge_1 = node_from_a.cross(edge_1)
                                v = link.dot(node_cross_edge_1) * inverse_determinant

                                if v >= 0.0 and u + v <= 1.0:
                                    t = edge_2.dot(node_cross_edge_1) * inverse_determinant

                                    if t >= 0.0 and t <= 1.0:
                                        ti.atomic_min(q[d, i, j, k], t)

def q_from_mesh(triangles, phi, nx, ny, nz, backend="cpu", periodic=(False, False, False)):
    """
    Exact Bouzidi fractions for a triangle mesh; phi (from sdf_from_mesh) defines which links are boundary links.

    Returns float32 q (19, nx, ny, nz).
    """

    runtime.init(backend)

    hit_fraction = ti.field(ti.f32, shape=(d3q19.direction_count, nx, ny, nz))
    hit_fraction.fill(2.0)

    _link_hits(np.ascontiguousarray(triangles, np.float32), np.ascontiguousarray(d3q19.lattice_velocities, np.int32), hit_fraction)

    mesh_hits = hit_fraction.to_numpy()
    q_sdf = q_from_sdf(phi, nx, ny, nz, periodic=periodic)
    boundary_link = q_sdf > 0.0
    hit = mesh_hits <= 1.0
    q = np.where(boundary_link & hit, np.maximum(mesh_hits, 1e-6), q_sdf)

    return q.astype(np.float32)
