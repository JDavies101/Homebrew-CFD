# mesh I/O and narrow-band distance: reduce to exact answers
import numpy as np
from src.geometry.mesh import read_stl, write_stl, icosphere, box_mesh
from src.geometry.mesh_distance import band_distance, sdf_from_mesh, q_from_mesh
from src.geometry.sdf import sdf_sphere, q_from_sdf
from src.geometry.wall_fraction import wall_fraction_sphere
from src.engine import lattice_d3q19 as d3q19

grid_size = 24
center = (11.3, 12.1, 11.7)  # off-lattice center so q takes many different values
radius = 6.4

def _sphere_sag(triangles):
    """
    Max gap between a flat icosphere face and the true sphere.

    Returns the sag in lattice units.
    """

    edge_vectors = triangles[:, 0] - triangles[:, 1]
    longest_edge = np.sqrt((edge_vectors * edge_vectors).sum(axis=1)).max()

    return radius - np.sqrt(radius * radius - (longest_edge * longest_edge) / 3.0)

def _direction_index(velocity):
    """
    Index of a D3Q19 lattice velocity.

    Returns the direction index.
    """

    return int(np.nonzero((d3q19.lattice_velocities == np.array(velocity)).all(axis=1))[0][0])

# test 1: binary STL round-trip preserves every vertex
def test_stl_round_trip(tmp_path):

    triangles = icosphere(center, radius, subdivisions=2)
    path = tmp_path / "sphere.stl"
    write_stl(path, triangles)
    read_back = read_stl(path)

    assert read_back.shape == triangles.shape
    assert np.allclose(read_back, triangles, atol=1e-5)  # float32 storage

# test 2: point-triangle distance against dense sampling of the triangle (independent check)
def test_band_distance_single_triangle():

    triangle = np.array([[[4.2, 3.1, 5.0], [9.7, 4.4, 6.3], [5.5, 9.8, 4.1]]])
    distance = band_distance(triangle, 14, 14, 14, band=3.0)
    u, v = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
    keep = (u + v) <= 1.0
    samples = triangle[0, 0] + u[keep, None] * (triangle[0, 1] - triangle[0, 0]) + v[keep, None] * (triangle[0, 2] - triangle[0, 0])
    X, Y, Z = np.meshgrid(np.arange(14), np.arange(14), np.arange(14), indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    sampled_distance = np.array([np.sqrt(((samples - node) * (samples - node)).sum(axis=1)).min() for node in nodes]).reshape(14, 14, 14)
    in_band = sampled_distance <= 2.9

    assert in_band.sum() > 300
    assert np.all(distance[in_band] <= sampled_distance[in_band] + 1e-5)  # exact is never above a sampled point
    assert np.all(sampled_distance[in_band] - distance[in_band] < 0.03)  # and sampling spacing bounds the gap
    assert np.all(np.isinf(distance[sampled_distance > 3.1]))  # nothing written outside the band

# test 3: icosphere distance matches | |x - c| - R | within the mesh's chordal error
def test_band_distance_icosphere():

    triangles = icosphere(center, radius, subdivisions=4)
    distance = band_distance(triangles, grid_size, grid_size, grid_size, band=3.0)
    X, Y, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing="ij")
    offset_x = X - center[0]
    offset_y = Y - center[1]
    offset_z = Z - center[2]
    exact = np.abs(np.sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z) - radius)
    sag = _sphere_sag(triangles)
    near = exact <= 2.5

    assert near.sum() > 1000
    assert np.all(np.abs(distance[near] - exact[near]) <= sag + 1e-4)

# test 4: inside/outside from ray parity matches the analytic sphere away from the facets
def test_mesh_sign_matches_sphere():

    triangles = icosphere(center, radius, subdivisions=4)
    grid, phi = sdf_from_mesh(triangles, grid_size, grid_size, grid_size)
    X, Y, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing="ij")
    offset_x = X - center[0]
    offset_y = Y - center[1]
    offset_z = Z - center[2]
    exact = np.sqrt(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z) - radius
    # nodes not inside the facet gap
    clear = np.abs(exact) > _sphere_sag(triangles) + 0.01

    assert (exact[clear] < 0).sum() > 500  # plenty of interior nodes checked
    assert np.array_equal(grid[clear] < 0, exact[clear] < 0)

# test 5: signed field and q from the mesh reduce to the analytic sphere
def test_mesh_phi_reduces_to_sphere():

    triangles = icosphere(center, radius, subdivisions=4)
    grid, phi = sdf_from_mesh(triangles, grid_size, grid_size, grid_size)
    sag = _sphere_sag(triangles)
    X, Y, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing="ij")
    exact = sdf_sphere(*center, radius)(X, Y, Z)
    band = np.abs(exact) <= 2.0

    q_mesh = q_from_sdf(phi, grid_size, grid_size, grid_size)
    q_reference = wall_fraction_sphere(grid_size, grid_size, grid_size, *center, radius)
    both = (q_mesh > 0) & (q_reference > 0)

    d, i, j, k = np.nonzero(both)
    link_vectors = d3q19.lattice_velocities[d].astype(np.float64)  # link vector of each boundary link
    wall_points = np.stack([i, j, k], axis=1) + q_reference[both][:, None] * link_vectors  # exact wall point on the link
    normals = (wall_points - np.array(center)) / radius  # sphere normal there
    link_normal_cosine = np.abs((link_vectors * normals).sum(axis=1))  # |c . n|, how steeply the link meets the wall
    q_error = np.abs(q_mesh[both] - q_reference[both])
    trilinear_bound = 3.0 / (8.0 * (radius - np.sqrt(3.0)))  # trilinear error bound: 3 axes x h^2/8 x curvature 1/r

    assert np.all(np.abs(grid[band] - exact[band]) <= sag + 1e-3)  # signed distance at nodes
    assert both.sum() > 0.99 * (q_reference > 0).sum()  # same links, bar facet-gap nodes
    assert (q_error * link_normal_cosine).max() < sag + trilinear_bound  # wall displacement along the normal
    assert np.median(q_error) < 0.02  # typical link is essentially exact

# test 6: flat faces give exact q on straight and diagonal links
def test_q_from_mesh_box_exact():

    box_grid_size = 16
    triangles = box_mesh((5.3, 2.2, 2.4), (9.6, 13.7, 13.6))
    grid, phi = sdf_from_mesh(triangles, box_grid_size, box_grid_size, box_grid_size)
    q = q_from_mesh(triangles, phi, box_grid_size, box_grid_size, box_grid_size)
    plus_x = _direction_index((1, 0, 0))
    plus_x_plus_y = _direction_index((1, 1, 0))
    minus_x = _direction_index((-1, 0, 0))
    inner = (slice(4, 11), slice(4, 11))

    assert np.allclose(q[plus_x, 5][inner], 0.3, atol=1e-5)  # x = 5 -> face at 5.3
    assert np.allclose(q[plus_x_plus_y, 5][inner], 0.3, atol=1e-5)  # diagonal hits the same face at t = 0.3
    assert np.allclose(q[minus_x, 10][inner], 0.4, atol=1e-5)  # x = 10 -> face at 9.6 going -x

# test 7: exact mesh q on the icosphere is limited only by the facet sag
def test_q_from_mesh_icosphere():

    triangles = icosphere(center, radius, subdivisions=4)
    grid, phi = sdf_from_mesh(triangles, grid_size, grid_size, grid_size)
    q_mesh = q_from_mesh(triangles, phi, grid_size, grid_size, grid_size)
    q_reference = wall_fraction_sphere(grid_size, grid_size, grid_size, *center, radius)
    sag = _sphere_sag(triangles)
    both = (q_mesh > 0) & (q_reference > 0)
    d, i, j, k = np.nonzero(both)
    link_vectors = d3q19.lattice_velocities[d].astype(np.float64)
    wall_points = np.stack([i, j, k], axis=1) + q_reference[both][:, None] * link_vectors
    normals = (wall_points - np.array(center)) / radius
    link_normal_cosine = np.abs((link_vectors * normals).sum(axis=1))
    q_error = np.abs(q_mesh[both] - q_reference[both])

    assert both.sum() > 0.99 * (q_reference > 0).sum()
    assert (q_error * link_normal_cosine).max() < sag + 1e-4  # no trilinear term: q comes from the exact mesh hit
