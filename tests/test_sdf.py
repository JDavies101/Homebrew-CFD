# SDF geometry: q and normals reduce to the analytic sphere results
import numpy as np
from src.geometry.sdf import sdf_sphere, solid_from_sdf, q_from_sdf, normals_from_sdf, node_values
from src.geometry.wall_fraction import wall_fraction_sphere
from src.engine.simulation3d import Simulation3D
from src.geometry.airfoil import naca_four_digit, place_section, extruded_section_sdf
from src.engine import lattice_d3q19 as d3q19

grid_size = 24
center = (11.3, 12.1, 11.7)  # off-lattice center so q takes many different values
radius = 6.4
# test 1: q from the SDF matches the analytic sphere q
def test_q_from_sdf_matches_analytic_sphere():

    phi = sdf_sphere(*center, radius)
    q_sdf = q_from_sdf(phi, grid_size, grid_size, grid_size)
    q_reference = wall_fraction_sphere(grid_size, grid_size, grid_size, *center, radius)

    assert (q_reference > 0).sum() > 500  # plenty of links, not a vacuous compare
    assert np.array_equal(q_sdf > 0, q_reference > 0)  # same set of boundary links
    assert np.allclose(q_sdf, q_reference, atol=1e-5)  # same crossing points
    assert q_sdf.min() >= 0.0 and q_sdf.max() <= 1.0

# test 2: solid from the SDF matches the sphere mask
def test_solid_from_sdf_matches_sphere_mask():

    phi = sdf_sphere(*center, radius)
    X, Y, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing="ij")
    reference_mask = ((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2 < radius ** 2).astype(np.int32)

    assert np.array_equal(solid_from_sdf(phi, grid_size, grid_size, grid_size), reference_mask)

# test 3: normals from the SDF are radial
def test_normals_from_sdf_are_radial():

    phi = sdf_sphere(*center, radius)
    rng = np.random.default_rng(1)
    points = rng.normal(size=(200, 3))
    points = np.array(center) + (radius + 0.5) * points / np.linalg.norm(points, axis=1, keepdims=True)  # points just outside
    normals = normals_from_sdf(phi, points[:, 0], points[:, 1], points[:, 2])
    exact = (points - np.array(center)) / np.linalg.norm(points - np.array(center), axis=1, keepdims=True)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert np.allclose(normals, exact, atol=1e-6)

# test 4: wall list from the SDF has exact wall distances and radial normals
def test_wall_list_from_sdf_sphere():

    phi = sdf_sphere(*center, radius)
    sim = Simulation3D(grid_size, grid_size, grid_size, "cpu")
    sim.solid.from_numpy(solid_from_sdf(phi, grid_size, grid_size, grid_size))
    sim.build_wall_list(phi=phi)
    wall_nodes = sim.wall_ijk.to_numpy().astype(np.float64)
    wall_distance = sim.wall_y1.to_numpy()
    wall_normals = sim.wall_n.to_numpy()
    offset = wall_nodes - np.array(center)
    node_radius = np.sqrt((offset * offset).sum(axis=1))

    assert sim.n_wall > 200
    assert np.allclose(wall_distance, node_radius - radius, atol=1e-5)  # exact distance to the sphere
    assert wall_distance.min() > 0.0 and wall_distance.max() <= np.sqrt(3.0) + 1e-6  # first fluid layer only
    assert np.allclose(np.abs((wall_normals * offset / node_radius[:, None]).sum(axis=1)), 1.0, atol=1e-4)  # radial (either sign)

# test 5: extruded node values equal the full-grid evaluation exactly
def test_extruded_node_values_exact():

    polygon_x, polygon_y = naca_four_digit("4412")
    placed_x, placed_y = place_section(polygon_x, polygon_y, 30.0, 4.0, 20.0, 8.5)
    phi = extruded_section_sdf(placed_x, placed_y)
    plane_values = node_values(phi, 70, 30, 5, extruded=True)
    full_values = node_values(phi, 70, 30, 5)

    assert plane_values.shape == (70, 30, 5)
    assert np.array_equal(plane_values, full_values)

# test 6: precomputed node values give the identical solid mask and q
def test_node_phi_reuse_identical():

    polygon_x, polygon_y = naca_four_digit("4412")
    placed_x, placed_y = place_section(polygon_x, polygon_y, 30.0, 4.0, 20.0, 8.5)
    phi = extruded_section_sdf(placed_x, placed_y)
    node_phi = node_values(phi, 70, 30, 5, extruded=True)

    assert np.array_equal(solid_from_sdf(phi, 70, 30, 5, node_phi), solid_from_sdf(phi, 70, 30, 5))
    assert np.array_equal(q_from_sdf(phi, 70, 30, 5, node_phi=node_phi), q_from_sdf(phi, 70, 30, 5))

# test 7: the shared path also holds for a full 3D SDF (sphere)
def test_node_phi_reuse_sphere():

    phi = sdf_sphere(*center, radius)
    node_phi = node_values(phi, grid_size, grid_size, grid_size)

    assert np.array_equal(q_from_sdf(phi, grid_size, grid_size, grid_size, node_phi=node_phi), q_from_sdf(phi, grid_size, grid_size, grid_size))

# test 8: links across a periodic z face are found; a closed face drops them
def test_periodic_links_across_face():

    nx, ny, nz = 12, 12, 4

    # z-invariant slab y < 5.3: every fluid node beside it links in all directions on every layer
    def phi(x, y, z):

        return y - 5.3

    q_periodic = q_from_sdf(phi, nx, ny, nz, periodic=(False, False, True))
    q_closed = q_from_sdf(phi, nx, ny, nz)
    plus_z_diagonal = int(np.where((d3q19.lattice_velocities == [0, -1, 1]).all(axis=1))[0][0])

    assert q_periodic[plus_z_diagonal, 6, 6, nz - 1] > 0.0
    assert q_closed[plus_z_diagonal, 6, 6, nz - 1] == 0.0
    assert np.allclose(q_periodic[plus_z_diagonal, :, 6, :], q_periodic[plus_z_diagonal, :, 6, :1])