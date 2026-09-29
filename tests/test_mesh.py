# mesh I/O and narrow-band distance: reduce to exact answers
import numpy as np
from src.geometry.mesh import read_stl, write_stl, icosphere, box_mesh
from src.geometry.mesh_distance import band_distance, sdf_from_mesh, q_from_mesh
from src.geometry.sdf import sdf_sphere, q_from_sdf
from src.geometry.wall_fraction import wall_fraction_sphere
from src.engine import lattice_d3q19 as L3

N = 24
C = (11.3, 12.1, 11.7)
R = 6.4


# test 1: binary STL round-trip preserves every vertex
def test_stl_roundtrip(tmp_path):
    tris = icosphere(C, R, subdiv=2)
    path = tmp_path / "sphere.stl"
    write_stl(path, tris)
    back = read_stl(path)
    assert back.shape == tris.shape
    assert np.allclose(back, tris, atol=1e-5)                 # float32 storage


# test 2: point-triangle distance against dense sampling of the triangle (independent check)
def test_band_distance_single_triangle():
    tri = np.array([[[4.2, 3.1, 5.0], [9.7, 4.4, 6.3], [5.5, 9.8, 4.1]]])
    dist = band_distance(tri, 14, 14, 14, band=3.0)
    u, v = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
    keep = (u + v) <= 1.0
    s = tri[0, 0] + u[keep, None] * (tri[0, 1] - tri[0, 0]) + v[keep, None] * (tri[0, 2] - tri[0, 0])
    X, Y, Z = np.meshgrid(np.arange(14), np.arange(14), np.arange(14), indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    ref = np.array([np.sqrt(((s - p) * (s - p)).sum(axis=1)).min() for p in pts]).reshape(14, 14, 14)
    inband = ref <= 2.9
    assert inband.sum() > 300
    assert np.all(dist[inband] <= ref[inband] + 1e-5)        # exact is never above a sampled point
    assert np.all(ref[inband] - dist[inband] < 0.03)          # and sampling spacing bounds the gap
    assert np.all(np.isinf(dist[ref > 3.1]))                  # nothing written outside the band


# test 3: icosphere distance matches | |x - c| - R | within the mesh's chordal error
def test_band_distance_icosphere():
    tris = icosphere(C, R, subdiv=4)
    dist = band_distance(tris, N, N, N, band=3.0)
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    dx = X - C[0]
    dy = Y - C[1]
    dz = Z - C[2]
    exact = np.abs(np.sqrt(dx * dx + dy * dy + dz * dz) - R)
    edge = np.sqrt(((tris[:, 0] - tris[:, 1]) * (tris[:, 0] - tris[:, 1])).sum(axis=1)).max()
    sag = R - np.sqrt(R * R - (edge * edge) / 3.0)             # max gap between flat face and sphere
    near = exact <= 2.5
    assert near.sum() > 1000
    assert np.all(np.abs(dist[near] - exact[near]) <= sag + 1e-4)

def _sphere_sag(tris):
    e = tris[:, 0] - tris[:, 1]
    edge = np.sqrt((e * e).sum(axis=1)).max()
    return R - np.sqrt(R * R - (edge * edge) / 3.0)

# test 4: inside/outside from ray parity matches the analytic sphere away from the facets
def test_mesh_sign_matches_sphere():
    tris = icosphere(C, R, subdiv=4)
    grid, phi = sdf_from_mesh(tris, N, N, N)
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")

    dx = X - C[0]
    dy = Y - C[1]
    dz = Z - C[2]

    s = np.sqrt(dx * dx + dy * dy + dz * dz) - R
    clear = np.abs(s) > _sphere_sag(tris) + 0.01

    assert (s[clear] < 0).sum() > 500 # nodes not inside the facet gap
    assert np.array_equal(grid[clear] < 0, s[clear] < 0) # plenty of interior nodes checked

# test 5: signed field and q from the mesh reduce to the analytic sphere
def test_mesh_phi_reduces_to_sphere():
    tris = icosphere(C, R, subdiv=4)
    grid, phi = sdf_from_mesh(tris, N, N, N)
    sag = _sphere_sag(tris)
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")

    exact = sdf_sphere(*C, R)(X, Y, Z)
    band = np.abs(exact) <= 2.0

    assert np.all(np.abs(grid[band] - exact[band]) <= sag + 1e-3)   # signed distance at nodes

    q_mesh = q_from_sdf(phi, N, N, N)
    q_ref = wall_fraction_sphere(N, N, N, *C, R)
    both = (q_mesh > 0) & (q_ref > 0)

    assert both.sum() > 0.99 * (q_ref > 0).sum()                    # same links, bar facet-gap nodes
    
    d, i, j, k = np.nonzero(both)
    E = L3.E[d].astype(np.float64)                                  # link vector of each boundary link
    P = np.stack([i, j, k], axis=1) + q_ref[both][:, None] * E      # exact wall point on the link
    n = (P - np.array(C)) / R                                       # sphere normal there
    cn = np.abs((E * n).sum(axis=1))                                # |c . n|, how steeply the link meets the wall
    dq = np.abs(q_mesh[both] - q_ref[both])
    interp = 3.0 / (8.0 * (R - np.sqrt(3.0)))                       # trilinear error bound: 3 axes x h^2/8 x curvature 1/r
    
    assert (dq * cn).max() < sag + interp                           # wall displacement along the normal
    assert np.median(dq) < 0.02                                     # typical link is essentially exact

def _dir(v):
    return int(np.nonzero((L3.E == np.array(v)).all(axis=1))[0][0])

# test 6: flat faces give exact q on straight and diagonal links
def test_q_from_mesh_box_exact():
    n = 16
    tris = box_mesh((5.3, 2.2, 2.4), (9.6, 13.7, 13.6))
    grid, phi = sdf_from_mesh(tris, n, n, n)
    q = q_from_mesh(tris, phi, n, n, n)
    px = _dir((1, 0, 0))
    pxy = _dir((1, 1, 0))
    mx = _dir((-1, 0, 0))
    inner = (slice(4, 11), slice(4, 11))
    assert np.allclose(q[px, 5][inner], 0.3, atol=1e-5)           # x = 5 -> face at 5.3
    assert np.allclose(q[pxy, 5][inner], 0.3, atol=1e-5)          # diagonal hits the same face at t = 0.3
    assert np.allclose(q[mx, 10][inner], 0.4, atol=1e-5)          # x = 10 -> face at 9.6 going -x


# test 7: exact mesh q on the icosphere is limited only by the facet sag
def test_q_from_mesh_icosphere():
    tris = icosphere(C, R, subdiv=4)
    grid, phi = sdf_from_mesh(tris, N, N, N)
    q_mesh = q_from_mesh(tris, phi, N, N, N)
    q_ref = wall_fraction_sphere(N, N, N, *C, R)
    sag = _sphere_sag(tris)
    both = (q_mesh > 0) & (q_ref > 0)
    assert both.sum() > 0.99 * (q_ref > 0).sum()
    d, i, j, k = np.nonzero(both)
    E = L3.E[d].astype(np.float64)
    P = np.stack([i, j, k], axis=1) + q_ref[both][:, None] * E
    n = (P - np.array(C)) / R
    cn = np.abs((E * n).sum(axis=1))
    dq = np.abs(q_mesh[both] - q_ref[both])
    assert (dq * cn).max() < sag + 1e-4                           # was sag + trilinear bound with q_from_sdf