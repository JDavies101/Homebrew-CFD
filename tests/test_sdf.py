# SDF geometry: q and normals reduce to the analytic sphere results
import numpy as np
from src.geometry.sdf import sdf_sphere, solid_from_sdf, q_from_sdf, normals_from_sdf
from src.geometry.wall_fraction import wall_fraction_sphere

N = 24
C = (11.3, 12.1, 11.7) # off-lattice centre so q takes many different values
R = 6.4

# test 1: q and normals reduce to the analytic sphere results
def test_q_from_sdf_matches_analytic_sphere():
    phi = sdf_sphere(*C, R)
    q_sdf = q_from_sdf(phi, N, N, N)
    q_ref = wall_fraction_sphere(N, N, N, *C, R)

    assert (q_ref > 0).sum() > 500 # plenty of links, not a vacuous compare
    assert np.array_equal(q_sdf > 0, q_ref > 0) # same set of boundary links
    assert np.allclose(q_sdf, q_ref, atol=1e-5) # same crossing points
    assert q_sdf.min() >= 0.0 and q_sdf.max() <= 1.0 

# test 2: solid boundary matches solid mask
def test_solid_from_sdf_matches_sphere_mask():
    phi = sdf_sphere(*C, R)
    X, Y, Z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    ref = ((X - C[0]) ** 2 + (Y - C[1]) ** 2 + (Z - C[2]) ** 2 < R ** 2).astype(np.int32)

    assert np.array_equal(solid_from_sdf(phi, N, N, N), ref)

# test 3: normals from sdf are radial
def test_normals_from_sdf_are_radial():
    phi = sdf_sphere(*C,R)
    rng = np.random.default_rng(1)
    p = rng.normal(size=(200, 3))
    p = np.array(C) + (R + 0.5) * p / np.linalg.norm(p, axis=1, keepdims=True) # points just outside
    n = normals_from_sdf(phi, p[:,0], p[:,1], p[:,2])
    exact = (p - np.array(C)) / np.linalg.norm(p - np.array(C), axis=1, keepdims=True)

    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    assert np.allclose(n, exact, atol=1e-6)