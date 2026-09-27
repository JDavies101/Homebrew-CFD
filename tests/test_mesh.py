# mesh I/O and narrow-band distance: reduce to exact answers
import numpy as np
from src.geometry.mesh import read_stl, write_stl, icosphere
from src.geometry.mesh_distance import band_distance

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