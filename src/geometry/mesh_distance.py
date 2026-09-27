# narrow-band unsigned distance from lattice nodes to a triangle mesh, one GPU thread per triangle
import numpy as np
import taichi as ti
from src.engine import runtime

@ti.func
def _closest_on_tri(p, a, b, c):
    # Ericson, Real-Time Collision Detection 5.1.5: closest point on triangle abc to p
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    res = a
    if d1 <= 0.0 and d2 <= 0.0:
        res = a # vertex region a
    else:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0.0 and d4 <= d3:
            res = b # vertex region b
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                res = a + (d1 / (d1 - d3)) * ab # edge ab
            else:
                cp = p - c
                d5 = ab.dot(cp)
                d6 = ac.dot(cp)
                if d6 >= 0.0 and d5 <= d6:
                    res = c # vertex region c
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        res = a + (d2 / (d2 - d6)) * ac # edge ac
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            res = b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b) # edge bc
                        else:
                            inv = 1.0 / (va + vb + vc)
                            res = a + ab * (vb * inv) + ac * (vc * inv) # face interior

    return res
    
@ti.kernel
def _band_distance(tris: ti.types.ndarray(), dist: ti.template(), band: ti.f32):
    for t in range(tris.shape[0]):
        a = ti.Vector([tris[t, 0, 0], tris[t, 0, 1], tris[t, 0, 2]])
        b = ti.Vector([tris[t, 1, 0], tris[t, 1, 1], tris[t, 1, 2]])
        c = ti.Vector([tris[t, 2, 0], tris[t, 2, 1], tris[t, 2, 2]])
        lo = ti.floor(ti.min(a, b, c) - band)
        hi = ti.ceil(ti.max(a, b, c) + band)
        i0 = ti.max(ti.cast(lo[0], ti.i32), 0)
        j0 = ti.max(ti.cast(lo[1], ti.i32), 0)
        k0 = ti.max(ti.cast(lo[2], ti.i32), 0)
        i1 = ti.min(ti.cast(hi[0], ti.i32), dist.shape[0] - 1)
        j1 = ti.min(ti.cast(hi[1], ti.i32), dist.shape[1] - 1)
        k1 = ti.min(ti.cast(hi[2], ti.i32), dist.shape[2] - 1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)])
                    e = p - _closest_on_tri(p, a, b, c)
                    d = ti.sqrt(e.dot(e))
                    if d <= band:
                        ti.atomic_min(dist[i, j, k], d)


def band_distance(tris, nx, ny, nz, band=3.0, backend="cpu"):
    # tris in lattice units (T, 3, 3); returns (nx, ny, nz) float32, inf outside the band
    runtime.init(backend)
    dist = ti.field(ti.f32, shape=(nx, ny, nz))
    dist.fill(np.inf)
    _band_distance(np.ascontiguousarray(tris, np.float32), dist, band)
    return dist.to_numpy()