# narrow-band unsigned distance from lattice nodes to a triangle mesh, one GPU thread per triangle
import numpy as np
import taichi as ti
from src.engine import runtime
from src.engine import lattice3d as L3
from src.geometry.sdf import q_from_sdf

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

@ti.kernel
def _ray_crossings(tris: ti.types.ndarray(), cross: ti.template(), ey: ti.f32, ez: ti.f32):
    # +x rays along grid lines (j + ey, k + ez); marks the first node past each triangle crossing
    for t in range(tris.shape[0]):
        a = ti.Vector([tris[t, 0, 0], tris[t, 0, 1], tris[t, 0, 2]])
        b = ti.Vector([tris[t, 1, 0], tris[t, 1, 1], tris[t, 1, 2]])
        c = ti.Vector([tris[t, 2, 0], tris[t, 2, 1], tris[t, 2, 2]])

        by_ = b[1] - a[1]
        bz_ = b[2] - a[2]
        cy_ = c[1] - a[1]
        cz_ = c[2] - a[2]
        det = by_ * cz_ - cy_ * bz_

        if det != 0.0: # skip triangles edge-on to the rays
            j0 = ti.max(ti.cast(ti.ceil(ti.min(a[1], b[1], c[1]) - ey), ti.i32), 0)
            j1 = ti.min(ti.cast(ti.floor(ti.max(a[1], b[1], c[1]) - ey), ti.i32), cross.shape[1] - 1)
            k0 = ti.max(ti.cast(ti.ceil(ti.min(a[2], b[2], c[2]) - ez), ti.i32), 0)
            k1 = ti.min(ti.cast(ti.floor(ti.max(a[2], b[2], c[2]) - ez), ti.i32), cross.shape[2] - 1)
            inv = 1.0 / det
            
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    py = ti.cast(j, ti.f32) + ey - a[1]
                    pz = ti.cast(k, ti.f32) + ez - a[2]
                    w1 = (py * cz_ - cy_ * pz) * inv # weight of b
                    w2 = (by_ * pz - py * bz_) * inv # weight of c
                    w0 = 1.0 - w1 - w2 # weight of a

                    if w0 >= 0.0 and w1 >= 0.0 and w2 >=0.0:
                        x = w0 * a[0] + w1 * b[0] + w2 * c[0]
                        i = ti.max(ti.cast(ti.ceil(x), ti.i32), 0)

                        if i < cross.shape[0]:
                            ti.atomic_add(cross[i, j, k], 1)

def trilinear(grid):
    # wrap a node-sampled field as phi(x, y, z) for arbitrary points (arrays or scalars)
    nx, ny, nz = grid.shape

    def phi(x, y, z):
        x = np.clip(np.asarray(x, np.float64), 0.0, nx - 1.0)
        y = np.clip(np.asarray(y, np.float64), 0.0, ny - 1.0)
        z = np.clip(np.asarray(z, np.float64), 0.0, nz - 1.0)

        i = np.minimum(np.floor(x).astype(np.int64), nx - 2)
        j = np.minimum(np.floor(y).astype(np.int64), ny - 2)
        k = np.minimum(np.floor(z).astype(np.int64), nz - 2)

        fx = x - i
        fy = y - j
        fz = z - k

        gx = 1.0 - fx
        gy = 1.0 - fy
        gz = 1.0 - fz

        return (grid[i, j, k] * gx * gy * gz + grid[i + 1, j, k] * fx * gy * gz
                + grid[i, j + 1, k] * gx * fy * gz + grid[i, j, k + 1] * gx * gy * fz
                + grid[i + 1, j + 1, k] * fx * fy * gz + grid[i + 1, j, k + 1] * fx * gy * fz
                + grid[i, j + 1, k + 1] * gx * fy * fz + grid[i + 1, j + 1, k + 1] * fx * fy * fz)
    return phi

def sdf_from_mesh(tris, nx, ny, nz, band=3.0, backend="cpu"):
    # closed triangle mesh (lattice units) -> (phi_grid, phi): phi > 0 fluid, < 0 inside the mesh
    runtime.init(backend)
    dist = band_distance(tris, nx, ny, nz, band, backend)
    cross = ti.field(ti.i32, shape=(nx, ny, nz))

    _ray_crossings(np.ascontiguousarray(tris, np.float32), cross, 0.00137, 0.00271)
    inside = (np.cumsum(cross.to_numpy(), axis=0) % 2) == 1
    mag = np.where(np.isfinite(dist), dist, band + 1.0)
    grid = np.where(inside, -mag, mag).astype(np.float32)

    return grid, trilinear(grid)

@ti.kernel
def _link_hits(tris: ti.types.ndarray(), E:ti.types.ndarray(), q: ti.template()):
    # nearest segment-triangle hit t in [0, 1] for every lattice link near each triangle (Moller-Trumbore)
    for m in range(tris.shape[0]):
        a = ti.Vector([tris[m, 0, 0], tris[m, 0, 1], tris[m, 0, 2]])
        b = ti.Vector([tris[m, 1, 0], tris[m, 1, 1], tris[m, 1, 2]])
        c = ti.Vector([tris[m, 2, 0], tris[m, 2, 1], tris[m, 2, 2]])

        e1 = b - a
        e2 = c - a

        lo = ti.floor(ti.min(a, b, c)) - 1.0
        hi = ti.ceil(ti.max(a, b, c)) + 1.0

        i0 = ti.max(ti.cast(lo[0], ti.i32), 0 )
        j0 = ti.max(ti.cast(lo[1], ti.i32), 0)
        k0 = ti.max(ti.cast(lo[2], ti.i32), 0)

        i1 = ti.min(ti.cast(hi[0], ti.i32), q.shape[1] - 1)
        j1 = ti.min(ti.cast(hi[1], ti.i32), q.shape[2] - 1)
        k1 = ti.min(ti.cast(hi[2], ti.i32), q.shape[3] - 1)

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    p = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)])
                    s = p - a

                    for d in range(E.shape[0]):
                        dv = ti.Vector([ti.cast(E[d, 0], ti.f32), ti.cast(E[d, 1], ti.f32), ti.cast(E[d, 2], ti.f32)])
                        h = dv.cross(e2)
                        det = e1.dot(h)

                        if ti.abs(det) > 1e-12:
                            inv = 1.0 / det
                            u = s.dot(h) * inv

                            if u >= 0.0 and u <= 1.0:
                                qv = s.cross(e1)
                                v = dv.dot(qv) * inv

                                if v >= 0.0 and u + v <= 1.0:
                                    t = e2.dot(qv) * inv

                                    if t >= 0.0 and t <= 1.0:
                                        ti.atomic_min(q[d, i, j, k], t)

def q_from_mesh(tris, phi, nx, ny, nz, backend="cpu"):
    # exact Bouzidi fractions for a triangle mesh; phi (from sdf_from_mesh) defines which links are boundary links
    runtime.init(backend)
    
    qh = ti.field(ti.f32, shape=(L3.Q, nx, ny, nz))
    qh.fill(2.0)

    _link_hits(np.ascontiguousarray(tris, np.float32), np.ascontiguousarray(L3.E, np.int32), qh)

    t = qh.to_numpy()
    q_phi = q_from_sdf(phi, nx, ny, nz)
    link = q_phi > 0.0
    hit = t <= 1.0
    q = np.where(link & hit, np.maximum(t, 1e-6), q_phi)

    return q.astype(np.float32)