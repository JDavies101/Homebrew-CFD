# signed distance fields: phi > 0 fluid, phi < 0 solid, phi = 0 on the surface.
# one phi gives the solid mask, per-link Bouzidi fractions q, wall normals, and wall distance.
# import numpy as np
from src.engine import lattice3d as L3
import numpy as np

def sdf_sphere(cx, cy, cz, R):
    # returns a function phi(x, y, z) that works on scalars or whole arrays
    def phi(x, y, z):
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - R
    return phi

def node_grid(nx, ny, nz):
    # float coordinates of every lattice node, each array shaped (nx, ny, nz)
    return np.meshgrid(np.arange(nx, dtype=np.float64),
                       np.arange(ny, dtype=np.float64),
                       np.arange(nz, dtype=np.float64),
                       indexing='ij')

def solid_from_sdf(phi, nx, ny, nz):
    X, Y, Z = node_grid(nx, ny, nz)
    return (phi(X, Y, Z) < 0.0).astype(np.int32)

def q_from_sdf(phi, nx, ny, nz, iters=40):
    X, Y, Z = node_grid(nx, ny, nz)
    solid = phi(X, Y, Z) < 0.0
    q = np.zeros((L3.Q, nx, ny, nz), np.float32)

    for d in range(L3.Q):
        ex, ey, ez = (int(v) for v in L3.E[d])
        if ex ==0 and ey == 0 and ez == 0:
            continue

        inb = ((X + ex >= 0) & (X + ex < nx) & (Y + ey >= 0) & (Y + ey < ny)
               & (Z + ez >= 0) & (Z + ez < nz))
        nbr = np.roll(solid, (-ex, -ey, -ez), axis=(0, 1, 2))
        link = (~solid) & nbr & inb
        idx = np.nonzero(link)
        x0 = X[idx]
        y0 = Y[idx]
        z0 = Z[idx]
        lo = np.zeros_like(x0)
        hi = np.ones_like(x0)

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            inside = phi(x0 + mid * ex, y0 + mid * ey, z0 + mid * ez) < 0.0
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)

        q[d][idx] = (0.5 * (lo + hi)).astype(np.float32)

    return q

def normals_from_sdf(phi, x, y, z, h=1e-4):
    # outward unit normal grad(phi)/|grad(phi)| by central differences; x, y, z are 1D arrays of points
    gx = (phi(x + h, y, z) - phi(x - h, y, z)) / (2.0 * h)
    gy = (phi(x, y + h, z) - phi(x, y - h, z)) / (2.0 * h)
    gz = (phi(x, y, z + h) - phi(x, y, z - h)) / (2.0 * h)
    g = np.sqrt(gx * gx + gy * gy + gz * gz)
    return np.stack([gx / g, gy / g, gz / g], axis=-1)