# sub-cell wall fractions q for interpolated bounce-back
from src.engine import lattice3d as L3
import numpy as np

def wall_fraction_cylinder(nx, ny, nz, cx, cy, R):
    q = np.zeros((L3.Q, nx, ny, nz), np.float32)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid = ((X - cx) ** 2 + (Y - cy) ** 2 < R ** 2)          # (nx,ny) bool

    for i in range(nx):
        for j in range(ny):
            if solid[i, j]:
                continue                                # only fluid nodes have boundary links
            for d in range(L3.Q):
                ex, ey = int(L3.E[d, 0]), int(L3.E[d, 1])
                ni, nj = i + ex, j + ey
                if not (0 <= ni < nx and 0 <= nj < ny) or not solid[ni, nj]:
                    continue                            # neighbor must be solid
                a = ex * ex + ey * ey
                if a == 0:                              # pure-z link can't hit the cylinder
                    continue
                dx, dy = i - cx, j - cy
                b = 2*(dx * ex + dy * ey)
                cc = dx * dx + dy * dy - R*R
                disc = b * b - 4 * a * cc
                if disc < 0:
                    continue
                t = (-b - np.sqrt(disc)) / (2 * a)        # first crossing from fluid side
                q[d, i, j, :] = t                       # same for every z (cylinder spans z)
    return q