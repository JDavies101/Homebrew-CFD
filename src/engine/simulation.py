import taichi as ti
import numpy as np
from src.engine import lattice as L

@ti.data_oriented
class Simulation:
    def __init__(self, nx, ny, backend="cpu"):
        ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu)
        self.Q = L.Q
        self.D = L.D
        self.nx, self.ny = nx, ny
        # populations + macroscopic
        self.f     = ti.field(ti.f32, shape=(self.Q, nx, ny))
        self.f_new = ti.field(ti.f32, shape=(self.Q, nx, ny))
        self.rho   = ti.field(ti.f32, shape=(nx, ny))
        self.u     = ti.field(ti.f32, shape=(self.D, nx, ny))
        self.solid = ti.field(ti.i32, shape=(nx, ny))
        self.lid   = ti.field(ti.i32, shape=(nx, ny))
        # lattice constants as fields, built from the NumPy descriptor
        self.E   = ti.field(ti.i32, shape=(self.Q, self.D))
        self.E.from_numpy(L.E.astype(np.int32))
        self.W   = ti.field(ti.f32, shape=self.Q)
        self.W.from_numpy(L.W.astype(np.float32))
        self.OPP = ti.field(ti.i32, shape=self.Q)
        self.OPP.from_numpy(L.OPP.astype(np.int32))

    @ti.kernel
    def macroscopic(self):
        for i, j in ti.ndrange(self.nx, self.ny):    # PARALLEL over cells
            # same body as before, but self.f, self.rho, self.u, self.E
            r = 0.0
            mx = 0.0
            my = 0.0
            for q in range(self.Q):            # serial: sum the 9 populations
                r += self.f[q, i, j]
                mx += self.f[q, i, j] * self.E[q, 0]
                my += self.f[q, i, j] * self.E[q, 1]
            self.rho[i, j] = r
            self.u[0, i, j] = mx / r
            self.u[1, i, j] = my / r

    @ti.kernel
    def collide(self, tau: ti.f32):
        for i, j in ti.ndrange(self.nx, self.ny):        # parallel over cells

            r = 0.0; mx = 0.0; my = 0.0
            for q in range(self.Q):
                r += self.f[q, i, j]
                mx += self.f[q, i, j] * self.E[q, 0]
                my += self.f[q, i, j] * self.E[q, 1]
            ux = mx / r
            uy = my / r
            usqr = ux*ux + uy*uy

            # 2. equilibrium + BGK relax, per direction
            for q in range(self.Q):
                eu = self.E[q,0]*ux + self.E[q,1]*uy
                feq = self.W[q] * r * (1 + (3 * eu) + (4.5 * eu*eu) - (1.5 * usqr))
                self.f[q,i,j] += -(1/tau)*(self.f[q,i,j]-feq)

    @ti.kernel
    def _stream(self):
        for i, j in ti.ndrange(self.nx, self.ny):        # parallel over destination cells
            for q in range(self.Q):
                src_i = (i - self.E[q,0]) % self.nx     # where this population came from
                src_j = (j - self.E[q,1]) % self.ny     # % nx/ny = periodic wrap
                self.f_new[q, i, j] = self.f[q, src_i, src_j]

    def stream(self):                        # plain Python: kernel + buffer swap
        self._stream()
        self.f.copy_from(self.f_new)

    @ti.kernel
    def bounce_back(self):
        for i, j in ti.ndrange(self.nx, self.ny):        # parallel over cells
            if self.solid[i, j] == 1:               # only at walls
                for q in range(self.Q):
                    o = self.OPP[q]
                    if q < o:                  # visit each opposite-pair once
                        tmp = self.f[q, i, j]
                        self.f[q, i, j] = self.f[o, i, j]
                        self.f[o, i, j] = tmp

    @ti.kernel
    def moving_wall(self, U: ti.f32):
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.lid[i, j] == 1:
                for q in range(self.Q):
                    o = self.OPP[q]
                    if q < o:
                        tmp = self.f[q, i, j]
                        self.f[q, i, j] = self.f[o, i, j]
                        self.f[o, i, j] = tmp  # bounce-back
                        corr = 6.0 * self.W[q] * self.E[q, 0] * U
                        self.f[q, i, j] += corr
                        self.f[o, i, j] -= corr        # opposite dir gets the negative
    
    def step(self, tau, U=0.0):        # NO @ti.kernel — plain Python
        
        self.collide(tau)
        self.stream()
        self.bounce_back()
        self.moving_wall(U)

    def run(self, steps, tau, U=0.0):  # NO @ti.kernel
        
        for _ in range(steps):
            self.step(tau, U)