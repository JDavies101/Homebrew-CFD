import taichi as ti
import numpy as np
from src.engine import lattice3d as L3

@ti.data_oriented
class Simulation3D:
    def __init__(self, nx, ny, nz, backend="cpu"):
        ti.init(arch=ti.cuda if backend == "cuda" else ti.cpu)
        self.Q = L3.Q
        self.D = L3.D
        self.nx, self.ny, self.nz = nx, ny, nz
        # populations + macroscopic
        self.f     = ti.field(ti.f32, shape=(self.Q, nx, ny, nz))
        self.f_new = ti.field(ti.f32, shape=(self.Q, nx, ny, nz))
        self.rho   = ti.field(ti.f32, shape=(nx, ny, nz))
        self.u     = ti.field(ti.f32, shape=(self.D, nx, ny, nz))
        self.solid = ti.field(ti.i32, shape=(nx, ny, nz))
        self.lid   = ti.field(ti.i32, shape=(nx, ny, nz))
        # lattice constants as fields, built from the NumPy descriptor
        self.E   = ti.field(ti.i32, shape=(self.Q, self.D))
        self.E.from_numpy(L3.E.astype(np.int32))
        self.W   = ti.field(ti.f32, shape=self.Q)
        self.W.from_numpy(L3.W.astype(np.float32))
        self.OPP = ti.field(ti.i32, shape=self.Q)
        self.OPP.from_numpy(L3.OPP.astype(np.int32))

    @ti.kernel
    def macroscopic(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):    # PARALLEL over cells
            # same body as before, but self.f, self.rho, self.u, self.E
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):            # serial: sum the 9 populations
                r += self.f[q, i, j, k]
                mx += self.f[q, i, j, k] * self.E[q, 0]
                my += self.f[q, i, j, k] * self.E[q, 1]
                mz += self.f[q, i, j, k] * self.E[q, 2]
            self.rho[i, j, k] = r
            self.u[0, i, j, k] = mx / r
            self.u[1, i, j, k] = my / r
            self.u[2, i, j, k] = mz / r

    @ti.kernel
    def collide(self, tau: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):        # parallel over cells

            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):
                r += self.f[q, i, j, k]
                mx += self.f[q, i, j, k] * self.E[q, 0]
                my += self.f[q, i, j, k] * self.E[q, 1]
                mz += self.f[q, i, j, k] * self.E[q, 2]
            ux = mx / r
            uy = my / r
            uz = mz / r
            usqr = ux*ux + uy*uy + uz*uz

            # 2. equilibrium + BGK relax, per direction
            for q in range(self.Q):
                eu = self.E[q,0]*ux + self.E[q,1]*uy + self.E[q,2]*uz
                feq = self.W[q] * r * (1 + (3 * eu) + (4.5 * eu*eu) - (1.5 * usqr))
                self.f[q,i,j,k] += -(1/tau)*(self.f[q,i,j,k]-feq)

    @ti.kernel
    def _stream(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):        # parallel over destination cells
            for q in range(self.Q):
                src_i = (i - self.E[q,0]) % self.nx     # where this population came from
                src_j = (j - self.E[q,1]) % self.ny     # % nx/ny = periodic wrap
                src_k = (k - self.E[q,2]) % self.nz
                self.f_new[q, i, j, k] = self.f[q, src_i, src_j, src_k]

    def stream(self):                        # plain Python: kernel + buffer swap
        self._stream()
        self.f.copy_from(self.f_new)

    @ti.kernel
    def bounce_back(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):        # parallel over cells
            if self.solid[i, j, k] == 1:               # only at walls
                for q in range(self.Q):
                    o = self.OPP[q]
                    if q < o:                  # visit each opposite-pair once
                        tmp = self.f[q, i, j, k]
                        self.f[q, i, j, k] = self.f[o, i, j, k]
                        self.f[o, i, j, k] = tmp

    @ti.kernel
    def moving_wall(self, U: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.lid[i, j, k] == 1:
                for q in range(self.Q):
                    o = self.OPP[q]
                    if q < o:
                        tmp = self.f[q, i, j, k]
                        self.f[q, i, j, k] = self.f[o, i, j, k]
                        self.f[o, i, j, k] = tmp  # bounce-back
                        corr = 6.0 * self.W[q] * self.E[q, 0] * U
                        self.f[q, i, j, k] += corr
                        self.f[o, i, j, k] -= corr        # opposite dir gets the negative
    
    @ti.kernel
    def collide_forced(self, tau: ti.f32, gx: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):
                r += self.f[q,i,j,k]
                mx += self.f[q,i,j,k] * self.E[q,0]
                my += self.f[q,i,j,k] * self.E[q,1]
                mz += self.f[q,i,j,k] * self.E[q,2]
            ux = (mx + gx*0.5) / r # half-force velocity correction
            uy = my / r
            uz = mz / r
            usqr = ux*ux + uy*uy + uz*uz
            prefac = 1.0 - 1.0 / (2.0 * tau)
            for q in range(self.Q):
                eu = self.E[q,0] * ux + self.E[q,1] * uy + self.E[q,2] * uz
                eF = self.E[q,0] * gx # F only in x
                uF = ux * gx
                feq = self.W[q] * r * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usqr)
                S = prefac * self.W[q] * (3 * (eF - uF) + 9 * eu * eF)
                self.f[q,i,j,k] += -(1.0 / tau) * (self.f[q,i,j,k] - feq) + S
    
    def step(self, tau, U=0.0):        # NO @ti.kernel — plain Python
        
        self.collide(tau)
        self.stream()
        self.bounce_back()
        self.moving_wall(U)

    def run(self, steps, tau, U=0.0):  # NO @ti.kernel
        
        for _ in range(steps):
            self.step(tau, U)