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
        self.force = ti.field(ti.f32, shape=L3.D)
        self.MIRROR_Y = ti.field(ti.i32, shape=L3.Q)
        self.MIRROR_Y.from_numpy(L3.MIRROR_Y.astype(np.int32))
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
    def free_slip_y(self):
        for i, k in ti.ndrange(self.nx, self.nz):
            for q in range(self.Q):
                m = self.MIRROR_Y[q]
                if q < m:
                    # bottom wall j = 0
                    t = self.f[q, i, 0, k]
                    self.f[q, i, 0, k] = self.f[m, i, 0, k]
                    self.f[m, i, 0, k] = t
                    # top wall j = ny - 1
                    self.f[q, i, self.ny - 1, k] = self.f[m, i , self.ny - 1, k]
                    self.f[m, i, self.ny - 1, k] = t

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
    
    @ti.kernel
    def inlet(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            for q in range(self.Q):
                eu = self.E[q,0] * U # u = (U,0,0) so e*u = E[q,0] * U
                usqr = U * U
                self.f[q,0,j,k] = self.W[q] * (1+ 3 * eu + 4.5 * eu * eu - 1.5 * usqr)
    
    @ti.kernel
    def inlet_neem(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            # neighbor (x = 1) moments
            rn = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):
                rn += self.f[q, 1, j, k]
                mx += self.f[q, 1, j, k] * self.E[q,0]
                my += self.f[q, 1, j, k] * self.E[q,1]
                mz += self.f[q, 1, j, k] * self.E[q,2]
            ux = mx / rn
            uy = my / rn
            uz = mz / rn
            usqr_n = ux * ux + uy * uy + uz * uz
            usqr_b = U * U
            for q in range(self.Q):
                eu_b = self.E[q,0] * U # imposed u = (U, 0, 0)
                feq_b = self.W[q] * rn * (1 + 3 * eu_b + 4.5 * eu_b * eu_b - 1.5 * usqr_b)
                eu_n = self.E[q,0] * ux + self.E[q,1] * uy + self.E[q,2] * uz
                feq_n = self.W[q] * rn * (1+ 3 * eu_n + 4.5 * eu_n * eu_n - 1.5 * usqr_n)
                self.f[q, 0, j, k] = feq_b + (self.f[q, 1, j, k] - feq_n)

    @ti.kernel
    def outlet(self):
        
        for j, k in ti.ndrange(self.ny, self.nz):
            for q in range(self.Q):
                self.f[q, self.nx - 1, j, k] = self.f[q, self.nx - 2, j , k]

    def cylinder(nx, ny, nz, cx, cy, r):
        solid = np.zeros((nx, ny, nz), np.int32)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing = "ij")
        disc = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2 # (nx, ny) boolean
        solid[disc] = 1
        return solid
    
    @ti.kernel
    def drag(self):
        for c in range(L3.D): # reset accumulator
            self.force[c] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid[i,j,k] == 0:
                for q in range(self.Q):
                    ni = i + self.E[q,0]
                    nj = j + self.E[q,1]
                    nk = k + self.E[q,2]
                    # if the neighbor is inside the domain and solid, then boundary link
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                        if self.solid[ni, nj, nk] == 1:
                            contrib = self.f[q,i,j,k] + self.f[self.OPP[q],i,j,k]
                            self.force[0] += contrib * self.E[q,0]
                            self.force[1] += contrib * self.E[q,1]
                            self.force[2] += contrib * self.E[q,2]
    
    def step(self, tau, U=0.0):        # NO @ti.kernel — plain Python
        
        self.collide(tau)
        self.stream()
        self.bounce_back()
        self.moving_wall(U)

    def run(self, steps, tau, U=0.0):  # NO @ti.kernel
        
        for _ in range(steps):
            self.step(tau, U)