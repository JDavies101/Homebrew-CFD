# 3D D3Q19 LBM solver, runs on cpu or cuda
# geometry-agnostic: consumes solid/lid/q fields, does not build shapes
import taichi as ti
import numpy as np
from src.engine import lattice3d as L3

@ti.data_oriented
class Simulation3D:
    # allocate fields and load the lattice constants
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
        self.MIRROR_Z = ti.field(ti.i32, shape=L3.Q)
        self.MIRROR_Z.from_numpy(L3.MIRROR_Z.astype(np.int32))
        self.q  = ti.field(ti.f32, shape=(L3.Q, nx, ny, nz))   # wall fractions (0 = not a boundary link)
        self.fc = ti.field(ti.f32, shape=(L3.Q, nx, ny, nz))   # post-collision snapshot (Bouzidi needs it)
        # lattice constants as fields, built from the NumPy descriptor
        self.E   = ti.field(ti.i32, shape=(self.Q, self.D))
        self.E.from_numpy(L3.E.astype(np.int32))
        self.W   = ti.field(ti.f32, shape=self.Q)
        self.W.from_numpy(L3.W.astype(np.float32))
        self.OPP = ti.field(ti.i32, shape=self.Q)
        self.OPP.from_numpy(L3.OPP.astype(np.int32))

    # density and velocity from the populations
    @ti.kernel
    def macroscopic(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):    # parallel over cells
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):            # serial: sum the 19 populations
                r += self.f[q, i, j, k]
                mx += self.f[q, i, j, k] * self.E[q, 0]
                my += self.f[q, i, j, k] * self.E[q, 1]
                mz += self.f[q, i, j, k] * self.E[q, 2]
            self.rho[i, j, k] = r
            self.u[0, i, j, k] = mx / r
            self.u[1, i, j, k] = my / r
            self.u[2, i, j, k] = mz / r

    # wrappers: each collision variant is a special case of collide_full
    # plain BGK
    def collide(self, tau: ti.f32):
        self.collide_full(tau, 0.0, 0.0, 0)

    # BGK + Guo body force in x
    def collide_forced(self, tau: ti.f32, gx: ti.f32):
        self.collide_full(tau, 0.0, gx,  0)

    # two-relaxation-time
    def collide_trt(self, tau: ti.f32):
        self.collide_full(tau, 0.0, 0.0, 1)

    # BGK + Smagorinsky subgrid viscosity
    def collide_les(self, tau: ti.f32, cs: ti.f32):
        self.collide_full(tau, cs,  0.0, 0)

    # one collision kernel: moments -> local tau (LES) -> TRT/BGK relax + Guo source
    # cs=0 disables LES, gx=0 disables forcing, trt=0 gives BGK
    @ti.kernel
    def collide_full(self, tau0: ti.f32, cs: ti.f32, gx: ti.f32, trt: ti.i32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):
                r += self.f[q, i, j, k]
                mx += self.f[q, i, j, k] * self.E[q, 0]
                my += self.f[q, i, j, k] * self.E[q, 1]
                mz += self.f[q, i, j, k] * self.E[q, 2]
            ux = (mx + 0.5 * gx) / r
            uy = my / r
            uz = mz / r
            usqr = ux*ux + uy*uy + uz*uz

            tau = tau0
            if cs > 0.0:
                Qxx=0.0
                Qyy=0.0
                Qzz=0.0
                Qxy=0.0
                Qxz=0.0
                Qyz=0.0
                for q in range(self.Q):
                    eu = self.E[q,0] * ux + self.E[q,1] * uy + self.E[q,2] * uz
                    feq = self.W[q] * r * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usqr)
                    neq = self.f[q,i,j,k] - feq
                    Qxx += self.E[q,0] * self.E[q,0] * neq
                    Qyy += self.E[q,1] * self.E[q,1] * neq
                    Qzz += self.E[q,2] * self.E[q,2] * neq
                    Qxy += self.E[q,0] * self.E[q,1] * neq
                    Qxz += self.E[q,0] * self.E[q,2] * neq
                    Qyz += self.E[q,1] * self.E[q,2] * neq

                Qmag = ti.sqrt(Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz))
                tau = 0.5 * (tau0 + ti.sqrt(tau0 * tau0 + 18.0 * cs * cs * Qmag / r))

            s_plus = 1.0 / tau
            s_minus = s_plus # trt = 0 -> BGK
            if trt == 1:
                s_minus = 1.0 / (0.5 + (3.0 / 16.0) / (tau - 0.5))
            pre_plus = 1.0 - 0.5 * s_plus # Guo prefactor
            pre_minus = 1.0 - 0.5 * s_minus
            for q in range(self.Q):
                m = self.OPP[q]
                if q <= m:                                   # each pair once
                    eu = self.E[q,0] * ux + self.E[q,1] * uy + self.E[q,2] * uz
                    even = self.W[q]*r*(1 + 4.5*eu*eu - 1.5*usqr)
                    odd  = self.W[q]*r*(3.0*eu)
                    eF, uF = self.E[q,0]*gx, ux*gx
                    sym  = self.W[q]*(9.0*eu*eF - 3.0*uF)
                    asym = self.W[q]*(3.0*eF)
                    Sq = pre_plus*sym + pre_minus*asym
                    Sm = pre_plus*sym - pre_minus*asym
                    fq, fm = self.f[q,i,j,k], self.f[m,i,j,k]
                    fp, fmn = 0.5*(fq+fm), 0.5*(fq-fm)
                    self.f[q,i,j,k] = fq - s_plus*(fp-even) - s_minus*(fmn-odd) + Sq
                    if q != m:
                        self.f[m,i,j,k] = fm - s_plus*(fp-even) + s_minus*(fmn-odd) + Sm

    # pull each population from its upstream neighbour into f_new
    @ti.kernel
    def _stream(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):        # parallel over destination cells
            for q in range(self.Q):
                src_i = (i - self.E[q,0]) % self.nx     # where this population came from
                src_j = (j - self.E[q,1]) % self.ny     # % nx/ny/nz = periodic wrap
                src_k = (k - self.E[q,2]) % self.nz
                self.f_new[q, i, j, k] = self.f[q, src_i, src_j, src_k]

    # stream then swap the buffer back into f
    def stream(self):                        # plain Python: kernel + buffer swap
        self._stream()
        self.f.copy_from(self.f_new)

    # no-slip wall: reverse the populations at solid nodes
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

    # Bouzidi wall: interpolate the reflected population using the sub-cell
    # wall fraction q, so curved surfaces are not staircased
    # reads fc (post-collision), writes f (post-stream)
    @ti.kernel
    def bounce_back_interp(self):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            for d in range(self.Q):
                qf = self.q[d, i, j, k]
                if qf > 0.0:                          # dir d from (i,j,k) crosses the wall
                    ob = self.OPP[d]
                    fi = self.fc[d, i, j, k]          # post-collision f_i at x_f
                    if qf <= 0.5:
                        iff = i - self.E[d, 0]; jff = j - self.E[d, 1]; kff = k - self.E[d, 2]
                        if 0 <= iff < self.nx and 0 <= jff < self.ny and 0 <= kff < self.nz:
                            fiff = self.fc[d, iff, jff, kff]
                            self.f[ob, i, j, k] = 2.0 * qf * fi + (1.0 - 2.0*qf) * fiff
                        else:                          # no upstream node -> fall back to halfway
                            self.f[ob, i, j, k] = fi
                    else:
                        fib = self.fc[ob, i, j, k]     # post-collision f_ibar at x_f
                        self.f[ob, i, j, k] = fi / (2.0 * qf) + (2.0 * qf - 1.0) / (2.0 * qf) * fib
        
    # free-slip y walls: specular reflection, mirrors the y component
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
                    t2 = self.f[q, i, self.ny-1, k]
                    self.f[q, i, self.ny - 1, k] = self.f[m, i , self.ny - 1, k]
                    self.f[m, i, self.ny - 1, k] = t2
    
    # free-slip z walls: specular reflection, mirrors the z component
    @ti.kernel
    def free_slip_z(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            for q in range(self.Q):
                m = self.MIRROR_Z[q]
                if q < m:
                    #  wall k = 0
                    t = self.f[q, i, j, 0]
                    self.f[q, i, j, 0] = self.f[m, i, j, 0]
                    self.f[m, i, j, 0] = t
                    # wall k = nz - 1
                    t2 = self.f[q, i, j, self.nz - 1]
                    self.f[q, i, j, self.nz - 1] = self.f[m, i , j, self.nz - 1]
                    self.f[m, i, j, self.nz - 1] = t2

    # lid: bounce-back plus a momentum kick so the wall drags the fluid at U
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
    
    # equilibrium inlet at x=0, rho fixed at 1
    # stable but soft: the free-stream can sag below U against blockage
    @ti.kernel
    def inlet(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            for q in range(self.Q):
                eu = self.E[q,0] * U # u = (U,0,0) so e*u = E[q,0] * U
                usqr = U * U
                self.f[q,0,j,k] = self.W[q] * (1+ 3 * eu + 4.5 * eu * eu - 1.5 * usqr)
    
    # Guo non-equilibrium extrapolation inlet: equilibrium at the imposed U with
    # rho taken from x=1, plus the neighbour non-equilibrium part
    # holds the free-stream rigidly, but diverges where two free-slip walls meet
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
    def inlet_neem_open(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            if self.solid[0, j, k] == 0 and self.solid[1, j, k] == 0:   # open column only
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

    # zero-gradient outlet: copy the second-to-last plane onto the last
    @ti.kernel
    def outlet(self):
        for j, k in ti.ndrange(self.ny, self.nz):
            for q in range(self.Q):
                self.f[q, self.nx - 1, j, k] = self.f[q, self.nx - 2, j , k]

    # force on the solid by momentum exchange, sum 2 c_i f_i over fluid->solid links
    # call after collide, before stream
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
                            self.force[0] += 2.0 * self.f[q,i,j,k] * self.E[q,0]
                            self.force[1] += 2.0 * self.f[q,i,j,k] * self.E[q,1]
                            self.force[2] += 2.0 * self.f[q,i,j,k] * self.E[q,2]

    # momentum exchange for interpolated walls: sum c_i (f_in + f_out)
    # call AFTER bounce_back_interp, so f holds the reconstructed reflection
    @ti.kernel
    def drag_interp(self):
        for c in range(L3.D):
            self.force[c] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            for d in range(self.Q):
                if self.q[d, i, j, k] > 0.0:          # same link set Bouzidi uses
                    ob = self.OPP[d]
                    fin  = self.fc[d, i, j, k]        # post-collision, into the wall
                    fout = self.f[ob, i, j, k]        # reconstructed reflection
                    self.force[0] += (fin + fout) * self.E[d, 0]
                    self.force[1] += (fin + fout) * self.E[d, 1]
                    self.force[2] += (fin + fout) * self.E[d, 2]    