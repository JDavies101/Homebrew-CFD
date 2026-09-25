# 3D D3Q19 LBM solver, runs on cpu or cuda
# geometry-agnostic: consumes solid/body/lid/q fields, does not build shapes
import taichi as ti
import numpy as np
from src.engine import lattice3d as L3
from src.engine import runtime

@ti.data_oriented
class Simulation3D:
    # allocate fields and load the lattice constants
    def __init__(self, nx, ny, nz, backend="cpu", interp=False):
        runtime.init(backend)
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
        if interp: # Bouzidi only: wall fractions + post-collision snapshot
            self.q  = ti.field(ti.f32, shape=(L3.Q, nx, ny, nz))   # wall fractions (0 = not a boundary link)
            self.fc = ti.field(ti.f32, shape=(L3.Q, nx, ny, nz))   # post-collision snapshot (Bouzidi needs it)
        self.nut_wall = ti.field(ti.f32, shape =(nx, ny, nz))  # wall-model eddy viscosity (0 away from walls)
        self.nut_les = ti.field(ti.f32, shape=(nx, ny, nz)) # WALE subgrid eddy viscosity (0 until les_wale runs)
        self.nut_sponge = ti.field(ti.f32, shape=(nx, ny, nz))   # absorbing-layer viscosity, set once, 0 by default
        self.body = ti.field(ti.i32, shape=(nx, ny, nz))       # body-only mask for drag (solid minus tunnel walls)
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

    # wrappers: each of these is a special case of collide_full (collide_reg is separate)
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

    # regularized collision: rebuild f_neq from the stress Pi only (drops the ghost moments
    # that blow up as tau -> 0.5), then relax. single rate, no trt split
    # cs=0 disables LES, gx=0 disables forcing. forced + regularized still needs the Guo
    # correction to Pi (regularizing zeroes f_neq's -F/2 first moment), so keep gx=0 here
    @ti.kernel
    def collide_reg(self, tau0: ti.f32, cs: ti.f32, gx: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue        # wall nodes only hold bounced populations for one step, don't collide them
            
            # moments
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for q in range(self.Q):
                f = self.f[q, i, j, k]
                r += f
                mx += f * self.E[q, 0]
                my += f * self.E[q, 1]
                mz += f * self.E[q, 2]

            ux = (mx + 0.5 * gx) / r # Guo half-force correction, same as collide_full
            uy = my / r
            uz = mz / r
            usqr = ux * ux + uy * uy + uz * uz

            # pass 1: non equilibrium stress tensor Pi = sum c_a c_b (f - feq)
            P = self._stress(i, j, k, r, ux, uy, uz, usqr)
            Pxx = P[0]
            Pyy = P[1]
            Pzz = P[2]
            Pxy = P[3]
            Pxz = P[4]
            Pyz = P[5]
            trace = Pxx + Pyy + Pzz

            # pass 2: reconstruct f_neq from Pi only then relax
            tau = tau0
            if cs > 0.0:                                     # LES: eddy-adjusted tau from stress magnitude
                Qmag = ti.sqrt(Pxx*Pxx + Pyy*Pyy + Pzz*Pzz + 2.0*(Pxy*Pxy + Pxz*Pxz + Pyz*Pyz))
                tau = 0.5 * (tau0 + ti.sqrt(tau0*tau0 + 18.0 * ti.sqrt(2.0) *cs*cs*Qmag / r))
            s = 1.0 / (tau + 3.0 * (self.nut_wall[i, j, k] + self.nut_les[i, j, k] + self.nut_sponge[i, j, k]))
            pre = 1.0 - 0.5 * s # Guo prefactor; BGK single rate
            for q in range(self.Q):
                Hq = (self.E[q, 0] * self.E[q, 0] * Pxx + self.E[q, 1] * self.E[q, 1] * Pyy 
                      + self.E[q, 2] * self.E[q, 2] * Pzz
                      + 2.0 * (self.E[q, 0] * self.E[q, 1] * Pxy
                               + self.E[q, 0] * self.E[q, 2] * Pxz
                               + self.E[q, 1] * self.E[q, 2] * Pyz)
                        - (1.0 / 3.0) * trace)
                # first-order Hermite term (a1 = -F/2) that regularization drops; restores f_neq's
                # -F/2 first moment so the forced momentum balance matches the full-f scheme. F = (gx,0,0).
                fneq_reg = 4.5 * self.W[q] * Hq - 1.5 * self.W[q] * self.E[q, 0] * gx

                eu = self.E[q, 0] * ux + self.E[q, 1] * uy + self.E[q, 2] * uz
                eF = self.E[q, 0] * gx
                uF = ux * gx
                Sq = pre * self.W[q] * (3.0 * (eF - uF) + 9.0 * eu * eF) # Guo source (BGK form)

                self.f[q, i, j, k] = self.feq(q, r, ux, uy, uz, usqr) + (1.0 - s) * fneq_reg + Sq
    
    # one collision kernel: moments -> local tau (LES + wall model) -> TRT/BGK relax + Guo source
    # cs=0 disables LES, gx=0 disables forcing, trt=0 gives BGK
    @ti.kernel
    def collide_full(self, tau0: ti.f32, cs: ti.f32, gx: ti.f32, trt: ti.i32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue        # wall nodes only hold bounced populations for one step, don't collide them
            
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
                Q = self._stress(i, j, k, r, ux, uy, uz, usqr)
                Qxx = Q[0]; Qyy = Q[1]; Qzz = Q[2]; Qxy = Q[3]; Qxz = Q[4]; Qyz = Q[5]

                Qmag = ti.sqrt(Qxx * Qxx + Qyy * Qyy + Qzz * Qzz + 2 * (Qxy * Qxy + Qxz * Qxz + Qyz * Qyz))
                tau = 0.5 * (tau0 + ti.sqrt(tau0 * tau0 + 18.0 * ti.sqrt(2.0) * cs * cs * Qmag / r))
            
            tau += 3.0 * (self.nut_wall[i, j, k] + self.nut_les[i, j, k] + self.nut_sponge[i, j, k])

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

    # log-law wall model: at fluid nodes touching solid, set nut_wall so the first node
    # carries tau_w = u_tau^2. only engages at y+ > 30 (below that the node is resolved)
    # call after macroscopic(), before collide. y1 = first-node wall distance = 0.5
    # (halfway bounce-back, now that wall nodes aren't collided)
    @ti.kernel
    def wall_model(self, nu: ti.f32, y1: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.nut_wall[i, j, k] = 0.0
            if self.solid[i, j, k] == 0:
                # discrete inward normal: sum lattice vectors pointing at solid neighbors
                gx = 0.0; gy = 0.0; gz = 0.0; nsolid = 0
                for q in range(self.Q):
                    ni = i + self.E[q,0]; nj = j + self.E[q,1]; nk = k + self.E[q,2]
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                        if self.solid[ni, nj, nk] == 1:
                            gx += self.E[q,0]; gy += self.E[q,1]; gz += self.E[q,2]
                            nsolid += 1
                gmag = ti.sqrt(gx*gx + gy*gy + gz*gz)
                if nsolid > 0 and gmag > 0.0:
                    nxn = -gx/gmag; nyn = -gy/gmag; nzn = -gz/gmag      # normal into the fluid
                    ux = self.u[0,i,j,k]; uy = self.u[1,i,j,k]; uz = self.u[2,i,j,k]
                    udn = ux*nxn + uy*nyn + uz*nzn
                    px = ux - udn*nxn; py = uy - udn*nyn; pz = uz - udn*nzn   # wall-parallel u
                    u1 = ti.sqrt(px*px + py*py + pz*pz)
                    u_tau = self.wall_utau(u1, y1, nu)
                    yplus = y1 * u_tau / nu
                    if yplus > 30.0 and u1 > 0.0:
                        self.nut_wall[i,j,k] = ti.max(0.0, u_tau*u_tau*y1/u1 - nu)   
        

    @ti.kernel
    def wall_model_fast(self, nu: ti.f32, y1: ti.f32):
        for m in range(self.n_wall):
            i = self.wall_ijk[m, 0]
            j = self.wall_ijk[m, 1]
            k = self.wall_ijk[m, 2]
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0

            for q in range(self.Q):
                f = self.f[q, i, j, k]
                r += f
                mx += f * self.E[q, 0]
                my += f * self.E[q, 1]
                mz += f * self.E[q, 2]

            ux = mx / r
            uy = my / r
            uz = mz / r
            nxn = self.wall_n[m, 0]
            nyn = self.wall_n[m, 1]
            nzn = self.wall_n[m, 2]
            udn = ux * nxn + uy * nyn + uz * nzn
            px = ux - udn * nxn
            py = uy - udn * nyn
            pz = uz - udn * nzn
            u1 = ti.sqrt(px * px + py * py + pz * pz)
            u_tau = self.wall_utau(u1, y1, nu)
            yplus = y1 * u_tau / nu
            val = 0.0
            if yplus > 30.0 and u1 > 0.0:
                val = ti.max(0.0, u_tau * u_tau * y1 / u1 - nu)
            
            self.nut_wall[i, j, k] = val

    @ti.kernel
    def les_wale(self, cw: ti.f32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.nut_les[i, j, k] = 0.0
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue

            up_x = self._uvec(i, j, k, 1, 0, 0)
            dn_x = self._uvec(i, j, k, -1, 0, 0)
            up_y = self._uvec(i, j, k, 0, 1, 0)
            dn_y = self._uvec(i, j, k, 0, -1, 0)
            up_z = self._uvec(i, j, k, 0, 0, 1)
            dn_z = self._uvec(i, j, k, 0, 0, -1)

            g = ti.Matrix.zero(ti.f32, 3, 3)          # g[a, b] = d u_a / d x_b
            for a in ti.static(range(3)):
                g[a, 0] = 0.5 * (up_x[a] - dn_x[a])
                g[a, 1] = 0.5 * (up_y[a] - dn_y[a])
                g[a, 2] = 0.5 * (up_z[a] - dn_z[a])

            S = 0.5 * (g + g.transpose())             # symmetric strain
            gsq = g @ g                               # g . g
            tr = gsq.trace()
            Sd = 0.5 * (gsq + gsq.transpose()) - (tr / 3.0) * ti.Matrix.identity(ti.f32, 3)

            SS = (S * S).sum()                        # S : S
            SdSd = (Sd * Sd).sum()                    # Sd : Sd

            delta = 1.0
            eps = 1e-12
            self.nut_les[i, j, k] = (cw * delta) ** 2 * SdSd ** 1.5 / (SS ** 2.5 + SdSd ** 1.25 + eps)

    @ti.func
    def _uvec(self, i, j, k, di, dj, dk):
        # velocity at neighbour (i+di, j+dj, k+dk): out of domain -> clamp to (i,j,k);
        # solid -> zero (no-slip); else the stored velocity
        ni = i + di
        nj = j + dj
        nk = k + dk
        v = ti.Vector([0.0, 0.0, 0.0])
        if ni < 0 or ni >= self.nx or nj < 0 or nj >= self.ny or nk < 0 or nk >= self.nz:
            v = ti.Vector([self.u[0, i, j, k], self.u[1, i, j, k], self.u[2, i, j, k]])
        elif self.solid[ni, nj, nk] == 1:
            v = ti.Vector([0.0, 0.0, 0.0])
        else:
            v = ti.Vector([self.u[0, ni, nj, nk], self.u[1, ni, nj, nk], self.u[2, ni, nj, nk]])
        return v

    def build_wall_list(self):
        solid = self.solid.to_numpy()
        E = L3.E
        nx, ny, nz = self.nx, self.ny, self.nz

        def shift(a, s):                       # a[c] -> a[c+s], zero-filled out of bounds
            out = np.zeros_like(a)
            src = tuple(slice(max(v, 0), a.shape[d] + min(v, 0)) for d, v in enumerate(s))
            dst = tuple(slice(max(-v, 0), a.shape[d] - max(v, 0)) for d, v in enumerate(s))
            out[dst] = a[src]
            return out

        g = np.zeros((3, nx, ny, nz))
        nsolid = np.zeros((nx, ny, nz), np.int32)
        for q in range(self.Q):
            nbr = shift(solid, E[q])           # 1 where the E[q] neighbour is solid, in bounds
            nsolid += nbr
            for d in range(3):
                g[d] += E[q, d] * nbr

        gmag = np.sqrt((g * g).sum(axis=0))
        cand = (solid == 0) & (nsolid > 0) & (gmag > 0.0)
        ijk = np.argwhere(cand).astype(np.int32)                    # (M,3)
        normals = (-g[:, cand] / gmag[cand]).T.astype(np.float32)   # (M,3)

        M = ijk.shape[0]
        self.n_wall = M
        self.wall_ijk = ti.field(ti.i32, shape=(M, 3)); self.wall_ijk.from_numpy(ijk)
        self.wall_n   = ti.field(ti.f32, shape=(M, 3)); self.wall_n.from_numpy(normals)
    
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

    # free-slip ceiling only (j = ny-1): far-field roof over a no-slip floor
    # not wired into ahmed yet: meeting the free-slip sides it hits the corner artifact
    @ti.kernel
    def free_slip_y_top(self):
        for i, k in ti.ndrange(self.nx, self.nz):
            for q in range(self.Q):
                m = self.MIRROR_Y[q]
                if q < m:
                    # top wall j = ny - 1
                    t = self.f[q, i, self.ny-1, k]
                    self.f[q, i, self.ny - 1, k] = self.f[m, i , self.ny - 1, k]
                    self.f[m, i, self.ny - 1, k] = t

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
                # inlet body:
                self.f[q, 0, j, k] = self.feq(q, 1.0, U, 0.0, 0.0, U * U)
    
    # Guo non-equilibrium extrapolation inlet: equilibrium at the imposed U with
    # rho taken from x=1, plus the neighbour non-equilibrium part
    # holds the free-stream rigidly, but diverges where two free-slip walls meet
    @ti.kernel
    def inlet_neem(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            self._neem_column(j, k, U)
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
                feq_b = self.feq(q, rn, U, 0.0, 0.0, usqr_b)
                feq_n = self.feq(q, rn, ux, uy, uz, usqr_n)
                self.f[q, 0, j, k] = feq_b + (self.f[q, 1, j, k] - feq_n)

    # same as inlet_neem but only drives open columns, leaves solid at the inlet plane alone
    @ti.kernel
    def inlet_neem_open(self, U: ti.f32):
        for j, k in ti.ndrange(self.ny, self.nz):
            if self.solid[0, j, k] == 0 and self.solid[1, j, k] == 0:   # open column only
                self._neem_column(j, k, U)
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

    @ti.func
    def _neem_column(self, j, k, U):
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
        for q in range(self.Q):
            feq_b = self.feq(q, rn, U, 0.0, 0.0, U * U)
            feq_n = self.feq(q, rn, ux, uy, uz, usqr_n)
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
        self._drag_mask(0)

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

    # momentum-exchange force; use_body=1 sums only body links (Cd), 0 sums all solid links
    @ti.func
    def _drag_mask(self, use_body: ti.i32):
        for c in range(L3.D):
            self.force[c] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid[i, j, k] == 0:
                for q in range(self.Q):
                    ni = i + self.E[q, 0]; nj = j + self.E[q, 1]; nk = k + self.E[q, 2]
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                        hit = (self.body[ni, nj, nk] == 1) if use_body else (self.solid[ni, nj, nk] == 1)
                        if hit:
                            self.force[0] += 2.0 * self.f[q, i, j, k] * self.E[q, 0]
                            self.force[1] += 2.0 * self.f[q, i, j, k] * self.E[q, 1]
                            self.force[2] += 2.0 * self.f[q, i, j, k] * self.E[q, 2]

    # same as drag but only links into the body, so tunnel walls don't pollute Cd
    # call after collide, before stream
    @ti.kernel
    def drag_body(self):
        self._drag_mask(1)

    # blow-up probe: max |f| over the field, forced huge on any NaN. a cheap GPU
    # reduction so the health check never pulls the whole f array on the healthy path
    @ti.kernel
    def f_absmax(self) -> ti.f32:
        hi = 0.0
        for q, i, j, k in self.f:
            v = self.f[q, i, j, k]
            if v != v:
                ti.atomic_max(hi, 1e30)
            else:
                ti.atomic_max(hi, ti.abs(v))
        return hi

    # equilibrium, the one copy collide_full / collide_reg / _init_eq all use
    @ti.func
    def feq(self, q, r, ux, uy, uz, usqr):
        eu = self.E[q,0] * ux + self.E[q,1] * uy + self.E[q,2] * uz
        return self.W[q] * r * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * usqr)
    
    @ti.func
    def _stress(self, i, j, k, r, ux, uy, uz, usqr):
        Pxx = 0.0
        Pyy = 0.0
        Pzz = 0.0
        Pxy = 0.0
        Pxz = 0.0
        Pyz = 0.0
        for q in range(self.Q):
            neq = self.f[q, i, j, k] - self.feq(q, r, ux, uy, uz, usqr)
            Pxx += neq * self.E[q, 0] * self.E[q, 0]
            Pyy += neq * self.E[q, 1] * self.E[q, 1]
            Pzz += neq * self.E[q, 2] * self.E[q, 2]
            Pxy += neq * self.E[q, 0] * self.E[q, 1]
            Pxz += neq * self.E[q, 0] * self.E[q, 2]
            Pyz += neq * self.E[q, 1] * self.E[q, 2]
        return ti.Vector([Pxx, Pyy, Pzz, Pxy, Pxz, Pyz])

    @ti.func
    def wall_utau(self, u1, y1, nu):          # device twin of turbulence/wall_function.friction_velocity
        u_tau = 0.0
        if u1 > 0.0:
            k = 0.41
            B = 5.2
            u_tau = ti.sqrt(nu * u1 / y1)                     # viscous initial guess
            for _ in range(50):                                # bound must be a compile-time constant
                f = u_tau * ((1.0/k) * ti.log(y1 * u_tau / nu) + B) - u1
                if ti.abs(f) < 1e-8:
                    break
                fp = (1.0/k) * ti.log(y1 * u_tau / nu) + B + 1.0/k
                nxt = u_tau - f / fp
                u_tau = nxt if nxt > 0.0 else u_tau * 0.5
        return u_tau

    # fill f with feq from whatever is in rho and u
    @ti.kernel
    def _init_eq(self):

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            r = self.rho[i, j, k]
            ux = self.u[0, i, j, k]
            uy = self.u[1, i, j, k]
            uz = self.u[2, i, j, k]
            usqr = ux * ux + uy * uy + uz * uz

            for q in range(self.Q):
                self.f[q, i, j, k] = self.feq(q, r, ux, uy, uz, usqr)
    
    # start from equilibrium at a prescribed velocity field (numpy in, fields loaded, kernel fills f)
    def init_equilibrium(self, ux, uy, uz, rho=1.0):
        self.u.from_numpy(np.stack([ux, uy, uz]).astype(np.float32))
        self.rho.from_numpy(np.full((self.nx, self.ny, self.nz), rho, np.float32))
        self._init_eq()