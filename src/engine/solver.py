from src.engine import lattice as L
import taichi as ti

nx, ny = 64, 64
f   = ti.field(ti.f32, shape=(L.Q, nx, ny))   # populations
f_new = ti.field(ti.f32, shape=(L.Q, nx, ny))
rho = ti.field(ti.f32, shape=(nx, ny))
u   = ti.field(ti.f32, shape=(L.D, nx, ny))   # u[0]=x, u[1]=y
solid = ti.field(ti.i32, shape=(nx, ny))   # 1 = wall, 0 = fluid
lid = ti.field(ti.i32, shape=(nx, ny))     # 1 = lid

@ti.kernel
def macroscopic():
    for i, j in ti.ndrange(nx, ny):     # PARALLEL over cells
        r = 0.0
        mx = 0.0
        my = 0.0
        for q in range(L.Q):            # serial: sum the 9 populations
            r += f[q, i, j]
            mx += f[q, i, j] * L.E[q, 0]
            my += f[q, i, j] * L.E[q, 1]
        rho[i, j] = r
        u[0, i, j] = mx / r
        u[1, i, j] = my / r

@ti.kernel
def collide(tau: ti.f32):
    for i, j in ti.ndrange(nx, ny):        # parallel over cells

        r = 0.0; mx = 0.0; my = 0.0
        for q in range(L.Q):
            r += f[q, i, j]
            mx += f[q, i, j] * L.E[q, 0]
            my += f[q, i, j] * L.E[q, 1]
        ux = mx / r
        uy = my / r
        usqr = ux*ux + uy*uy

        # 2. equilibrium + BGK relax, per direction
        for q in range(L.Q):
            eu = L.E[q,0]*ux + L.E[q,1]*uy
            feq = L.W[q] * r * (1 + (3 * eu) + (4.5 * eu*eu) - (1.5 * usqr))
            f[q,i,j] += -(1/tau)*(f[q,i,j]-feq)

@ti.kernel
def _stream():
    for i, j in ti.ndrange(nx, ny):        # parallel over destination cells
        for q in range(L.Q):
            src_i = (i - L.E[q,0]) % nx     # where this population came from
            src_j = (j - L.E[q,1]) % ny     # % nx/ny = periodic wrap
            f_new[q, i, j] = f[q, src_i, src_j]

def stream():                        # plain Python: kernel + buffer swap
    _stream()
    f.copy_from(f_new)

@ti.kernel
def bounce_back():
    for i, j in ti.ndrange(nx, ny):        # parallel over cells
        if solid[i, j] == 1:               # only at walls
            for q in range(L.Q):
                o = L.OPP[q]
                if q < o:                  # visit each opposite-pair once
                    tmp = f[q, i, j]
                    f[q, i, j] = f[o, i, j]
                    f[o, i, j] = tmp

@ti.kernel
def moving_wall(U: ti.f32):
    for i, j in ti.ndrange(nx, ny):
        if lid[i, j] == 1:
            for q in range(L.Q):
                o = L.OPP[q]
                if q < o:
                    tmp = f[q, i, j]; f[q, i, j] = f[o, i, j]; f[o, i, j] = tmp  # bounce-back
                    corr = 6.0 * L.W[q] * L.E[q, 0] * U
                    f[q, i, j] += corr
                    f[o, i, j] -= corr        # opposite dir gets the negative