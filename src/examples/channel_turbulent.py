# turbulent plane channel Re_tau=180 -> log law (Moser-Kim-Mansour); DNS, cs=0
# milestone: trip and *sustain* turbulence (U_c bounded near ~17.7 u_tau, RMS steady).
# at fixed forcing, relaminarizing means running away to the laminar state -> diverges.
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.geometry.step import step

Re_tau = 180
u_tau = 0.0045              # small -> U_c ~ 17.7 u_tau stays low Mach
delta = 64                  # half-height in cells; y+ at first node ~ 0.5*Re_tau/delta ~ 1.4
ny = 2 * delta + 2          # solid rows j=0, j=ny-1; halfway walls -> H = ny-2 = 2*delta
nu = u_tau * delta / Re_tau # = 0.0016
tau = 3 * nu + 0.5          # ~0.505, near 0.5 -> TRT mandatory, BGK would blow up
gx = u_tau ** 2 / delta     # tau_w = rho*g*delta  ->  u_tau = sqrt(g*delta)
U_c = 17.7 * u_tau          # turbulent centerline from the log law (~0.08)
nx = 120                    # minimal channel (Jimenez-Moin): L_x+ ~ 340
nz = 60                     #                                  L_z+ ~ 170

n_z = 2                     # spanwise roll pairs across the box
beta = 2 * np.pi * n_z / nz
v_roll = 0.10 * U_c         # roll (u_y) amplitude
a_streak = 0.10 * U_c       # streak (u_x) amplitude
a_noise = 0.05 * U_c        # broadband noise amplitude

trt = 1
cs = 0.0                    # DNS: Smagorinsky off (over-damps, relaminarizes at this Re)
steps = 200000              # eddy turnover ~ delta/u_tau ~ 1.4e4 -> ~14 turnovers
check_every = 2000
warmup = 30000              # discard transient before averaging
sample_every = 200           # sample force every N steps (avoids per-step GPU sync)

rng = np.random.default_rng(0)   # reproducible IC across trip attempts

def add_rolls_and_streaks(u_x, u_y, u_z, ETA, Z):   # SSP seed: div-free rolls + streaks (Jake)
    A_psi = v_roll / beta
    window = (1 - ETA ** 2) ** 2
    u_y += -A_psi * beta * window * np.sin(beta * Z)
    u_z += (A_psi / delta) * 4.0 * ETA * (1 - ETA ** 2) * np.cos(beta * Z)
    u_x += a_streak * window * np.sin(beta * Z)
    return(u_x, u_y, u_z)


def initial_velocity():          # mean parabola + rolls/streaks + noise
    y = np.arange(ny) - (ny - 1) / 2.0
    ETA = (y / delta)[None, :, None]
    Z = np.arange(nz)[None, None, :]

    u_x = np.broadcast_to(U_c * (1 - ETA ** 2), (nx, ny, nz)).astype(np.float32).copy()
    u_y = np.zeros((nx, ny, nz), np.float32)
    u_z = np.zeros((nx, ny, nz), np.float32)

    add_rolls_and_streaks(u_x, u_y, u_z, ETA, Z)

    # noise is the only x-dependent term -> lets the x-invariant rolls break down to 3D
    u_x += (a_noise * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)
    u_y += (a_noise * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)
    u_z += (a_noise * (2 * rng.random((nx, ny, nz)) - 1)).astype(np.float32)
    return u_x, u_y, u_z


def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(step(nx, ny, nz, 0, 0))     # S=0 -> the two wall rows only

    u_x, u_y, u_z = initial_velocity()
    sim.init_equilibrium(u_x, u_y, u_z)

    uc = []
    urms = []
    vrms = []
    wrms = []
    prog = Progress(steps)
    for s in range(steps):
        sim.collide_full(tau, cs, gx, trt)
        sim.stream()
        sim.bounce_back()

        if s >= warmup and s % sample_every == 0:
            sim.macroscopic()
            u = sim.u.to_numpy()
            uc.append(float(u[0][:, ny // 2, :].mean()))          # centerline mean u_x
            up = u[0] - u[0].mean(axis=(0, 2), keepdims=True) # fluctuation off x-z mean profile
            urms.append(float(np.sqrt((up ** 2).mean())) / u_tau)
            vrms.append(float(np.sqrt((u[1] ** 2).mean())) / u_tau)
            wrms.append(float(np.sqrt((u[2] ** 2).mean())) / u_tau)

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    prog.done()
    print(f"  U_c/u_tau={np.mean(uc)/u_tau:.2f}  rms(u,v,w)/u_tau="
      f"({np.mean(urms):.2f},{np.mean(vrms):.2f},{np.mean(wrms):.2f})  (avg of {len(uc)} samples)")
    # next: <u>+(y+) with fitted delta_eff vs log law + MKM; peak u'_rms/u_tau ~ 2.7 at y+~15

if __name__ == "__main__":
    main()
