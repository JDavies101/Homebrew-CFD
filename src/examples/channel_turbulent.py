# turbulent plane channel Re_tau=180 -> log law (Moser-Kim-Mansour); DNS, cs=0
# milestone: trip and *sustain* turbulence (U_c bounded near ~17.7 u_tau, RMS steady).
# at fixed forcing, relaminarizing means running away to the laminar state -> diverges.
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.geometry.step import step
import matplotlib.pyplot as plt

Re_tau = 180
u_tau = 0.0045              # small -> U_c ~ 17.7 u_tau stays low Mach
delta = 64                  # half-height in cells; y+ at first node ~ 0.5*Re_tau/delta ~ 1.4
ny = 2 * delta + 2          # solid rows j=0, j=ny-1; halfway walls -> H = ny-2 = 2*delta
nu = u_tau * delta / Re_tau # = 0.0016
tau = 3 * nu + 0.5          # ~0.505, near 0.5 -> TRT mandatory, BGK would blow up
gx = u_tau ** 2 / delta     # tau_w = rho*g*delta  ->  u_tau = sqrt(g*delta)
U_c = 17.7 * u_tau          # turbulent centerline from the log law (~0.08)
nx = round(337 * delta / Re_tau)   # hold L_x+ ~ 337 as delta shrinks (coarsening sweep)
nz = round(169 * delta / Re_tau)   # hold L_z+ ~ 169

n_z = 2                     # spanwise roll pairs across the box
beta = 2 * np.pi * n_z / nz
v_roll = 0.10 * U_c         # roll (u_y) amplitude
a_streak = 0.10 * U_c       # streak (u_x) amplitude
a_noise = 0.05 * U_c        # broadband noise amplitude

trt = 1
cs = 0.0                    # DNS: Smagorinsky off (over-damps, relaminarizes at this Re)
steps = round(600000 * delta / 64)       # ~constant turnovers across the delta sweep
check_every = 2000
warmup = steps // 4              # discard ~10 turnovers of transient
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
    sum_ux = []
    sum_uxx = []
    sum_uyy = []
    sum_uzz = []
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

            sum_ux.append(u[0].mean(axis=(0, 2)))
            sum_uxx.append((u[0] ** 2).mean(axis=(0, 2)))
            sum_uyy.append((u[1] ** 2).mean(axis=(0, 2)))
            sum_uzz.append((u[2] ** 2).mean(axis=(0, 2)))

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    prog.done()

    sum_ux  = np.array(sum_ux)     # (Nsamples, ny)  — or pass the list straight to np.mean
    ubar      = np.mean(sum_ux,  axis=0)                     # (ny,) mean profile  -> the log law
    urms_prof = np.sqrt(np.mean(sum_uxx, axis=0) - ubar**2)  # variance about the space-time mean
    vrms_prof = np.sqrt(np.mean(sum_uyy, axis=0))
    wrms_prof = np.sqrt(np.mean(sum_uzz, axis=0))

    delta_eff = delta + 0.33                       # body-force wall offset (from the laminar oracle)
    u_tau2 = np.sqrt(gx * delta_eff)               # friction velocity from the force balance (exact)
    u1 = 0.5 * (ubar[1] + ubar[-2])                # mean u_x at the first fluid node off each wall
    d1 = delta_eff - (delta - 0.5)                 # its wall distance in cells (~0.83)
    u_tau_grad = np.sqrt(nu * u1 / d1)             # resolved wall shear: tau_w = nu * du/dy
    y_j = np.arange(ny) - (ny - 1) / 2.0
    d = delta_eff - np.abs(y_j)                     # wall distance in cells (<0 on the solid rows)
    asym = float(np.abs(ubar - ubar[::-1]).max() / ubar.max())   # pre-fold top/bottom check

    fold = lambda a: 0.5 * (a + a[::-1])           # average the two symmetric halves
    y_plus = (fold(d) * u_tau2 / nu)
    u_plus = fold(ubar) / u_tau2
    urms_p = fold(urms_prof) / u_tau2
    vrms_p = fold(vrms_prof) / u_tau2
    wrms_p = fold(wrms_prof) / u_tau2
    sl = slice(1, ny // 2)                          # one half, fluid nodes only (drop the wall row)
    y_plus, u_plus = y_plus[sl], u_plus[sl]
    urms_p, vrms_p, wrms_p = urms_p[sl], vrms_p[sl], wrms_p[sl]

    yp_ref = np.logspace(np.log10(y_plus.min()), np.log10(y_plus.max()), 200)
    sublayer = yp_ref                               # u+ = y+
    loglaw = np.log(yp_ref) / 0.41 + 5.2            # u+ = (1/kappa) ln y+ + B, kappa=0.41 B=5.2

    ipk = int(urms_p.argmax())
    print(f"centerline U+   = {u_plus[-1]:.2f}         [MKM ~18.3]")
    print(f"u'_rms peak     = {urms_p[ipk]:.2f} at y+ = {y_plus[ipk]:.1f}   [MKM ~2.65 at y+~15]")
    print(f"first node y+   = {y_plus[0]:.2f}          [near-wall resolution]")
    print(f"wall-shear err  = {100*(u_tau_grad/u_tau2 - 1):+.1f}%          [->0 resolved, grows coarse]")
    print(f"top/bottom asym = {asym*100:.1f}%          [convergence, want a few %]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogx(y_plus, u_plus, "o-", ms=3, label="LBM")
    ax1.semilogx(yp_ref, sublayer, "k:", label="u+ = y+")
    ax1.semilogx(yp_ref, loglaw, "k--", label="log law")
    ax1.set(xlabel="y+", ylabel="u+", ylim=(0, 20), title="mean velocity"); ax1.legend()
    ax2.semilogx(y_plus, urms_p, label="u'")
    ax2.semilogx(y_plus, vrms_p, label="v'")
    ax2.semilogx(y_plus, wrms_p, label="w'")
    ax2.set(xlabel="y+", ylabel="rms / u_tau", title="fluctuations"); ax2.legend()
    fig.tight_layout(); fig.savefig("results/channel_loglaw.png", dpi=130)
    print("wrote results/channel_loglaw.png")

if __name__ == "__main__":
    main()
