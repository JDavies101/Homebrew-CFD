import numpy as np
from src.geometry.ahmed_body import ahmed_body
from src.post.plotting import plot_mask_slice, plot_velocity_slice
from src.engine.simulation3d import Simulation3D
from src.post.progress import Progress
from src.post.vtk import write_field
import sys
from src.geometry.sponge import sponge, relax_profile
from src.post.run_log import RunRecord

H = 32                      # small for iteration; go to 48 for real runs
Lb = round(1044/288 * H)
Wb = round(389 / 288 * H)
x0 = Lb                       # 1 body-length upstream
nx = 6 * Lb                   # up + body + 4-length wake
ny = round(3.75 * H)          # ~2.5 H of air above the body
nz = round(3.667 * H)         # ~10% blockage
U = 0.05                    # low Mach
Re_H = 30000                  # laminar-ish separated wake; machinery, not the reference Cd
nu = U * H / Re_H           # ~0.0053 -> tau ~0.516 (TRT)
tau = 3*nu + 0.5
A = Wb * H                 # frontal area for Cd
T_ft = nx / U                        # flow-through time in steps (~14000 at H=32)
warmup = round(5 * T_ft)             # establish flow + wake
steps  = round(11 * T_ft)            # + ~6 flow-throughs (~30 shedding periods) to average
#warmup = round(3 * T_ft)
#steps  = round(6 * T_ft)
#steps = 15000 
ramp = round(T_ft)
sample_every = 25   # sample force every N steps (avoids per-step GPU sync)
check_every  = max(1, steps // 300) # progress readout interval
sgs = sys.argv[3] if len(sys.argv) > 3 else "wale"      # "smag" or "wale"
cs_floor = 0.04 # Smagorinsky floor under WALE: damps grid-scale noise where WALE's nut ~ 0
cs = 0.084 if sgs == "smag" else cs_floor
cw = 0.5
gx = 0.0
trt = 1
y1 = 0.5 # wall distance, halfway bounce back
sponge_width = 8                    # absorbing layer at inlet/outlet/side walls, cells
sponge_nu = 0.0                     # peak sponge viscosity; 0.0 disables the sponge
relax_width = 24                      # x layers (inlet/outlet), cells
relax_width_z = 12                    # z-wall layers, cells
relax_sigma = 0.1
relax_alpha = 1.0 / 2000.0  # z-layer running-mean rate (about a 2000-step memory)
outlet_bc = "pressure"              # "pressure" (rho = 1 at the exit) or "copy" (zero-gradient)
phi = int(sys.argv[1]) if len(sys.argv) > 1 else 25   # slant angle, deg
nose = sys.argv[2] if len(sys.argv) > 2 else "round"   # "round" or "square"
tag = f"phi{phi}" + ("" if nose == "round" else f"_{nose}") + ("" if sgs == "smag" else f"_{sgs}")

# block-averaged mean and standard error of a correlated series
def block_stats(x, nb):
    n = len(x) // nb * nb                   # drop the ragged tail
    means = np.asarray(x[:n]).reshape(nb, -1).mean(axis=1)
    return means.mean(), means.std(ddof=1) / np.sqrt(nb)

# mean flow in the first fluid cell above the slant, mid-span. u_t = velocity along the slant
# (downstream-and-down); u_t > 0 means the mean flow follows the surface (attached), < 0 reversed
def slant_check(u_mid, body_mid):
    t = np.array([np.cos(np.radians(phi)), -np.sin(np.radians(phi))])
    x_rear = x0 + Lb
    slant_dx = round(round(222 / 288 * H) * np.cos(np.radians(phi)))
    rows = []
    for i in range(x_rear - slant_dx, x_rear):
        j = np.nonzero(body_mid[i])[0].max() + 1            # first fluid cell above the surface
        rows.append((i, j, u_mid[0, i, j] * t[0] + u_mid[1, i, j] * t[1]))
    ut = np.array([r[2] for r in rows])
    print(f"slant u_t/U (first fluid cell, {len(ut)} columns):", np.round(ut / U, 3))
    print(f"attached fraction (u_t > 0.05 U): {np.mean(ut > 0.05 * U):.2f}")   # threshold: ~0 is dead water, not attached

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")

    body = ahmed_body(nx, ny, nz, x0, H, phi, nose)

    plot_mask_slice(body, axis=2, index=nz//2)[0].savefig("results/ahmed_xy.png")   # side view
    plot_mask_slice(body, axis=1, index=10)[0].savefig("results/ahmed_xz.png")      # plan view

    solid = body.copy()
    solid[:, 0, :] = 1
    solid[:, -1, :] = 1   # + floor & ceiling (no-slip)
    sim.solid.from_numpy(solid)
    sim.body.from_numpy(body)
    sim.build_wall_list()

    if sponge_nu > 0.0:
        sim.nut_sponge.from_numpy(sponge(nx, ny, nz, width=sponge_width, nu_max=sponge_nu))

    sim.init_equilibrium(np.zeros((nx, ny, nz), np.float32),   # start from rest, inlet ramps up
                     np.zeros((nx, ny, nz), np.float32),
                     np.zeros((nx, ny, nz), np.float32))
    
    sim.sigma.from_numpy(relax_profile(nx, nz, relax_width, 0, relax_sigma))       # x layers only, free-stream target
    sim.sigma_z.from_numpy(relax_profile(nx, nz, 0, relax_width_z, relax_sigma))   # z layers only, running-mean target                                                       # mean starts at rest density
    
    u_sum = np.zeros((3, nx, ny, nz), np.float32)
    n_u = 0
    mean_every = 500

    cd = []
    run = RunRecord("ahmed", sim, steps=steps, u_ref=U, nu=nu, tau=round(tau, 6), Re=Re_H,
                    geometry=f"H={H} phi={phi} {nose}", warmup=warmup, avg_Tft=round((steps - warmup) / T_ft, 1),
                    collision="regularized", sgs=f"smag cs={cs}" if sgs == "smag" else f"wale cw={cw} + smag floor cs={cs}",
                    wall_model="log-law y+>30", walls="staircase BB", forcing="none",
                    boundaries=f"NEEM-open inlet / {outlet_bc} outlet / free-slip z / no-slip floor+ceiling",
                    sponge=f"relax x {relax_width} (free stream) / z {relax_width_z} (running mean from ramp end, alpha {relax_alpha:.1e}), sigma {relax_sigma}")
    
    prog = Progress(steps)
    for s in range(steps):
        if sgs == "wale":
            sim.macroscopic()               # WALE needs current u (not needed on the smag path)
            sim.les_wale(cw)
        sim.wall_model_fast(nu, y1)
        r = min(s / ramp, 1.0)
        U_in = U * 0.5 * (1.0 - np.cos(np.pi * r))
        if s == ramp:
            sim.macroscopic()
            sim.rho_bar.copy_from(sim.rho)
            sim.u_bar.copy_from(sim.u)
        zon = 1 if s >= ramp else 0
        sim.collide_reg(tau, cs, gx, U_in, relax_alpha, zon)
        if s >= warmup and s % mean_every == 0:
            sim.macroscopic()
            u_sum += sim.u.to_numpy()
            n_u += 1
        if s >= warmup and s % sample_every == 0:
            sim.drag_body()                                          # only on sample steps now
            cd.append(float(sim.force.to_numpy()[0]) / (0.5*U*U*A))
        sim.stream()
        sim.inlet_neem_open(U_in)        # reuse: drives open rows, skips the solid floor at the inlet
        if outlet_bc == "pressure":
            sim.outlet_pressure(1.0)
        else:
            sim.outlet()
        sim.free_slip_z()                # side walls
        sim.bounce_back()                # floor + ceiling + body

        if s % check_every == 0:
            sim.macroscopic()
            prog.update(s, float(np.nanmax(np.abs(sim.u.to_numpy()))))

    prog.done()                          # finish the bar (newline) before any other output
    run.stop()
    sim.macroscopic()

    u_mean = u_sum / n_u
    kmid = nz // 2
    np.save(f"results/ahmed_umean_mid_{tag}.npy", u_mean[:, :, :, kmid])   # mid-span mean, re-analysis
    slant_check(u_mean[:, :, :, kmid], body[:, :, kmid])
    u_mean[:, body.astype(bool)] = np.nan                    # hide the body interior
    fig, ax = plot_velocity_slice(u_mean, axis=2, index=nz//2, comp=0)
    fig.savefig(f"results/ahmed_mean_{tag}.png", dpi=130)
    ax.set_xlim(x0 - 20, x0 + 60); ax.set_ylim(0, 2 * H)
    fig.savefig(f"results/ahmed_mean_nose_{tag}.png", dpi=160)
    ax.set_xlim(x0 + Lb - 50, x0 + Lb + 80); ax.set_ylim(0, 2 * H)
    fig.savefig(f"results/ahmed_mean_slant_{tag}.png", dpi=160)

    u = sim.u.to_numpy()
    write_field(f"results/ahmed_{tag}", sim.rho.to_numpy(), u)
    plot_velocity_slice(u, axis=2, index=nz//2, comp=0)[0].savefig(f"results/ahmed_wake_{tag}.png", dpi=130)
    cd = np.asarray(cd)
    np.save(f"results/ahmed_cd_{tag}.npy", cd)            # raw series for re-analysis
    
    for nb in (5, 10, 20):
        m, se = block_stats(cd, nb)
        print(f"Cd = {m:.4f} +/- {se:.4f}   ({nb} blocks)")
    h = len(cd) // 2
    print(f"drift: 1st half {cd[:h].mean():.4f}, 2nd half {cd[h:].mean():.4f}")
    print("wall-engaged nodes:", int((sim.nut_wall.to_numpy() > 0).sum()))

    if sgs == "wale":
        sim.macroscopic()
        sim.les_wale(cw)
        nl = sim.nut_les.to_numpy()
        fl = sim.solid.to_numpy() == 0
        print(f"nut_les/nu: mean {nl[fl].mean() / nu:.2f}  max {nl[fl].max() / nu:.2f}")

    m5, se5 = block_stats(cd, 5)
    run.finish(metric="Cd", value=round(float(m5), 4), se=round(float(se5), 4),
               drift_1st=round(float(cd[:h].mean()), 4), drift_2nd=round(float(cd[h:].mean()), 4))

if __name__ == "__main__":
    main()
