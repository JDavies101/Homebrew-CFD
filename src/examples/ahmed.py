import numpy as np
from src.geometry.ahmed_body import ahmed_body
from src.post.plotting import plot_mask_slice, plot_velocity_slice
from src.engine.simulation3d import Simulation3D
from src.post.progress import Progress
from src.post.vtk import write_field
import sys

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
sample_every = 25   # sample force every N steps (avoids per-step GPU sync)
check_every  = max(1, steps // 300) # progress readout interval
cs = 0.084
gx = 0.0
trt = 1
y1 = 0.5 # wall distance, halfway bounce back
phi = int(sys.argv[1]) if len(sys.argv) > 1 else 25   # slant angle, deg
nose = sys.argv[2] if len(sys.argv) > 2 else "round"   # "round" or "square"
tag = f"phi{phi}" + ("" if nose == "round" else f"_{nose}")   # output-file suffix

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

    sim.init_equilibrium(np.full((nx, ny, nz), U, np.float32),   # u_x = U everywhere
                     np.zeros((nx, ny, nz), np.float32),
                     np.zeros((nx, ny, nz), np.float32))
    
    u_sum = np.zeros((3, nx, ny, nz), np.float32)
    n_u = 0
    mean_every = 500

    cd = []
    prog = Progress(steps)
    for s in range(steps):
        #sim.macroscopic()
        sim.wall_model_fast(nu, y1)
        sim.collide_reg(tau, cs, gx)
        if s >= warmup and s % mean_every == 0:
            sim.macroscopic()
            u_sum += sim.u.to_numpy()
            n_u += 1
        if s >= warmup and s % sample_every == 0:
            sim.drag_body()                                          # only on sample steps now
            cd.append(float(sim.force.to_numpy()[0]) / (0.5*U*U*A))
        sim.stream()
        sim.inlet_neem_open(U)           # reuse: drives open rows, skips the solid floor at the inlet
        sim.outlet()
        sim.free_slip_z()                # side walls
        sim.bounce_back()                # floor + ceiling + body

        if s % check_every == 0:
            sim.macroscopic()
            prog.update(s, float(np.nanmax(np.abs(sim.u.to_numpy()))))

    prog.done()                          # finish the bar (newline) before any other output
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

if __name__ == "__main__":
    main()
