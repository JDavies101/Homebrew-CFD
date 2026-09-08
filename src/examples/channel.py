# forced plane-channel golden oracle: laminar Poiseuille, peak = g*delta^2/(2 nu)
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice3d as L
from src.post.progress import Progress
from src.geometry.step import step

S = 0                       # S=0 -> step() gives the two no-slip wall rows only
ny = 66
nx = 8
nz = 8
x_step = 0
U_c = 0.05
tau = 0.8
nu = (tau - 0.5) / 3
delta = (ny - 2) / 2
gx = 2 * nu * U_c / delta ** 2
TRT = 1                     # 1 -> TRT (Lambda=3/16), 0 -> BGK; run both, compare peaks
steps = 40000
check_every = 500           # progress readout interval


def fit_parabola(prof):
    # prof: u_x averaged over x,z, shape (ny,). Fit fluid nodes to a*(d^2 - y^2).
    j = np.arange(1, len(prof) - 1)            # fluid nodes only
    y = j - (len(prof) - 1) / 2.0              # centerline coords (halfway walls)
    u = prof[1:-1]
    c1, c0 = np.polyfit(y**2, u, 1)            # u ~ c1*y^2 + c0  =>  a=-c1, a*d^2=c0
    a = -c1
    delta_fit = np.sqrt(c0 / a)
    peak = c0                                  # value at y=0
    resid = u - (c1 * y**2 + c0)
    r2 = 1.0 - resid.var() / u.var()
    return r2, delta_fit, peak

def main():
    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(step(nx, ny, nz, x_step, S))
    sim.f.from_numpy(np.tile(L.W[:,None,None,None], (1,nx,ny,nz)).astype(np.float32))
    
    prog = Progress(steps)
    for s in range(steps):
        sim.collide_full(tau, 0.0, gx, TRT)
        sim.stream()
        sim.bounce_back()          # the two wall rows, no-slip

        if s % check_every == 0:
            sim.macroscopic()
            hmax = float(np.nanmax(np.abs(sim.u.to_numpy())))
            prog.update(s, hmax)

    sim.macroscopic()
    prof = sim.u.to_numpy()[0].mean(axis=(0, 2))   # u_x averaged over x,z -> (ny,)
    r2, delta_fit, peak = fit_parabola(prof)
    U_analytic = gx * delta**2 / (2 * nu)
    op = "TRT" if TRT else "BGK"
    print(f"\n[{op}]  R^2 = {r2:.6f}   delta_fit = {delta_fit:.3f} (nominal {delta:.1f})"
          f"   peak = {peak:.5f}   analytic = {U_analytic:.5f}"
          f"   ({100*(peak/U_analytic - 1):+.2f}%)")

if __name__ == "__main__":
    main()
