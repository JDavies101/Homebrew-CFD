# forced plane-channel golden oracle: laminar Poiseuille, peak = g delta^2 / (2 nu)
import numpy as np
from src.engine.simulation3d import Simulation3D
from src.engine import lattice_d3q19 as d3q19
from src.geometry.step_body import step
from src.post.progress import Progress
from src.post.run_log import RunRecord

# geometry: step with height 0 gives the two no-slip wall rows only
nx = 8
ny = 66
nz = 8
half_height = (ny - 2) / 2  # delta

# flow
centerline_velocity = 0.05  # U_c
relaxation_time = 0.8
viscosity = (relaxation_time - 0.5) / 3
body_force_x = 2 * viscosity * centerline_velocity / half_height ** 2  # g
trt = 1  # 1 -> TRT (Lambda = 3/16), 0 -> BGK; run both, compare peaks

# timing
steps = 40000
check_every = 500  # progress readout interval

def fit_parabola(profile):
    """
    Fit the fluid nodes of u_x(y) (averaged over x, z; shape (ny,)) to u = c1 y^2 + c0, y from the centerline.

    Returns (R^2, fitted half-height, peak velocity).
    """

    j = np.arange(1, len(profile) - 1)  # fluid nodes only
    y = j - (len(profile) - 1) / 2.0  # centerline coordinates (halfway walls)
    velocity = profile[1:-1]
    c1, c0 = np.polyfit(y ** 2, velocity, 1)  # u = a (d^2 - y^2): a = -c1, a d^2 = c0
    curvature = -c1
    fitted_half_height = np.sqrt(c0 / curvature)
    peak = c0  # value at y = 0
    residual = velocity - (c1 * y ** 2 + c0)
    r_squared = 1.0 - residual.var() / velocity.var()

    return r_squared, fitted_half_height, peak

def main():
    """
    Run the forced channel and compare the peak with the analytic Poiseuille value.
    """

    sim = Simulation3D(nx, ny, nz, backend="cuda")
    sim.solid.from_numpy(step(nx, ny, nz, 0, 0))
    sim.f.from_numpy(np.tile(d3q19.lattice_weights[:, None, None, None], (1, nx, ny, nz)).astype(np.float32))

    run = RunRecord("channel_laminar", sim, steps=steps, tau=round(relaxation_time, 6),
                    geometry=f"delta={half_height}", collision="TRT" if trt else "BGK", sgs="none",
                    walls="halfway BB", boundaries="periodic x,z / no-slip y walls", forcing="body force gx")
    progress = Progress(steps)
    for time_step in range(steps):
        sim.collide_full(relaxation_time, 0.0, body_force_x, trt)
        sim.stream()
        sim.bounce_back()  # the two wall rows, no-slip

        if time_step % check_every == 0:
            sim.macroscopic()
            progress.update(time_step, float(np.nanmax(np.abs(sim.u.to_numpy()))))

    progress.done()
    run.stop()
    sim.macroscopic()
    profile = sim.u.to_numpy()[0].mean(axis=(0, 2))  # u_x averaged over x, z -> (ny,)
    r_squared, fitted_half_height, peak = fit_parabola(profile)
    analytic_peak = body_force_x * half_height ** 2 / (2 * viscosity)
    collision_name = "TRT" if trt else "BGK"
    print(f"[{collision_name}]  R^2 = {r_squared:.6f}   delta_fit = {fitted_half_height:.3f} (nominal {half_height:.1f})"
          f"   peak = {peak:.5f}   analytic = {analytic_peak:.5f}"
          f"   ({100 * (peak / analytic_peak - 1):+.2f}%)")
    run.finish(metric="peak u", value=round(float(peak), 6), reference=round(float(analytic_peak), 6),
               other=f"R^2 {r_squared:.6f}; delta_fit {fitted_half_height:.3f} (nominal {half_height:.1f})")

if __name__ == "__main__":
    main()
