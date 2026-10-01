# forced plane-channel golden oracle: laminar Poiseuille, peak = g delta^2 / (2 nu)
import numpy as np
from src.run.case import Flow, Domain, Turbulence, Timing, Case
from src.run.runner import run_case

# geometry: the two no-slip wall rows (y = 0, ny - 1), periodic x and z
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

    # at rest (f = w), driven by the body force
    case = Case(name="channel_laminar", tag="",
                flow=Flow(free_stream_velocity=centerline_velocity, relaxation_time_override=relaxation_time, body_force_x=body_force_x),
                domain=Domain(nx=nx, ny=ny, nz=nz, x_boundary="periodic", relax_width_x=0, layer_kind="separate"),
                turbulence=Turbulence(sgs="none", smagorinsky_constant=0.0),
                timing=Timing(steps_override=steps, warmup_override=0, sample_window="none", check_every=check_every),
                collision="trt" if trt else "bgk", inlet="none", start="rest_ramp")
    result = run_case(case, geometry=f"delta={half_height}", walls="halfway BB", boundaries="periodic x,z / no-slip y walls")
    sim = result.sim
    run = result.run

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
