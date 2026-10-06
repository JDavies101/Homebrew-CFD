# diagnostic: wing force split by surface region (side x chord station) from the per-link momentum exchange
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.engine import lattice_d3q19 as d3q19
from src.run.case_file import load_case_file, build_case
from src.run.runner import run_case

sample_every = 250
station_count = 10

def main():
    """
    Run one case file and print the mean C_y and C_x contribution of each (side, chord station) region of part 1.
    """

    case_file = load_case_file(sys.argv[1])
    output_path = Path(sys.argv[2])
    case = build_case(case_file, "cuda")
    part = case.parts[0]
    solid = part.solid[:, :, 0].astype(bool)
    solid_columns = np.nonzero(solid.any(axis=1))[0]
    leading_x = solid_columns.min()
    trailing_x = solid_columns.max()

    # mid-height of the solid in each column: a fluid node below it is on the lower (ground-facing) side
    middle_y = np.full(solid.shape[0], np.nan)
    for column in solid_columns:
        rows = np.nonzero(solid[column])[0]
        middle_y[column] = 0.5 * (rows.min() + rows.max())

    sums = {}

    def accumulate(sim, time_step):
        """
        Every sample_every averaging steps, add each link's momentum exchange c (f_in + f_out) to its region.
        """

        sums["sim"] = sim
        if time_step < case.warmup_steps() or time_step % sample_every != 0:
            return False

        count = sim.link_count
        directions = sim.link_direction.to_numpy()[:count]
        nodes = sim.link_node.to_numpy()[:count]
        incoming = sim.fc.to_numpy()[directions, nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        outgoing = sim.f.to_numpy()[d3q19.opposite_direction[directions], nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        contribution = (incoming + outgoing)[:, None] * d3q19.lattice_velocities[directions]
        if "links" not in sums:
            column = np.clip(nodes[:, 0], leading_x, trailing_x)
            side = np.where(nodes[:, 1] < middle_y[column], "lower", "upper")
            station = np.clip(((nodes[:, 0] - leading_x) / (trailing_x - leading_x + 1) * station_count).astype(int), 0, station_count - 1)
            sums["links"] = (side, station)
            sums["force"] = np.zeros((count, 3))
            sums["samples"] = 0
        sums["force"] += contribution
        sums["samples"] += 1

        # time-averaged momentum flux Pi = sum c c f on the z = 0 layer, for the box balance
        populations = sim.f.to_numpy()[:, :, :, 0].astype(np.float64)
        velocities = d3q19.lattice_velocities.astype(np.float64)
        for name, (a, b) in (("flux_xx", (0, 0)), ("flux_xy", (0, 1)), ("flux_yy", (1, 1))):
            flux = np.einsum("d,d,dij->ij", velocities[:, a], velocities[:, b], populations)
            sums[name] = sums.get(name, 0.0) + flux

        return False

    result = run_case(case, backend="cuda", after_step=accumulate, path=str(output_path.with_suffix(".record.csv")))
    side, station = sums["links"]
    dynamic_pressure_area = 0.5 * case.flow.free_stream_velocity * case.flow.free_stream_velocity * part.reference_area
    coefficients = sums["force"] / sums["samples"] / dynamic_pressure_area

    lines = [f"case {sys.argv[1]}", f"run C_y {np.mean(np.array(result.force_coefficients)[:, 0, 1]):.4f}, link sum C_y {coefficients[:, 1].sum():.4f}, C_x {coefficients[:, 0].sum():.4f}",
             "side station x_from x_to links C_x C_y"]
    for side_name in ("lower", "upper"):
        for index in range(station_count):
            selected = (side == side_name) & (station == index)
            lines.append(f"{side_name} {index} {index / station_count:.1f} {(index + 1) / station_count:.1f} {int(selected.sum())} "
                         f"{coefficients[selected, 0].sum():+.5f} {coefficients[selected, 1].sum():+.5f}")
    # independent check: steady momentum balance on a box around the wing, F = -(flux of Pi = sum c c f through the box)
    flux_xx = sums["flux_xx"] / sums["samples"]
    flux_xy = sums["flux_xy"] / sums["samples"]
    flux_yy = sums["flux_yy"] / sums["samples"]
    lowest_y = int(np.nonzero(solid.any(axis=0))[0].min())
    for margin in (6, 10, 16):
        x_start = leading_x - 4 * margin
        x_end = trailing_x + 6 * margin
        y_start = lowest_y - margin
        y_end = lowest_y + 6 * margin
        force_y = -((flux_xy[x_end, y_start:y_end].sum() - flux_xy[x_start, y_start:y_end].sum())
                    + (flux_yy[x_start:x_end, y_end].sum() - flux_yy[x_start:x_end, y_start].sum()))
        force_x = -((flux_xx[x_end, y_start:y_end].sum() - flux_xx[x_start, y_start:y_end].sum())
                    + (flux_xy[x_start:x_end, y_end].sum() - flux_xy[x_start:x_end, y_start].sum()))
        depth = case.domain.nz
        lines.append(f"box margin {margin}: C_y {force_y * depth / dynamic_pressure_area:+.4f}, C_x {force_x * depth / dynamic_pressure_area:+.4f} "
                     f"(x {x_start}-{x_end}, y {y_start}-{y_end})")

    output_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
