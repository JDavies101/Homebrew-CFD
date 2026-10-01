# command-line solve: python -m src.run case.json [--runs DIR] [--backend cuda|cpu] [--dev-log]
import argparse
import numpy as np
from src import __version__
from src.post.statistics import block_statistics
from src.run.case_file import load_case_file, save_case_file, build_case
from src.run.run_folder import case_hash, create_run_folder, write_json_atomic
from src.run.runner import run_case

def part_statistics(series):
    """
    Mean, standard error (5 blocks) and half-record drift of each coefficient component.

    Returns {"x": {...}, "y": {...}, "z": {...}}.
    """

    half = len(series) // 2
    statistics = {}
    for axis, axis_name in enumerate(("x", "y", "z")):
        mean, standard_error = block_statistics(series[:, axis], 5)
        statistics[axis_name] = {"mean": float(mean), "se": float(standard_error),
                                 "drift_first": float(series[:half, axis].mean()), "drift_second": float(series[half:, axis].mean())}

    return statistics

def main():
    """
    Load a case file, run it into a new run folder, write progress, results and the run record.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("case_path")
    parser.add_argument("--runs", default="runs")  # parent directory of run folders
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dev-log", action="store_true")  # append the record to docs/run_log.csv instead of the run folder
    options = parser.parse_args()

    # run folder: the case as it will run, then live progress from the start
    case_file = load_case_file(options.case_path)
    folder = create_run_folder(options.runs, case_file.name)
    save_case_file(case_file, folder / "case.json")
    digest = case_hash(case_file)
    progress_path = folder / "progress.json"
    stop_path = folder / "STOP"
    write_json_atomic(progress_path, {"status": "building", "step": 0, "steps": 0})
    print(f"run folder: {folder}")

    def report_progress(time_step, steps, max_velocity):
        """
        Publish progress; a STOP file in the run folder ends the run cleanly.

        Returns True to stop.
        """

        write_json_atomic(progress_path, {"status": "running", "step": time_step + 1, "steps": steps, "max_velocity": max_velocity})

        return stop_path.exists()

    case = build_case(case_file, options.backend)
    record_path = None if options.dev_log else str(folder / "record.csv")
    result = run_case(case, backend=options.backend, progress_callback=report_progress, path=record_path,
                      geometry=", ".join(f"{spec.name} ({spec.kind})" for spec in case_file.geometry),
                      walls=", ".join(f"{spec.name} {spec.wall}" for spec in case_file.geometry),
                      boundaries=f"inlet {case.inlet} / x {case.domain.x_boundary} / y {case.domain.y_boundary} / z {case.domain.side_walls}")

    # results: per-part coefficient statistics and history
    status = "finished"
    if result.blow_up_step >= 0:
        status = "blow_up"
    elif result.stopped_step >= 0:
        status = "stopped"
    coefficients = result.force_coefficients
    np.save(folder / "force_coefficients.npy", coefficients)
    parts = {}
    if len(coefficients) > 0:
        parts = {spec.name: part_statistics(coefficients[:, index, :]) for index, spec in enumerate(case_file.geometry)}
    write_json_atomic(folder / "result.json", {"status": status, "case_hash": digest, "version": __version__, "backend": options.backend,
                                               "steps": case.total_steps(), "blow_up_step": result.blow_up_step,
                                               "stopped_step": result.stopped_step, "parts": parts})
    write_json_atomic(progress_path, {"status": status, "step": case.total_steps(), "steps": case.total_steps(), "steps_completed": result.steps_completed, "warmup": case.warmup_steps(),})

    # run record: first part's y coefficient as the headline (the wing's -CL is its negative)
    if parts:
        first_name = case_file.geometry[0].name
        first = parts[first_name]
        result.run.finish(metric=f"{first_name} C_y", value=round(first["y"]["mean"], 4), se=round(first["y"]["se"], 4),
                          drift_1st=round(first["y"]["drift_first"], 4), drift_2nd=round(first["y"]["drift_second"], 4),
                          other=f"C_x {first['x']['mean']:.4f}; status {status}; case {digest[:12]}; v{__version__}")
    else:
        reason = f"stopped before averaging began (step {result.steps_completed} of warmup {case.warmup_steps()})" if status != "finished" else "no parts"
        print(reason)
        result.run.finish(metric="status", value=status, other=f"{reason}; case {digest[:12]}; v{__version__}")

if __name__ == "__main__":
    main()