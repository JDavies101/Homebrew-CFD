# command-line solve: python -m src.run case.json [--runs DIR] [--backend cuda|cpu] [--dev-log] [--preview OUT.npz]
import time
process_start = time.perf_counter() # launch timing: the imports below count toward start-up
import argparse
import sys
import numpy as np
from src import __version__
from src.post.statistics import block_statistics
from src.run.case_file import load_case_file, save_case_file, build_case
from src.run.run_folder import case_hash, create_run_folder, write_json_atomic
from src.run.runner import run_case
from src.geometry.preview import save_preview

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

class Tee:
    """
    Write to several text streams at once (console, if any, and the run's solver.log), flushing every write.
    """

    def __init__(self, *streams):
        """
        Keep the streams that exist (a windowed .exe has no console: sys.stdout is None).
        """

        self.streams = [stream for stream in streams if stream is not None]

    def write(self, text):
        """
        Write to every stream.
        """

        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        """
        Flush every stream.
        """

        for stream in self.streams:
            stream.flush()

def main():
    """
    Load a case file, run it into a new run folder, write progress, results and the run record.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("case_path")
    parser.add_argument("--runs", default="runs")  # parent directory of run folders
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dev-log", action="store_true")  # append the record to docs/run_log.csv instead of the run folder
    parser.add_argument("--run-folder", default=None)  # exact folder to create (the app chooses it); default: a new dated folder under --runs
    parser.add_argument("--preview", default=None)  # build the geometry only and save it to this .npz (no run folder, no solver)
    options = parser.parse_args()

    case_file = load_case_file(options.case_path)

    # geometry preview only: the parts exactly as the solver would get them, then stop
    if options.preview is not None:
        case = build_case(case_file, options.backend)
        save_preview(options.preview, case.parts, (case_file.domain.nx, case_file.domain.ny, case_file.domain.nz))
        print(f"Preview written: {options.preview}")
        return

    # run folder: the case as it will run, then live progress from the start
    folder = create_run_folder(options.runs, case_file.name, options.run_folder)

    # everything printed from here on (and any traceback) also goes to the run's solver.log
    log_file = open(folder / "solver.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Imports {time.perf_counter() - process_start:.1f} s")

    save_case_file(case_file, folder / "case.json")
    digest = case_hash(case_file)
    progress_path = folder / "progress.json"
    stop_path = folder / "STOP"
    write_json_atomic(progress_path, {"status": "building", "step": 0, "steps": 0})
    print(f"Run folder: {folder.resolve()}")

    live_path = folder / "coefficients_live.csv"
    live_path.write_text(",".join(f"{spec.name}_{axis}" for spec in case_file.geometry for axis in "xyz") + "\n")
    samples_written = 0

    def report_progress(time_step, steps, max_velocity, force_coefficients):
        """
        Publish progress and any new coefficient samples; time the first step; a STOP file ends the run cleanly.

        Returns True to stop.
        """

        nonlocal samples_written
        if time_step == 0:
            print(f"\nFirst step (kernel compile) {time.perf_counter() - timing['solver_ready']:.1f} s")
        if len(force_coefficients) > samples_written:
            with live_path.open("a") as live_file:
                for sample in force_coefficients[samples_written:]:
                    live_file.write(",".join(f"{value:.7g}" for part_row in sample for value in part_row) + "\n")
            samples_written = len(force_coefficients)

        phase = "warmup" if time_step < case.warmup_steps() else "averaging"
        write_json_atomic(progress_path, {"status": phase, "step": time_step + 1, "steps": steps, "max_velocity": max_velocity,
                                          "warmup": case.warmup_steps(), "flow_through_steps": case.flow_through_steps(),
                                          "sample_every": case.timing.sample_every})

        return stop_path.exists()

    def mark_solver_ready(sim):
        """
        Time the solver setup (Taichi init, field allocation, geometry upload).
        """

        timing["solver_ready"] = time.perf_counter()
        print(f"Solver setup (Taichi init, fields, upload) {timing['solver_ready'] - solver_start:.1f} s")

    timing = {}

    # geometry, then the solver (setup and first-step timings come from the hooks above)
    geometry_start = time.perf_counter()
    case = build_case(case_file, options.backend)
    print(f"Geometry {time.perf_counter() - geometry_start:.1f} s")
    solver_start = time.perf_counter()
    record_path = None if options.dev_log else str(folder / "record.csv")
    result = run_case(case, backend=options.backend, before_loop=mark_solver_ready, progress_callback=report_progress, path=record_path,
                      geometry=", ".join(f"{spec.name} ({spec.kind})" for spec in case_file.geometry),
                      walls=", ".join(f"{spec.name} {spec.wall}" for spec in case_file.geometry),
                      boundaries=f"inlet {case.inlet} / x {case.domain.x_boundary} / y {case.domain.y_boundary} / z {case.domain.side_walls}")

    loop_seconds = time.perf_counter() - timing["solver_ready"]
   
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
                                               "steps": case.total_steps(), "steps_completed": result.steps_completed, "warmup": case.warmup_steps(),
                                               "blow_up_step": result.blow_up_step, "stopped_step": result.stopped_step, "parts": parts,
                                               "cells": case.domain.nx * case.domain.ny * case.domain.nz, "loop_seconds": round(loop_seconds, 2)})
    write_json_atomic(progress_path, {"status": status, "step": result.steps_completed, "steps": case.total_steps()})

    # run record: first part's y coefficient as the headline (the wing's -CL is its negative)
    if parts:
        first_name = case_file.geometry[0].name
        first = parts[first_name]
        result.run.finish(metric=f"{first_name} C_y", value=round(first["y"]["mean"], 4), se=round(first["y"]["se"], 4),
                          drift_1st=round(first["y"]["drift_first"], 4), drift_2nd=round(first["y"]["drift_second"], 4),
                          other=f"C_x {first['x']['mean']:.4f}; status {status}; case {digest[:12]}; v{__version__}")
    else:
        reason = f"Stopped before averaging began (step {result.steps_completed} of warmup {case.warmup_steps()})" if status != "finished" else "No parts"
        print(reason)
        result.run.finish(metric="status", value=status, other=f"{reason}; case {digest[:12]}; v{__version__}")

if __name__ == "__main__":
    main()