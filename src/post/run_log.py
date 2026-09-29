"""
Run log: one row per solver run, appended to docs/run_log.csv (versioned with the code; results/ is gitignored).

Two calls per example:

    run = RunRecord("cylinder", sim, steps=steps, u_ref=U, nu=nu, tau=tau, Re=Re, collision="BGK")
    ... time loop ...
    run.stop()  # optional: exclude post-processing from the timing
    run.finish(metric="Cd", value=cd, reference=1.4)

RunRecord starts the wall-clock timer. finish() collects everything it can from the sim itself
(grid, cells, MLUPS, field health, wall-model engagement, nut_les, VRAM, backend, git commit,
command) and appends one row. Rows start as status "unreviewed"; the verdict (good / bad /
invalid / superseded / reference) and its reason are set during review.
"""
import csv
import datetime
import os
import subprocess
import sys
import time
import numpy as np

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "run_log.csv"))

# csv column names (the file schema: renaming one needs a migration of docs/run_log.csv)
FIELDS = [
    "run_id", "date", "commit", "case", "command", "status", "reason", "superseded_by",
    "geometry", "Re", "grid", "cells", "tau", "Ma", "steps", "warmup", "avg_Tft",
    "collision", "sgs", "wall_model", "walls", "boundaries", "sponge", "forcing",
    "metric", "value", "se", "drift_1st", "drift_2nd", "reference", "err_pct",
    "wall_engaged", "nut_les_nu_mean", "nut_les_nu_max", "max_u",
    "wall_time_s", "mlups", "other", "notes",
    "backend", "vram_fields_gb", "vram_gpu_gb", "vram_gpu_scope",
    "rho_min", "rho_max", "hot_cells", "hot_where",
]

# environment

def _git_commit():
    """
    Short hash of HEAD, with "+dirty" when tracked files have uncommitted changes.

    Returns the commit string, or "unknown".
    """

    try:
        commit_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        changes = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()

        return commit_hash + ("+dirty" if changes else "")
    except Exception:
        return "unknown"

def _command():
    """
    Rebuild the command line that launched this run.

    Returns "python -m <module> <args>".
    """

    main_module = sys.modules.get("__main__")
    spec = getattr(main_module, "__spec__", None)
    module_name = spec.name if spec is not None else os.path.basename(sys.argv[0])

    return " ".join(["python -m", module_name] + sys.argv[1:])

def _backend():
    """
    Taichi arch the runtime was initialised with.

    Returns the arch name, or "" when unknown.
    """

    try:
        from src.engine import runtime

        return runtime._arch or ""
    except Exception:
        return ""

def _dtype_bytes(dtype):
    """
    Bytes per element of a Taichi dtype, read from its name.

    Returns 8, 4, 2 or 1.
    """

    dtype_name = str(dtype)
    if "64" in dtype_name:
        return 8
    if "16" in dtype_name:
        return 2
    if "8" in dtype_name:
        return 1

    return 4

def fields_gb(sim):
    """
    Memory held by the sim's Taichi fields: what the solver itself allocated.

    Returns gigabytes.
    """

    total_bytes = 0
    for value in vars(sim).values():
        if "Field" in type(value).__name__ and hasattr(value, "shape") and hasattr(value, "dtype"):
            total_bytes += int(np.prod(value.shape)) * _dtype_bytes(value.dtype)

    return total_bytes / 1e9

def gpu_used_gb():
    """
    GPU memory in use: this process if the driver reports it (Linux / TCC), else the whole device.

    Returns (gigabytes, scope), or (None, "n/a") when nvidia-smi is unavailable.
    """

    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout
        for line in output.strip().splitlines():
            process_id, memory_mb = [part.strip() for part in line.split(",")[:2]]
            if process_id == str(os.getpid()) and memory_mb.replace(".", "").isdigit():
                return float(memory_mb) / 1024.0, "process"

        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10).stdout

        return float(output.strip().splitlines()[0]) / 1024.0, "device"
    except Exception:
        return None, "n/a"

# field health

def field_health(velocity, density, solid=None, u_ref=None, band=4):
    """
    Speed and density summary over fluid cells, plus where the hot cells (|u| > 2 u_ref) sit:
    counts in a band-cell layer on each domain face (x-, x+, y-, y+, z-, z+) and the interior.

    Returns a dict of run-log fields.
    """

    speed = np.sqrt((velocity.astype(np.float64) ** 2).sum(axis=0))
    fluid = np.ones(speed.shape, bool) if solid is None else (solid == 0)
    finite = np.isfinite(speed) & np.isfinite(density)
    health = {}
    nonfinite_count = int((fluid & ~finite).sum())
    valid = fluid & finite

    if valid.any():
        health["max_u"] = round(float(speed[valid].max()), 4)
        health["rho_min"] = round(float(density[valid].min()), 4)
        health["rho_max"] = round(float(density[valid].max()), 4)
    else:
        health["max_u"] = "nan"

    if u_ref:
        hot = valid & (speed > 2.0 * u_ref)
        health["hot_cells"] = int(hot.sum())
        axis_names = ["x", "y", "z"]
        location_counts = []
        near_any_face = np.zeros(speed.shape, bool)
        for axis in range(speed.ndim):
            index = np.arange(speed.shape[axis])
            for side, in_layer in (("-", index < band), ("+", index >= speed.shape[axis] - band)):
                broadcast_shape = [1] * speed.ndim
                broadcast_shape[axis] = speed.shape[axis]
                face_layer = np.broadcast_to(in_layer.reshape(broadcast_shape), speed.shape)
                near_any_face |= face_layer
                location_counts.append(f"{axis_names[axis]}{side}:{int((hot & face_layer).sum())}")
        location_counts.append(f"interior:{int((hot & ~near_any_face).sum())}")
        if nonfinite_count:
            location_counts.append(f"nonfinite:{nonfinite_count}")
        health["hot_where"] = " ".join(location_counts)
    elif nonfinite_count:
        health["hot_where"] = f"nonfinite:{nonfinite_count}"

    return health

# csv

def _migrate(path):
    """
    Rewrite an older log under the current header (new columns empty).
    """

    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        rows = list(reader)

    if header == FIELDS:
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})

def _next_id(path):
    """
    Next free run id in the log.

    Returns max(run_id) + 1, or 1 for a new log.
    """

    if not os.path.exists(path):
        return 1

    with open(path, newline="", encoding="utf-8") as file:
        run_ids = [int(row["run_id"]) for row in csv.DictReader(file) if str(row.get("run_id", "")).isdigit()]

    return max(run_ids, default=0) + 1

def log_run(path=None, **fields):
    """
    Low-level append (RunRecord.finish() is the normal entry point). cells, steps and wall_time_s,
    when all given, also fill in mlups.

    Returns the new run id.
    """

    path = path or LOG_PATH
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown run-log fields: {sorted(unknown)}")

    row = {key: "" for key in FIELDS}
    row.update(fields)
    if os.path.exists(path):
        _migrate(path)
    row["run_id"] = _next_id(path)
    row["date"] = datetime.datetime.now().isoformat(timespec="minutes")
    if not row["commit"]:
        row["commit"] = _git_commit()
    if not row["command"]:
        row["command"] = _command()
    if not row["status"]:
        row["status"] = "unreviewed"

    cells = fields.get("cells")
    steps = fields.get("steps")
    wall_time = fields.get("wall_time_s")
    if cells and steps and wall_time:
        row["mlups"] = round(cells * steps / wall_time / 1e6, 1)

    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    return row["run_id"]

# the one call examples use

class RunRecord:
    """
    Times one run and appends its row to the run log on finish().
    """

    def __init__(self, case, sim=None, steps=None, u_ref=None, nu=None, path=None, **fields):
        """
        Construct right before the time loop: the timer starts here, setup is excluded.
        """

        self.case = case
        self.sim = sim
        self.steps = steps
        self.u_ref = u_ref
        self.nu = nu
        self.path = path
        self.fields = dict(fields)
        self.start_time = time.time()
        self.stop_time = None

    def stop(self):
        """
        Optional: call right after the time loop so post-processing is not timed.
        """

        self.stop_time = time.time()

    def finish(self, u=None, rho=None, solid=None, **fields):
        """
        Collect run data and append the row. u / rho / solid: numpy arrays for runs without a Taichi sim.

        Returns the new run id.
        """

        wall_time = (self.stop_time or time.time()) - self.start_time
        row = dict(self.fields)
        row.update(fields)
        row["case"] = self.case
        row["wall_time_s"] = round(wall_time)
        if self.steps:
            row["steps"] = self.steps
        if self.u_ref:
            row.setdefault("Ma", round(self.u_ref * 3 ** 0.5, 3))
        if "reference" in row and "value" in row and "err_pct" not in row:
            try:
                row["err_pct"] = round(100.0 * (float(row["value"]) / float(row["reference"]) - 1.0), 2)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # fields straight from the sim
        sim = self.sim
        if sim is not None:
            try:
                sim.macroscopic()
                u = sim.u.to_numpy()
                rho = sim.rho.to_numpy()
                solid = sim.solid.to_numpy()
            except Exception:
                pass
            if hasattr(sim, "nut_wall"):
                row.setdefault("wall_engaged", int((sim.nut_wall.to_numpy() > 0).sum()))
            if hasattr(sim, "nut_les") and self.nu:
                eddy_viscosity = sim.nut_les.to_numpy()
                fluid = sim.solid.to_numpy() == 0
                if eddy_viscosity[fluid].max() > 0:
                    row.setdefault("nut_les_nu_mean", round(float(eddy_viscosity[fluid].mean() / self.nu), 2))
                    row.setdefault("nut_les_nu_max", round(float(eddy_viscosity[fluid].max() / self.nu), 2))
            row["vram_fields_gb"] = round(fields_gb(sim), 3)

        if u is not None:
            grid_shape = u.shape[1:]
            row.setdefault("grid", "x".join(str(size) for size in grid_shape))
            row.setdefault("cells", int(np.prod(grid_shape)))
            if rho is not None:
                row.update(field_health(u, rho, solid, self.u_ref))

        backend = _backend()
        row["backend"] = backend
        if backend == "cuda":
            gpu_gb, scope = gpu_used_gb()
            if gpu_gb is not None:
                row["vram_gpu_gb"] = round(gpu_gb, 2)
            row["vram_gpu_scope"] = scope

        run_id = log_run(path=self.path, **row)

        # one-line summary
        mlups = ""
        if row.get("cells") and self.steps and wall_time > 0:
            mlups = f", {row['cells'] * self.steps / wall_time / 1e6:.0f} MLUPS"
        result = f"{row.get('metric', '')} {row.get('value', '')}".strip()
        vram = f"VRAM {row.get('vram_fields_gb', '?')} GB fields"
        if row.get("vram_gpu_gb", "") != "":
            vram += f", {row['vram_gpu_gb']} GB GPU ({row['vram_gpu_scope']})"
        print(f"run {run_id} logged: {result} | {wall_time:.0f} s{mlups} | max|u| {row.get('max_u', '?')}, "
              f"rho {row.get('rho_min', '?')}-{row.get('rho_max', '?')}, hot {row.get('hot_cells', '?')} | {vram}")

        return run_id
