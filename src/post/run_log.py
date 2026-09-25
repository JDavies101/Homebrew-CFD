# run log: one row per solver run, appended to docs/run_log.csv (versioned with the code;
# results/ is gitignored). Two calls per example:
#
#     run = RunRecord("cylinder", sim, steps=steps, u_ref=U, nu=nu, tau=tau, Re=Re, collision="BGK")
#     ... time loop ...
#     run.stop()                      # optional: exclude post-processing from the timing
#     run.finish(metric="Cd", value=cd, reference=1.4)
#
# RunRecord starts the wall-clock timer. finish() collects everything it can from the sim itself
# (grid, cells, MLUPS, field health, wall-model engagement, nut_les, VRAM, backend, git commit,
# command) and appends one row. Rows start as status "unreviewed"; the verdict (good / bad /
# invalid / superseded / reference) and its reason are set during review.
import csv
import datetime
import os
import subprocess
import sys
import time

import numpy as np

LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "run_log.csv"))

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


# ---------------------------------------------------------------- environment

def _git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _command():
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    name = spec.name if spec is not None else os.path.basename(sys.argv[0])
    return " ".join(["python -m", name] + sys.argv[1:])


def _backend():
    try:
        from src.engine import runtime
        return runtime._arch or ""
    except Exception:
        return ""


def _dtype_bytes(dtype):
    s = str(dtype)
    if "64" in s:
        return 8
    if "16" in s:
        return 2
    if "8" in s:
        return 1
    return 4


def fields_gb(sim):
    # bytes held by the sim's Taichi fields: what the solver itself allocated
    total = 0
    for v in vars(sim).values():
        if "Field" in type(v).__name__ and hasattr(v, "shape") and hasattr(v, "dtype"):
            total += int(np.prod(v.shape)) * _dtype_bytes(v.dtype)
    return total / 1e9


def gpu_used_gb():
    # (GB, scope): this process if the driver reports it (Linux / TCC), else the whole device
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            pid, mem = [p.strip() for p in line.split(",")[:2]]
            if pid == str(os.getpid()) and mem.replace(".", "").isdigit():
                return float(mem) / 1024.0, "process"
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024.0, "device"
    except Exception:
        return None, "n/a"


# ---------------------------------------------------------------- field health

def field_health(u, rho, solid=None, u_ref=None, band=4):
    # speed / density summary over fluid cells, plus where the "hot" cells (|u| > 2 u_ref) sit:
    # counts in a `band`-cell layer on each domain face (x-, x+, y-, y+, z-, z+) and the interior
    speed = np.sqrt((u.astype(np.float64) ** 2).sum(axis=0))
    fluid = np.ones(speed.shape, bool) if solid is None else (solid == 0)
    finite = np.isfinite(speed) & np.isfinite(rho)
    out = {}
    bad = int((fluid & ~finite).sum())
    ok = fluid & finite
    if ok.any():
        out["max_u"] = round(float(speed[ok].max()), 4)
        out["rho_min"] = round(float(rho[ok].min()), 4)
        out["rho_max"] = round(float(rho[ok].max()), 4)
    else:
        out["max_u"] = "nan"
    if u_ref:
        hot = ok & (speed > 2.0 * u_ref)
        out["hot_cells"] = int(hot.sum())
        names = ["x", "y", "z"]
        parts = []
        near_any = np.zeros(speed.shape, bool)
        for a in range(speed.ndim):
            idx = np.arange(speed.shape[a])
            for side, sel in (("-", idx < band), ("+", idx >= speed.shape[a] - band)):
                shape = [1] * speed.ndim
                shape[a] = speed.shape[a]
                m = np.broadcast_to(sel.reshape(shape), speed.shape)
                near_any |= m
                parts.append(f"{names[a]}{side}:{int((hot & m).sum())}")
        parts.append(f"interior:{int((hot & ~near_any).sum())}")
        if bad:
            parts.append(f"nonfinite:{bad}")
        out["hot_where"] = " ".join(parts)
    elif bad:
        out["hot_where"] = f"nonfinite:{bad}"
    return out


# ---------------------------------------------------------------- csv

def _migrate(path):
    # rewrite an older log under the current header (new columns empty)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    if header == FIELDS:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def _next_id(path):
    if not os.path.exists(path):
        return 1
    with open(path, newline="", encoding="utf-8") as fh:
        ids = [int(r["run_id"]) for r in csv.DictReader(fh) if str(r.get("run_id", "")).isdigit()]
    return max(ids, default=0) + 1


def log_run(path=None, **fields):
    # low-level append; RunRecord.finish() is the normal entry point.
    # cells, steps and wall_time_s (when all given) also fill in mlups
    path = path or LOG_PATH
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown run-log fields: {sorted(unknown)}")
    row = {k: "" for k in FIELDS}
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
    c = fields.get("cells")
    n = fields.get("steps")
    t = fields.get("wall_time_s")
    if c and n and t:
        row["mlups"] = round(c * n / t / 1e6, 1)
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row["run_id"]


# ---------------------------------------------------------------- the one call examples use

class RunRecord:
    def __init__(self, case, sim=None, steps=None, u_ref=None, nu=None, path=None, **fields):
        # construct right before the time loop: the timer starts here, setup is excluded
        self.case = case
        self.sim = sim
        self.steps = steps
        self.u_ref = u_ref
        self.nu = nu
        self.path = path
        self.fields = dict(fields)
        self.t0 = time.time()
        self.t1 = None

    def stop(self):
        # optional: call right after the time loop so post-processing is not timed
        self.t1 = time.time()

    def finish(self, u=None, rho=None, solid=None, **fields):
        # u / rho / solid: pass numpy arrays for runs without a Taichi sim (e.g. the NumPy solver)
        wall_time = (self.t1 or time.time()) - self.t0
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
                nl = sim.nut_les.to_numpy()
                fl = sim.solid.to_numpy() == 0
                if nl[fl].max() > 0:
                    row.setdefault("nut_les_nu_mean", round(float(nl[fl].mean() / self.nu), 2))
                    row.setdefault("nut_les_nu_max", round(float(nl[fl].max() / self.nu), 2))
            row["vram_fields_gb"] = round(fields_gb(sim), 3)
        if u is not None:
            shape = u.shape[1:]
            row.setdefault("grid", "x".join(str(s) for s in shape))
            row.setdefault("cells", int(np.prod(shape)))
            if rho is not None:
                row.update(field_health(u, rho, solid, self.u_ref))
        backend = _backend()
        row["backend"] = backend
        if backend == "cuda":
            gb, scope = gpu_used_gb()
            if gb is not None:
                row["vram_gpu_gb"] = round(gb, 2)
            row["vram_gpu_scope"] = scope
        run_id = log_run(path=self.path, **row)
        mlups = ""
        if row.get("cells") and self.steps and wall_time > 0:
            mlups = f", {row['cells'] * self.steps / wall_time / 1e6:.0f} MLUPS"
        res = f"{row.get('metric', '')} {row.get('value', '')}".strip()
        vram = f"VRAM {row.get('vram_fields_gb', '?')} GB fields"
        if row.get("vram_gpu_gb", "") != "":
            vram += f", {row['vram_gpu_gb']} GB GPU ({row['vram_gpu_scope']})"
        print(f"run {run_id} logged: {res} | {wall_time:.0f} s{mlups} | max|u| {row.get('max_u', '?')}, "
              f"rho {row.get('rho_min', '?')}-{row.get('rho_max', '?')}, hot {row.get('hot_cells', '?')} | {vram}")
        return run_id
