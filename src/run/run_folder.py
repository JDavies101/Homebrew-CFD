# run folders: one directory per run holding the case, live progress, results and the run record
import datetime
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

def case_hash(case_file):
    """
    SHA-256 of the case settings in canonical form (sorted keys), so results can be matched to the exact case.

    Returns the hex digest.
    """

    canonical = json.dumps(asdict(case_file), sort_keys=True)

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def create_run_folder(runs_directory, name):
    """
    Make runs_directory/<date-time>_<name>; never reuses an existing folder.

    Returns the folder Path.
    """

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = Path(runs_directory) / f"{stamp}_{name}"
    folder.mkdir(parents=True, exist_ok=False)

    return folder

def write_json_atomic(path, data, attempts=5):
    """
    Write JSON to a temp file, then rename over the target, so a reader never sees half a file.
    Retries the rename briefly: on Windows it fails while another process has the target open.
    """

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)