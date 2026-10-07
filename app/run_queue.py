# run queue: case snapshots waiting to run one at a time, kept in queue.json across sessions
import json
import uuid
from pathlib import Path
from src.run.case_file import save_case_file
from src.run.run_folder import write_json_atomic

class RunQueue:
    """
    Ordered queue entries, each a case snapshot with its runs folder, state, run folder and headline result.
    """

    def __init__(self, folder):
        """
        Load queue.json from the folder if present; an entry still running when the app closed becomes interrupted.
        """

        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "queue.json"
        self.entries = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())

        for entry in self.entries:
            if entry["state"] == "running":
                entry["state"] = "interrupted"
        
    def save(self):
        """
        Write queue.json atomically.
        """

        write_json_atomic(self.path, self.entries)

    def add(self, case_file, runs_directory, study="", value=None):
        """
        Snapshot the case as it is now, so later edits to the open case do not change the queued run.
        A study folder path and swept value tag the entry as one run of a parametric study.

        Returns the new entry.
        """

        identifier = uuid.uuid4().hex[:8]
        snapshot = self.folder / f"{identifier}_{case_file.name}.json"
        save_case_file(case_file, snapshot)
        entry = {"id": identifier, "name": case_file.name, "case": str(snapshot), 
                 "runs": str(runs_directory), "state": "pending", "run_folder": "", "headline": "",
                 "study": study, "value": value}
        self.entries.append(entry)
        self.save()

        return entry

    def study_entries(self, study):
        """
        Returns the entries of one study (identified by its folder path), in queue order.
        """

        folder = str(Path(study).resolve())

        return [entry for entry in self.entries if entry.get("study") == folder]

    def find(self, identifier):
        """
        Returns the entry with this id.
        """

        return next(entry for entry in self.entries if entry["id"] == identifier)
    
    def project_entries(self, runs_directory):
        """
        Returns the entries that run into this runs folder (one project's), in queue order.
        """

        folder = Path(runs_directory).resolve()

        return [entry for entry in self.entries if Path(entry["runs"]).resolve() == folder]

    def next_pending(self, runs_directory=None):
        """
        Returns the first pending entry (of one runs folder when given), or None when nothing is waiting.
        """

        candidates = self.entries if runs_directory is None else self.project_entries(runs_directory)

        return next((entry for entry in candidates if entry["state"] == "pending"), None)
    
    def mark(self, identifier, state, run_folder="", headline=""):
        """
        Set an entry's state, plus its run folder and headline result once it has them.
        """

        entry = self.find(identifier)
        entry["state"] = state
        if run_folder:
            entry["run_folder"] = run_folder

        if headline:
            entry["headline"] = headline

        self.save()

    def remove(self, identifier):
        """
        Drop a pending entry and its snapshot; a running entry is never removed.
        """

        entry = self.find(identifier)
        if entry["state"] == "running":
            return
        
        Path(entry["case"]).unlink(missing_ok=True)
        self.entries.remove(entry)
        self.save()

    def move(self, identifier, offset):
        """
        Move a pending entry up (offset -1) or down (+1) past the next pending entry; running and finished entries
        keep their place, and a move past either end does nothing.
        """

        entry = self.find(identifier)
        pending = [candidate for candidate in self.entries if candidate["state"] == "pending"]
        if entry not in pending:
            return

        target = pending.index(entry) + offset
        if target < 0 or target >= len(pending):
            return

        first = self.entries.index(entry)
        second = self.entries.index(pending[target])
        self.entries[first], self.entries[second] = self.entries[second], self.entries[first]
        self.save()

    def clear_pending(self, runs_directory):
        """
        Drop every pending entry of one runs folder with its snapshot; the running entry and finished ones (whose
        results the studies read) stay.
        """

        for entry in self.project_entries(runs_directory):
            if entry["state"] != "pending":
                continue

            Path(entry["case"]).unlink(missing_ok=True)
            self.entries.remove(entry)

        self.save()

    def active_entries(self, runs_directory):
        """
        Returns the pending and running entries of one runs folder, in queue order (what the queue table shows).
        """

        return [entry for entry in self.project_entries(runs_directory) if entry["state"] in ("pending", "running")]