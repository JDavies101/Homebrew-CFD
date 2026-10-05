# run queue: case snapshots waiting to run one at a time, kept in queue.json across sessions
import json
import uuid
from pathlib import Path
from src.run.case_file import save_case_file
from src.run.run_folder import write_json_atomic

finished_states =("finished", "stopped", "blow_up", "failed", "interrupted")

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
    
    def next_pending(self):
        """
        Returns the first pending entry, or None when nothing is waiting.
        """

        return next((entry for entry in self.entries if entry["state"] == "pending"), None)
    
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
        Move an entry up (offset -1) or down (+1) in the order, clamped to the ends.
        """

        index = self.entries.index(self.find(identifier))
        target = min(max(index + offset, 0), len(self.entries) - 1)
        self.entries.insert(target, self.entries.pop(index))
        self.save()

    def clear_finished(self):
        """
        Drop every finished entry and its snapshot (the run folders stay).
        """

        for entry in [entry for entry in self.entries if entry["state"] in finished_states]:
            Path(entry["case"]).unlink(missing_ok=True)
            self.entries.remove(entry)

        self.save()