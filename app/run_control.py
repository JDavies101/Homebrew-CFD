# run control: launch the solver as a separate process, follow its run folder, stop it cleanly
import json
import sys
import numpy as np
from pathlib import Path
from PySide6.QtCore import QObject, QProcess, QTimer, Signal

project_root = Path(__file__).resolve().parent.parent
poll_milliseconds = 500
run_folder_prefix = "Run folder: "

class RunController(QObject):
    """
    One solver process at a time: start it, relay its output and progress, stop it, report the result.
    """

    output = Signal(str) # one solver output line
    progress = Signal(dict) # progress.json contents
    finished = Signal(dict) # result.json contents plus exit_code ({} + exit_code if the solver died first)
    samples = Signal(object) # (column names, values array (samples, columns)) whenever new samples arrive

    def __init__(self, parent=None):
        """
        Idle controller with a progress poll timer.
        """

        super().__init__(parent)
        self.process = None
        self.run_folder = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.sample_count = 0

    def is_running(self):
        """
        Returns True while a solver process is alive.
        """

        return self.process is not None and self.process.state() != QProcess.NotRunning

    def start(self, case_path, runs_directory):
        """
        Launch python -u -m src.run on a saved case file; -u so output lines arrive as they are printed.
        """

        self.run_folder = None
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(project_root))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.on_finished)
        self.process.start(sys.executable, ["-u", "-m", "src.run", str(case_path), "--runs", str(runs_directory)])
        self.timer.start(poll_milliseconds)
        self.sample_count = 0

    def read_output(self):
        """
        Relay output lines; learn the run folder from the solver's first line; drop progress-bar redraws.
        """

        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            if line.startswith(run_folder_prefix):
                self.run_folder = (project_root / line[len(run_folder_prefix):].strip()).resolve()
            if line.strip() and not line.startswith(("[#", "[-")):
                self.output.emit(line)

    def poll(self):
        """
        Read progress.json; skip quietly if it is not there yet or is being replaced.
        """

        if self.run_folder is None:
            return
        try:
            data = json.loads((self.run_folder / "progress.json").read_text())
        except (OSError, ValueError):
            return
        self.progress.emit(data)

        # live coefficient samples: complete rows only (the solver may be mid-append)
        try:
            lines = (self.run_folder / "coefficients_live.csv").read_text().splitlines()
        except OSError:
            return
        if len(lines) < 2:
            return
        header = lines[0].split(",")
        rows = [line.split(",") for line in lines[1:] if line.count(",") == len(header) - 1]
        if len(rows) > self.sample_count:
            self.sample_count = len(rows)
            self.samples.emit((header, np.array(rows, dtype=float)))

    def stop(self):
        """
        Ask the solver to stop at its next health check (STOP file); kill only if it has no run folder yet.
        """

        if not self.is_running():
            return
        if self.run_folder is not None:
            (self.run_folder / "STOP").touch()
        else:
            self.process.kill()

    def stop_and_wait(self, timeout_milliseconds=15000):
        """
        Stop cleanly and wait; kill if the solver does not exit in time (used when the app closes).
        """

        self.stop()
        if self.process is not None and not self.process.waitForFinished(timeout_milliseconds):
            self.process.kill()
            self.process.waitForFinished(3000)

    def on_finished(self, exit_code, exit_status):
        """
        Final poll, then report result.json (or just the exit code if the solver failed before writing it).
        """

        self.timer.stop()
        self.poll()
        result = {}
        if self.run_folder is not None:
            try:
                result = json.loads((self.run_folder / "result.json").read_text())
            except (OSError, ValueError):
                result = {}
        result["exit_code"] = exit_code
        result["run_folder"] = str(self.run_folder) if self.run_folder is not None else ""
        self.finished.emit(result)