# main window: setup tree, property forms, case summary with live validation, file handling and autosave
from pathlib import Path
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget, QToolBar, QDockWidget, QPlainTextEdit, QProgressBar)
from src import __version__
from src.run.case import Case, setting_choices, check_choice
from src.run.case_file import load_case_file, save_case_file, geometry_choices
from app.property_form import PropertyForm
from app.viewport import CaseViewport, named_views, corner_views
from app.run_control import RunController
from app.monitor_plot import MonitorPlot

template_directory = Path(__file__).resolve().parent.parent / "cases" / "templates"
autosave_milliseconds = 60_000
# the UI offers every choice the engine accepts, except a custom start (a case file cannot hold an initial field)
form_choices = {**setting_choices, **geometry_choices, "start": ("rest_ramp", "uniform")}
solver_fields = ("name", "collision", "inlet", "start", "allow_below_floor")

class MainWindow(QMainWindow):
    """
    Setup tree on the left, the selected node's properties in the middle, case summary and validation on the right.
    """

    def __init__(self):
        """
        Build menus, panes and the autosave timer; start empty until a case is opened.
        """

        super().__init__()
        self.case_file = None
        self.path = None
        self.dirty = False

        # panes
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self.show_node)
        self.forms = QStackedWidget()
        self.summary = QLabel("Open a case or start from a template (File menu).")
        self.summary.setAlignment(Qt.AlignTop)
        self.summary.setWordWrap(True)
        summary_pane = QWidget()
        QVBoxLayout(summary_pane).addWidget(self.summary)
        splitter = QSplitter()
        self.viewport = CaseViewport(self)
        self.part_reports = []
        right_pane = QSplitter(Qt.Vertical)
        right_pane.addWidget(self.viewport.plotter)
        right_pane.addWidget(summary_pane)
        right_pane.setSizes([560, 200])
        splitter.addWidget(self.tree)
        splitter.addWidget(self.forms)
        splitter.addWidget(right_pane)
        splitter.setSizes([200, 380, 620])
        self.setCentralWidget(splitter)

        self.monitor = MonitorPlot()
        monitor_dock = QDockWidget("Monitors", self)
        monitor_dock.setWidget(self.monitor)
        self.addDockWidget(Qt.RightDockWidgetArea, monitor_dock)

        # runs: controller, console dock, status-bar progress
        self.runs = RunController(self)
        self.runs.output.connect(self.on_run_output)
        self.runs.progress.connect(self.on_run_progress)
        self.runs.finished.connect(self.on_run_finished)
        self.runs.samples.connect(self.monitor.update_samples)
        self.last_result = None
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        console_dock = QDockWidget("Console", self)
        console_dock.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)
        self.run_progress = QProgressBar()
        self.run_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.run_progress)

        self.build_menus()

        # autosave beside the case file, never over it
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(autosave_milliseconds)
        self.resize(1200, 760)
        self.update_title()

    def build_menus(self):
        """
        File menu: new from each template, open, save, save as, quit.
        """

        file_menu = self.menuBar().addMenu("&File")
        template_menu = file_menu.addMenu("New from template")
        for template_path in sorted(template_directory.glob("*.json")):
            action = template_menu.addAction(template_path.stem)
            action.triggered.connect(lambda checked=False, template_path=template_path: self.open_path(template_path, as_template=True))
        
        for label, shortcut, handler in (("&Open...", QKeySequence.Open, self.open_dialog), ("&Save", QKeySequence.Save, self.save),
                                         ("Save &as...", QKeySequence.SaveAs, self.save_as), ("&Quit", QKeySequence.Quit, self.close)):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            file_menu.addAction(action)

        # view: fit, projection, named views; the common ones also on a toolbar
        view_menu = self.menuBar().addMenu("&View")
        toolbar = QToolBar("View")
        self.addToolBar(toolbar)
        fit_action = QAction("Fit", self)
        fit_action.setShortcut(QKeySequence("Home"))
        fit_action.triggered.connect(self.viewport.fit)
        orthographic_action = QAction("Orthographic", self)
        orthographic_action.setCheckable(True)
        orthographic_action.toggled.connect(self.viewport.set_orthographic)
        for action in (fit_action, orthographic_action):
            view_menu.addAction(action)
            toolbar.addAction(action)
        view_menu.addSeparator()
        toolbar.addSeparator()
        
        for name in named_views:
            action = view_menu.addAction(name)
            action.triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))
            
            if name in ("Side (+z)", "Top (+y)", "Front (-x, from inlet)", "Isometric"):
                toolbar.addAction(action)
        
        corner_menu = view_menu.addMenu("Corners")
        
        for name in corner_views:
            corner_menu.addAction(name).triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))

        # run: start / stop, also on the toolbar
        run_menu = self.menuBar().addMenu("&Run")
        self.run_action = QAction("Run", self)
        self.run_action.setShortcut(QKeySequence("F5"))
        self.run_action.triggered.connect(self.start_run)
        self.stop_action = QAction("Stop", self)
        self.stop_action.setShortcut(QKeySequence("Shift+F5"))
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.runs.stop)
        toolbar.addSeparator()
        for action in (self.run_action, self.stop_action):
            run_menu.addAction(action)
            toolbar.addAction(action)

    def open_dialog(self):
        """
        Ask for a case file and open it.
        """

        path, _ = QFileDialog.getOpenFileName(self, "Open case", "", "Case files (*.json)")

        if path:
            self.open_path(path)

    def open_path(self, path, as_template=False):
        """
        Load a case file into the tree; a template opens untitled so saving never overwrites it.
        """

        if not self.confirm_discard():
            return
        try:
            self.case_file = load_case_file(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.critical(self, "Cannot open case", f"{path}\n\n{error}")
            return
        
        self.path = None if as_template else Path(path)
        self.dirty = as_template
        self.build_tree()
        self.part_reports = self.viewport.draw(self.case_file, reset_camera=True)
        self.refresh_summary()

    def build_tree(self):
        """
        One tree node per settings group, one child per geometry part, each with its own form.
        """

        self.tree.clear()
        while self.forms.count():
            self.forms.removeWidget(self.forms.widget(0))
        
        case_file = self.case_file
        nodes = (("Solver", case_file, solver_fields), ("Flow", case_file.flow, None), ("Domain", case_file.domain, None),
                 ("Models", case_file.turbulence, None), ("Timing", case_file.timing, None))
        for label, target, field_names in nodes:
            self.add_node(self.tree, label, target, field_names)
        
        geometry_node = QTreeWidgetItem(self.tree, ["Geometry"])
        for spec in case_file.geometry:
            self.add_node(geometry_node, spec.name, spec, None)
        
        self.tree.expandAll()
        self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def add_node(self, parent, label, target, field_names):
        """
        Add a tree node with its property form on the stack.
        """

        form = PropertyForm(target, form_choices, field_names)
        form.changed.connect(self.on_changed)
        item = QTreeWidgetItem(parent, [label])
        item.setData(0, Qt.UserRole, self.forms.addWidget(form))

    def show_node(self, item, previous):
        """
        Show the form of the selected node (group nodes without a form keep the current one).
        """

        index = item.data(0, Qt.UserRole) if item is not None else None
        
        if index is not None:
            self.forms.setCurrentIndex(index)

    def on_changed(self):
        """
        Any accepted edit: mark unsaved and re-check the case.
        """

        self.dirty = True
        self.part_reports = self.viewport.draw(self.case_file)
        self.refresh_summary()

    def refresh_summary(self):
        """
        Derived numbers (tau, Mach, cells, steps) and the validation verdict, recomputed after every edit.
        """

        case_file = self.case_file
        case = Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                    timing=case_file.timing, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                    allow_below_floor=case_file.allow_below_floor)
        
        error = self.validation_error()
        verdict = "<span style='color:#2a2'>Ready to run</span>" if error is None else f"<span style='color:#d33'>{error}</span>"

        cells = case_file.domain.nx * case_file.domain.ny * case_file.domain.nz
        mach_number = case_file.flow.free_stream_velocity * 3 ** 0.5
        lines = [f"<b>{case_file.name}</b>", verdict, "",
                 f"tau = {case_file.flow.relaxation_time:.5f}", f"Mach = {mach_number:.3f}",
                 f"cells = {cells:,}", f"steps = {case.total_steps():,} (warmup {case.warmup_steps():,})",
                "", "<b>Parts</b>", *self.part_reports]
        if self.last_result is not None:
            lines += ["", "<b>Last run</b>", *self.result_lines(self.last_result)]

        self.summary.setText("<br>".join(lines))
        self.update_title()

    def validation_error(self):
        """
        The engine's verdict on the current case, without building geometry.

        Returns the error message, or None when the case can run.
        """

        case_file = self.case_file
        case = Case(name=case_file.name, tag=case_file.name, flow=case_file.flow, domain=case_file.domain, turbulence=case_file.turbulence,
                    timing=case_file.timing, collision=case_file.collision, inlet=case_file.inlet, start=case_file.start,
                    allow_below_floor=case_file.allow_below_floor)
        try:
            case.validate()
            for spec in case_file.geometry:
                check_choice("geometry kind", spec.kind, geometry_choices["kind"])
                check_choice("geometry wall", spec.wall, geometry_choices["wall"])
        except (ValueError, ZeroDivisionError) as error:
            return str(error)

        return None

    def save(self):
        """
        Save to the current file, or ask for one.

        Returns True when saved.
        """

        if self.path is None:
            return self.save_as()
        
        save_case_file(self.case_file, self.path)
        self.dirty = False
        self.update_title()

        return True

    def save_as(self):
        """
        Ask for a file name and save there.

        Returns True when saved.
        """

        path, _ = QFileDialog.getSaveFileName(self, "Save case", f"{self.case_file.name}.json", "Case files (*.json)")
        if not path:
            return False
        
        self.path = Path(path)

        return self.save()

    def autosave(self):
        """
        Write unsaved edits to <file>.autosave beside the case (never over it).
        """

        if self.dirty and self.path is not None:
            save_case_file(self.case_file, self.path.with_name(self.path.name + ".autosave"))

    def confirm_discard(self):
        """
        Before replacing or closing a case with unsaved edits: save, discard, or cancel.

        Returns True to continue.
        """

        if self.case_file is None or not self.dirty:
            return True
        
        answer = QMessageBox.question(self, "Unsaved changes", "Save changes to the current case?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self.save()

        return answer == QMessageBox.Discard

    def closeEvent(self, event):
        """
        Ask before closing with a run in progress or unsaved edits; stop the solver before exiting.
        """

        if self.runs.is_running():
            answer = QMessageBox.question(self, "Run in progress", "Stop the running solver and quit?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.runs.stop_and_wait()
        if self.confirm_discard():
            self.viewport.close()
            event.accept()
        else:
            event.ignore()

    def update_title(self):
        """
        Window title: app version, case file name, * when unsaved.
        """

        name = self.path.name if self.path is not None else (self.case_file.name + " (unsaved)" if self.case_file else "")
        self.setWindowTitle(f"Homebrew CFD {__version__} - {name}{' *' if self.dirty else ''}")

    def start_run(self):
        """
        Validate, make sure the case is saved, then launch the solver into runs/ beside the case file.
        """

        if self.case_file is None or self.runs.is_running():
            return
        error = self.validation_error()
        if error is not None:
            QMessageBox.warning(self, "Cannot run", error)
            return
        if (self.dirty or self.path is None) and not self.save():
            return
        self.console.clear()
        self.monitor.clear()
        self.last_result = None
        self.run_progress.setValue(0)
        self.run_progress.setVisible(True)
        self.run_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.runs.start(self.path, self.path.parent / "runs")
        self.statusBar().showMessage("starting solver...")

    def on_run_output(self, line):
        """
        Append one solver line to the console.
        """

        self.console.appendPlainText(line)

    def on_run_progress(self, data):
        """
        Progress bar and status line from progress.json.
        """

        steps = max(1, int(data.get("steps", 0)))
        self.run_progress.setMaximum(steps)
        self.run_progress.setValue(int(data.get("step", 0)))
        message = f"{data.get('status', '')}  step {data.get('step', 0):,} / {steps:,}"
        if "max_velocity" in data:
            message += f"  max|u| {data['max_velocity']:.3g}"
        self.statusBar().showMessage(message)

    def on_run_finished(self, result):
        """
        Back to idle; show the outcome in the status bar and the summary.
        """

        self.run_progress.setVisible(False)
        self.run_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        status = result.get("status", f"solver exited with code {result['exit_code']} (see console)")
        self.statusBar().showMessage(f"run {status}", 10000)
        self.last_result = result
        self.refresh_summary()

    def result_lines(self, result):
        """
        Summary lines for a run result: status, then each part's mean force coefficients.

        Returns a list of strings.
        """

        lines = [f"status: {result.get('status', 'failed')}"]
        for name, statistics in result.get("parts", {}).items():
            lines.append(f"{name}: C_x {statistics['x']['mean']:.4f}  C_y {statistics['y']['mean']:.4f} (SE {statistics['y']['se']:.4f})")
        if result.get("run_folder"):
            lines.append(f"<small>{result['run_folder']}</small>")

        return lines