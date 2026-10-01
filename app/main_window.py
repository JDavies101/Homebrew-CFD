# main window: setup tree, property forms, case summary with live validation, file handling and autosave
from pathlib import Path
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget, QDockWidget, QPlainTextEdit, QProgressBar, QMenu, QStyle)
from src import __version__
from src.run.case import Case, setting_choices, check_choice
from src.run.case_file import load_case_file, save_case_file, geometry_choices
from app.property_form import PropertyForm
from app.viewport import CaseViewport, named_views, corner_views
from app.run_control import RunController
from app.monitor_plot import MonitorPlot
from app.ribbon import Ribbon

template_directory = Path(__file__).resolve().parent.parent / "cases" / "templates"
autosave_milliseconds = 60_000
# the UI offers every choice the engine accepts, except a custom start (a case file cannot hold an initial field)
form_choices = {**setting_choices, **geometry_choices, "start": ("rest_ramp", "uniform")}
solver_fields = ("name", "collision", "inlet", "start", "allow_below_floor")

# splitter handles: wide enough to see and grab, highlighted on hover (icon violet)
splitter_style = ("QSplitter::handle { background: #2a3444; } QSplitter::handle:hover { background: #2d4dc8; } "
                  "QSplitter::handle:horizontal { width: 6px; } QSplitter::handle:vertical { height: 6px; }")

def sentence_case(text):
    """
    Capitalize the first letter only (keeps symbols like tau and C_y as written).

    Returns the string.
    """

    return text[:1].upper() + text[1:]

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

        #results page: monitors above the last-run summary, the split is adjustable
        self.monitor = MonitorPlot()
        self.results_summary = QLabel("No run yet.")
        self.results_summary.setAlignment(Qt.AlignTop)
        self.results_summary.setWordWrap(True)
        results_page = QSplitter(Qt.Vertical)
        results_page.addWidget(self.monitor)
        results_page.addWidget(self.results_summary)
        results_page.setSizes([600, 160])

        # pages: setup (tree, forms, viewport, and summary) and results, switched by the ribbon
        self.pages = QStackedWidget()
        self.pages.addWidget(splitter)
        self.pages.addWidget(results_page)
        self.setCentralWidget(self.pages)
        self.setStyleSheet(splitter_style)

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

        self.build_ribbon()

        # autosave beside the case file, never over it
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start(autosave_milliseconds)
        self.resize(1200, 760)
        self.update_title()

    def make_action(self, text, icon, shortcut, handler):
        """
        A window-level action: its shortcut works whichever ribbon tab is showing.

        Returns the action.
        """

        action = QAction(text, self)
        if icon is not None:
            action.setIcon(icon)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if handler is not None:
            action.triggered.connect(handler)
        self.addAction(action)

        return action

    def build_ribbon(self):
        """
        Ribbon tabs Home (case files), View (camera) and Run (solver).
        """

        icon = self.style().standardIcon
        self.ribbon = Ribbon()
        self.setMenuWidget(self.ribbon)

        # home: new from a template, open, save
        template_menu = QMenu(self)
        
        for template_path in sorted(template_directory.glob("*.json")):
            action = template_menu.addAction(template_path.stem)
            action.triggered.connect(lambda checked=False, template_path=template_path: self.open_path(template_path, as_template=True))
        
        new_action = self.make_action("New", icon(QStyle.SP_FileIcon), None, None)
        new_action.setMenu(template_menu)
        open_action = self.make_action("Open", icon(QStyle.SP_DialogOpenButton), QKeySequence.Open, self.open_dialog)
        save_action = self.make_action("Save", icon(QStyle.SP_DialogSaveButton), QKeySequence.Save, self.save)
        save_as_action = self.make_action("Save as", icon(QStyle.SP_DialogSaveAllButton), QKeySequence.SaveAs, self.save_as)
        self.make_action("Quit", None, QKeySequence.Quit, self.close)
        self.case_loading_actions = [new_action, open_action]
        self.ribbon.add_tab("Home").add_group("Case", (new_action, open_action, save_action, save_as_action))

        # view: fit, projection, the common views as buttons, every view and corner in a menu
        fit_action = self.make_action("Fit", icon(QStyle.SP_BrowserReload), QKeySequence("Home"), self.viewport.fit)
        orthographic_action = self.make_action("Orthographic", icon(QStyle.SP_FileDialogContentsView), None, None)
        orthographic_action.setCheckable(True)
        orthographic_action.toggled.connect(self.viewport.set_orthographic)
        view_menu = QMenu(self)
        common_views = []
        
        for name in named_views:
            action = view_menu.addAction(name)
            action.triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))
            if name in ("Side (+z)", "Top (+y)", "Front (-x, from inlet)", "Isometric"):
                common_views.append(action)
        
        corner_menu = view_menu.addMenu("Corners")
        
        for name in corner_views:
            corner_menu.addAction(name).triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))
        
        all_views_action = self.make_action("All views", icon(QStyle.SP_DesktopIcon), None, None)
        all_views_action.setMenu(view_menu)
        view_tab = self.ribbon.add_tab("View")
        view_tab.add_group("Camera", (fit_action, orthographic_action))
        view_tab.add_group("Views", (*common_views, all_views_action))

        # run: start / stop
        self.run_action = self.make_action("Run", icon(QStyle.SP_MediaPlay), QKeySequence("F5"), self.start_run)
        self.stop_action = self.make_action("Stop", icon(QStyle.SP_MediaStop), QKeySequence("Shift+F5"), self.runs.stop)
        self.stop_action.setEnabled(False)
        self.ribbon.add_tab("Run").add_group("Solver", (self.run_action, self.stop_action))

        # results: the monitors page with run control at hand
        self.results_tab = self.ribbon.add_tab("Results")
        self.results_tab.add_group("Solver", (self.run_action, self.stop_action))
        self.ribbon.currentChanged.connect(self.show_page)

    def show_page(self, index):
        """
        The Results ribbon tab shows the results page; every other tab shows the setup page.
        """

        self.pages.setCurrentIndex(1 if self.ribbon.widget(index) is self.results_tab else 0)

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
        if self.runs.is_running():
            verdict = "<span style='color:#27c'>Running</span>"

        elif error is None:
            verdict = "<span style='color:#2a2'>Ready to run</span>"

        else:
            verdict = f"<span style='color:#d33'>{error}</span>"

        cells = case_file.domain.nx * case_file.domain.ny * case_file.domain.nz
        mach_number = case_file.flow.free_stream_velocity * 3 ** 0.5
        lines = [f"<b>{case_file.name}</b>", verdict, "",
                 f"tau = {case_file.flow.relaxation_time:.5f}", f"Mach = {mach_number:.3f}",
                 f"cells = {cells:,}", f"steps = {case.total_steps():,} (warmup {case.warmup_steps():,})",
                "", "<b>Parts</b>", *self.part_reports]
        if self.last_result is not None:
            lines += ["", "<b>Last run</b>", *self.result_lines(self.last_result)]

        if self.runs.is_running():
            self.results_summary.setText("Running")

        elif self.last_result is not None:
            self.results_summary.setText("<br>".join(["<b>Last run</b>", 
                                                      *self.result_lines(self.last_result)]))
        
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
            return sentence_case(str(error))

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

    def set_running(self, running):
        """
        Lock or unlock the window for a live run: inputs and case loading frozen, tree still navigable.
        """

        self.forms.setEnabled(not running)
        for action in self.case_loading_actions:
            action.setEnabled(not running)

        self.run_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        self.run_progress.setVisible(running)
    
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
        self.set_running(True)
        self.ribbon.setCurrentWidget(self.results_tab)
        self.runs.start(self.path, self.path.parent / "runs", self.case_file.name)
        self.statusBar().showMessage("Starting solver...")
        self.refresh_summary()

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
        message = f"{sentence_case(data.get('status', ''))}  step {data.get('step', 0):,} / {steps:,}"
        if "max_velocity" in data:
            message += f"  max |u| {data['max_velocity']:.3g}"
        
        self.statusBar().showMessage(message)

        if "flow_through_steps" in data:
            self.monitor.set_timeline(data["steps"], data["warmup"], 
                                      data["flow_through_steps"], data["sample_every"])

    def on_run_finished(self, result):
        """
        Back to idle; show the outcome in the status bar and the summary.
        """

        self.set_running(False)
        status = result.get("status", f"solver exited with code {result['exit_code']} (see console)")        
        self.statusBar().showMessage(f"Run {status}", 10000)
        self.last_result = result
        self.refresh_summary()

    def result_lines(self, result):
        """
        Summary lines for a run result: status, then each part's mean force coefficients.

        Returns a list of strings.
        """

        lines = [f"Status: {result.get('status', 'failed')}"]
        for name, statistics in result.get("parts", {}).items():
            lines.append(f"{name}: C_x {statistics['x']['mean']:.4f}  C_y {statistics['y']['mean']:.4f} (SE {statistics['y']['se']:.4f})")
        if result.get("run_folder"):
            lines.append(f"<small>{result['run_folder']}</small>")

        return lines