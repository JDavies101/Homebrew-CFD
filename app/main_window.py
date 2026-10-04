# main window: setup tree, property forms, case summary with live validation, file handling and autosave
from pathlib import Path
from PySide6.QtCore import QTimer, Qt, QSettings, QStandardPaths
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget, QDockWidget, QPlainTextEdit, QProgressBar, QMenu)
from src import __version__
from src.run.case import setting_choices, check_choice
from src.run.case_file import load_case_file, save_case_file, geometry_choices, stl_spec, part_free_case
from app.property_form import PropertyForm
from app.viewport import CaseViewport, named_views, corner_views, read_part_mesh, estimate_lines
from app.run_control import RunController
from app.monitor_plot import MonitorPlot
from app.ribbon import Ribbon
from app import theme
from app.new_study import StudyPage
from src.run.estimate import default_throughput
from src.post.run_log import gpu_total_gb

template_directory = Path(__file__).resolve().parent.parent / "cases" / "templates"
autosave_milliseconds = 60_000
recent_cases_limit = 8
# the UI offers every choice the engine accepts, except a custom start (a case file cannot hold an initial field)
form_choices = {**setting_choices, **geometry_choices, "start": ("rest_ramp", "uniform")}
solver_fields = ("name", "collision", "inlet", "start", "allow_below_floor")

def workspace_directory():
    """
    Documents\\Homebrew CFD Projects, with its Runs folder, created if missing: the home for cases and runs
    (never the install folder, which an uninstall or upgrade can delete, and never the source folder).

    Returns the Path.
    """

    documents = Path(QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation))
    workspace = documents / "Homebrew CFD Projects"
    (workspace / "Runs").mkdir(parents=True, exist_ok=True)

    return workspace

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
        self.summary = QLabel("No case open: New study (Ctrl+N), New from a template, or Open.")
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

        self.device_total_gb = gpu_total_gb()

        # pages: setup (tree, forms, viewport, and summary), results, and the start page, switched by the ribbon
        self.pages = QStackedWidget()
        self.pages.addWidget(splitter)
        self.pages.addWidget(results_page)
        self.study_page = StudyPage(self.viewport.stl_cache, self.dialog_directory, self.recent_paths, self.estimate_inputs)
        self.study_page.created.connect(self.on_study_created)
        self.study_page.open_requested.connect(self.on_study_open_requested)
        self.study_page.blank_requested.connect(self.on_study_blank_requested)
        self.pages.addWidget(self.study_page)
        self.setCentralWidget(self.pages)
        self.splitters = {"setup": splitter, "setup_right": right_pane, "results": results_page}

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
        self.console_dock = console_dock
        console_dock.setObjectName("console")
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

        # layout: remember the as built default (for reset layout) then restore the user's last layout
        self.default_layout = self.layout_state()
        self.restore_layout(self.saved_layout())

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

        self.ribbon = Ribbon()
        self.setMenuWidget(self.ribbon)

        # home: new from a template, open, save
        template_menu = QMenu(self)
        
        for template_path in sorted(template_directory.glob("*.json")):
            action = template_menu.addAction(template_path.stem)
            action.triggered.connect(lambda checked=False, template_path=template_path: self.open_path(template_path, as_template=True))
        
        new_study_action = self.make_action("New study", theme.icon("new_study"), QKeySequence.New, self.new_study)
        new_action = self.make_action("New", theme.icon("new"), None, None)
        new_action.setMenu(template_menu)
        open_action = self.make_action("Open", theme.icon("open"), QKeySequence.Open, self.open_dialog)
        save_action = self.make_action("Save", theme.icon("save"), QKeySequence.Save, self.save)
        save_as_action = self.make_action("Save as", theme.icon("save_as"), QKeySequence.SaveAs, self.save_as)
        self.make_action("Quit", None, QKeySequence.Quit, self.close)
        self.case_loading_actions = [new_study_action, new_action, open_action]
        home_tab = self.ribbon.add_tab("Home")
        home_tab.add_group("Case", (new_study_action, new_action, open_action, save_action, save_as_action))

        # home: geometry parts
        import_action = self.make_action("Import STL", theme.icon("import_stl"), None, self.import_stl)
        remove_part_action = self.make_action("Remove part", theme.icon("remove_part"), None, self.remove_part)
        self.case_loading_actions += [import_action, remove_part_action]
        home_tab.add_group("Geometry", (import_action, remove_part_action))

        # view: fit, projection, the common views as buttons, every view and corner in a menu
        fit_action = self.make_action("Fit", theme.icon("fit"), QKeySequence("Home"), self.viewport.fit)
        orthographic_action = self.make_action("Orthographic", theme.icon("orthographic"), None, None)
        orthographic_action.setCheckable(True)
        orthographic_action.toggled.connect(self.viewport.set_orthographic)
        view_menu = QMenu(self)
        view_icons = {"Side (+z)": "view_side", "Top (+y)": "view_top", "Front (-x, from inlet)": "view_front",
                      "Isometric": "view_isometric"}
        common_views = []

        for name in named_views:
            action = view_menu.addAction(name)
            action.triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))
            if name in view_icons:
                action.setIcon(theme.icon(view_icons[name]))
                common_views.append(action)
        
        corner_menu = view_menu.addMenu("Corners")
        
        for name in corner_views:
            corner_menu.addAction(name).triggered.connect(lambda checked=False, name=name: self.viewport.set_view(name))
        
        all_views_action = self.make_action("All views", theme.icon("all_views"), None, None)
        all_views_action.setMenu(view_menu)
        view_tab = self.ribbon.add_tab("View")
        view_tab.add_group("Camera", (fit_action, orthographic_action))
        view_tab.add_group("Views", (*common_views, all_views_action))
        reset_layout_action = self.make_action("Reset layout", theme.icon("reset_layout"), 
                                               None, self.reset_layout)
        console_action = self.console_dock.toggleViewAction()
        console_action.setIcon(theme.icon("console"))
        view_tab.add_group("Window", (console_action, reset_layout_action))

        # run: start / stop
        self.run_action = self.make_action("Run", theme.icon("run"), QKeySequence("F5"), self.start_run)
        self.stop_action = self.make_action("Stop", theme.icon("stop"), QKeySequence("Shift+F5"), self.runs.stop)
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

    def show_start(self, visible):
        """
        Show the start page with the ribbon hidden, or return to the normal setup view.
        """

        self.ribbon.setVisible(not visible)
        self.pages.setCurrentIndex(2 if visible else 0)

    def dialog_directory(self):
        """
        Where file dialogs start: the current case's folder, else the workspace.

        Returns the Path.
        """

        if self.path is not None:
            return self.path.parent
        
        return workspace_directory()

    def open_dialog(self):
        """
        Ask for a case file and open it.
        """

        path, _ = QFileDialog.getOpenFileName(self, "Open case", str(self.dialog_directory()), "Case files (*.json)")
        if path:
            self.open_path(path)

    def open_path(self, path, as_template=False):
        """
        Load a case file into the tree; a template opens untitled so saving never overwrites it.
        """

        if not self.confirm_discard():
            return
        try:
            case_file = load_case_file(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            QMessageBox.critical(self, "Cannot open case", f"{path}\n\n{error}")
            return

        self.show_case(case_file, None if as_template else Path(path))

    def show_case(self, case_file, path):
        """
        Make a case current: tree, viewport and summary; a case without a path is untitled and unsaved.
        """

        self.case_file = case_file
        self.path = path
        self.dirty = path is None
        self.build_tree()
        self.part_reports = self.viewport.draw(case_file, reset_camera=True)
        self.refresh_summary()
        self.show_start(False)
        if path is not None:
            self.remember_recent(path)

    def new_study(self):
        """
        Show the start page, fresh and with the current recent-cases list.
        """

        if not self.confirm_discard():
            return

        self.study_page.reset()
        self.study_page.set_recent(self.recent_paths())
        self.show_start(True)

    def on_study_created(self, case_file):
        """
        The start page finished a case: open it untitled for review.
        """

        self.show_case(case_file, None)

    def on_study_open_requested(self, path):
        """
        The start page asked for an existing case: the given path, or the open dialog when path is empty.
        """

        if path:
            self.open_path(path)
        else:
            self.open_dialog()

    def on_study_blank_requested(self):
        """
        The start page asked for a blank case: the empty 3D template, untitled.
        """

        self.open_path(template_directory / "empty_3d.json", as_template=True)

    def recent_paths(self):
        """
        Recently opened case files, newest first, that still exist on disk.

        Returns a list of path strings.
        """

        settings = QSettings("Homebrew CFD", "Homebrew CFD")
        stored = settings.value("recent_cases")
        if stored is None:
            paths = []
        elif isinstance(stored, str):
            paths = [stored]
        else:
            paths = list(stored)

        return [path for path in paths if Path(path).exists()]

    def remember_recent(self, path):
        """
        Put a case file at the front of the recent-cases list, capped at recent_cases_limit entries.
        """

        path = str(path)
        paths = [entry for entry in self.recent_paths() if entry != path]
        paths.insert(0, path)
        settings = QSettings("Homebrew CFD", "Homebrew CFD")
        settings.setValue("recent_cases", paths[:recent_cases_limit])

    def estimate_inputs(self):
        """
        What the estimate needs from this machine.

        Returns (throughput in MLUPS, measured, total GPU memory in GB or None).
        """

        stored = QSettings("Homebrew CFD", "Homebrew CFD").value("throughput_cuda")
        if stored is None:
            return default_throughput["cuda"], False, self.device_total_gb

        return float(stored), True, self.device_total_gb

    def remember_throughput(self, result):
        """
        Store this machine's throughput from a finished run long enough to swamp the kernel compile.
        """

        steps = result.get("steps_completed", 0)
        if result.get("status") != "finished" or steps < 1000 or not result.get("loop_seconds"):
            return

        throughput = result["cells"] * steps / result["loop_seconds"] / 1e6
        QSettings("Homebrew CFD", "Homebrew CFD").setValue("throughput_cuda", round(throughput, 1))

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

    def import_stl(self):
        """
        Ask for an STL, inspect it, and add it as a new part placed to fit the domain.
        """

        # no case yet: start an untitled one from the empty 3D template
        if self.case_file is None:
            self.open_path(template_directory / "empty_3d.json", as_template=True)


        path, _ = QFileDialog.getOpenFileName(self, "Import STL", str(self.dialog_directory()), "STL files (*.stl)")
        if not path:
            return
        try:
            _, inspection = read_part_mesh(path, self.viewport.stl_cache)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot import STL", f"{path}\n\n{error}")
            return

        # unique part name from the file name
        names = {spec.name for spec in self.case_file.geometry}
        base_name = Path(path).stem
        name = base_name
        number = 2
        while name in names:
            name = f"{base_name}_{number}"
            number += 1

        self.case_file.geometry.append(stl_spec(path, inspection, self.case_file.domain, name))
        self.build_tree()
        geometry_node = self.tree.topLevelItem(self.tree.topLevelItemCount() - 1)
        self.tree.setCurrentItem(geometry_node.child(geometry_node.childCount() - 1))
        self.on_changed()

    def remove_part(self):
        """
        Remove the selected geometry part; does nothing when the selection is not a part.
        """

        item = self.tree.currentItem()
        parent = item.parent() if item is not None else None
        if parent is None or parent.text(0) != "Geometry":
            return

        del self.case_file.geometry[parent.indexOfChild(item)]
        self.build_tree()
        self.on_changed()

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
        case = part_free_case(case_file)
        
        error = self.validation_error()
        if self.runs.is_running():
            verdict = f"<span style='color:{theme.accent}'>Running</span>"

        elif error is None:
            verdict = f"<span style='color:{theme.ok}'>Ready to run</span>"

        else:
            verdict = f"<span style='color:{theme.error}'>{error}</span>"

        cells = case_file.domain.nx * case_file.domain.ny * case_file.domain.nz
        mach_number = case_file.flow.free_stream_velocity * 3 ** 0.5
        lines = [f"<b>{case_file.name}</b>", verdict, "",
                 f"tau = {case_file.flow.relaxation_time:.5f}", f"Mach = {mach_number:.3f}",
                 f"cells = {cells:,}", f"steps = {case.total_steps():,} (warmup {case.warmup_steps():,})"]
        if error is None:
            lines += estimate_lines(case_file, *self.estimate_inputs())
        lines += ["", "<b>Parts</b>", *self.part_reports]
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
        case = part_free_case(case_file)
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
        self.remember_recent(self.path)

        return True

    def save_as(self):
        """
        Ask for a file name and save there.

        Returns True when saved.
        """

        suggested = self.dialog_directory() / f"{self.case_file.name}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save case", str(suggested), "Case files (*.json)")
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
            self.save_layout()
            self.viewport.close()
            self.study_page.close_preview()
            event.accept()
        else:
            event.ignore()

    def update_title(self):
        """
        Window title: app version, case file name, * when unsaved.
        """

        name = self.path.name if self.path is not None else (self.case_file.name + " (unsaved)" if self.case_file else "")
        self.setWindowTitle(f"Homebrew CFD {__version__} - {name}{' *' if self.dirty else ''}")

    def layout_state(self):
        """
        Current window geometry, dock state and splitter positions.

        Returns a dict of QByteArray values.
        """

        state = {"window/geometry": self.saveGeometry(), "window/state": self.saveState()}
        for name, splitter in self.splitters.items():
            state[f"splitters/{name}"] = splitter.saveState()

        return state
    
    def saved_layout(self):
        """
        The layout stored when the window last closed (empty on first start).

        Returns a dict of QByteArray values.
        """

        settings = QSettings("Homebrew CFD", "Homebrew CFD")

        return {key: settings.value(key) for key in settings.allKeys() if
                key.startswith(("window/", "splitters/"))}
    
    def restore_layout(self, state):
        """
        Apply a layout; missing entries keep the current one.
        """

        if state.get("window/geometry") is not None:
            self.restoreGeometry(state["window/geometry"])

        if state.get("window/state") is not None:
            self.restoreState(state["window/state"])

        for name, splitter in self.splitters.items():
            if state.get(f"splitters/{name}") is not None:
                splitter.restoreState(state[f"splitters/{name}"])

    def save_layout(self):
        """
        Store the current layout for the next start.
        """

        settings = QSettings("Homebrew CFD", "Homebrew CFD")
        for key, value in self.layout_state().items():
            settings.setValue(key, value)

    def reset_layout(self):
        """
        Back to the as-built layout.
        """

        self.restore_layout(self.default_layout)
    
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
        Validate, make sure the case is saved, then launch the solver into the workspace Runs folder.
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
        self.runs.start(self.path, workspace_directory() / "Runs", self.case_file.name)
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
        self.remember_throughput(result)
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