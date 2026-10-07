# main window: setup tree, property forms, case summary with live validation, file handling and autosave
import shutil
from pathlib import Path
from PySide6.QtCore import QSize, QTimer, Qt, QSettings, QStandardPaths
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QTabWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget, QPlainTextEdit, QProgressBar, QMenu, QScrollArea, QToolBar, QToolButton)
from src import __version__
from src.run.case import setting_choices, check_choice, Timing
from src.run.case_file import load_case_file, save_case_file, geometry_choices, stl_spec, part_free_case, ahmed_fit_error
from src.run.estimate import default_throughput
from src.post.run_log import gpu_total_gb
from src.geometry.preview import load_preview
from app.property_form import PropertyForm, PartForm
from app.viewport import CaseViewport, named_views, corner_views, read_part_mesh, estimate_lines, preview_report
from app.run_control import RunController, PreviewController
from app.monitor_plot import MonitorPlot
from app.ribbon import Ribbon
from app import theme
from app.new_study import StudyPage
from src.run.project import create_project, is_project, runs_directory_for, adopt_geometry
from app.manual import ManualWindow
from app.run_queue import RunQueue
from app.study_form import StudyForm
from app.sweep_form import SweepForm
from app.study_view import StudyView, study_rows, read_samples
from src.run.parametric import (write_study, load_study, sweep_labels, default_study_name, apply_study_settings, study_defaults,
                                StudySolver)

template_directory = Path(__file__).resolve().parent.parent / "cases" / "templates"
autosave_milliseconds = 60_000
stop_confirm_milliseconds = 3000
recent_cases_limit = 8
# bump when the window structure changes: layouts saved under an older version are ignored once
layout_version = 2
# the UI offers every choice the engine accepts, except a custom start (a case file cannot hold an initial field)
form_choices = {**setting_choices, **geometry_choices, "start": ("rest_ramp", "uniform")}
# tree item data: the role holding what kind of node it is (study_branch, study or sweep)
kind_role = Qt.UserRole + 1
queue_columns = ("#", "Case", "Study value", "State", "Result", "Run folder")

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
    Tree and settings form on the left; graphics above summary, console and queue tabs on the right.
    """

    def __init__(self):
        """
        Build menus, panes and the autosave timer; start empty until a case is opened.
        """

        super().__init__()
        self.case_file = None
        self.path = None
        self.dirty = False

        # tree and settings form side by side on the left
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_tree_menu)
        self.tree.currentItemChanged.connect(self.show_node)
        self.forms = QStackedWidget()
        self.geometry_item = None
        self.study_item = None
        self.result_items = {} # study folder path -> its node under Results
        self.study_items = {} # study folder path -> its tree node under Study
        self.studies = [] # every study form, in the order of the nodes under Study
        self.draft_forms = [] # studies not written yet
        self.saved_forms = {} # study folder path -> its form (read-only)
        left_pane = QSplitter()
        left_pane.addWidget(self.tree)
        left_pane.addWidget(self.forms)
        left_pane.setSizes([200, 400])

        # graphics area: the case viewport, the live monitors, and the study plot, one at a time
        self.viewport = CaseViewport(self)
        self.part_reports = []
        self.monitor = MonitorPlot()
        self.study_view = StudyView()
        self.study_monitor = MonitorPlot()
        self.graphics = QStackedWidget()
        self.graphics.addWidget(self.viewport.plotter)
        self.graphics.addWidget(self.monitor)
        self.graphics.addWidget(self.study_view)
        self.graphics.addWidget(self.study_monitor)

        # bottom tabs: summary, console, queue
        self.summary = QLabel("No case open: the start page (Ctrl+N), New from a template, or Open.")
        self.summary.setAlignment(Qt.AlignTop)
        self.summary.setWordWrap(True)
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setWidget(self.summary)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.queue_table = QTableWidget(0, len(queue_columns))
        self.queue_table.setHorizontalHeaderLabels(queue_columns)
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.queue_table.customContextMenuRequested.connect(self.show_queue_menu)
        queue_pane = QWidget()
        queue_layout = QVBoxLayout(queue_pane)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.addWidget(self.build_queue_toolbar())
        queue_layout.addWidget(self.queue_table)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(summary_scroll, "Summary")
        self.bottom_tabs.addTab(self.console, "Console")
        self.queue_tab_index = self.bottom_tabs.addTab(queue_pane, "Queue")
        self.bottom_tabs.setTabVisible(self.queue_tab_index, False)  # shown once the project has a parametric sweep queued

        right_pane = QSplitter(Qt.Vertical)
        right_pane.addWidget(self.graphics)
        right_pane.addWidget(self.bottom_tabs)
        right_pane.setSizes([570, 190])
        main_splitter = QSplitter()
        main_splitter.addWidget(left_pane)
        main_splitter.addWidget(right_pane)
        main_splitter.setSizes([600, 600])

        self.device_total_gb = gpu_total_gb()

        # pages: the setup view and the start page
        self.pages = QStackedWidget()
        self.pages.addWidget(main_splitter)
        self.study_page = StudyPage(self.viewport.stl_cache, self.dialog_directory, self.recent_paths, self.estimate_inputs)
        self.study_page.created.connect(self.on_study_created)
        self.study_page.open_requested.connect(self.on_study_open_requested)
        self.study_page.blank_requested.connect(self.on_study_blank_requested)
        self.study_page.template_requested.connect(lambda path: self.open_path(path, as_template=True))
        self.study_page.set_templates(sorted(template_directory.glob("*.json")))
        self.pages.addWidget(self.study_page)
        self.setCentralWidget(self.pages)
        self.splitters = {"main": main_splitter, "left": left_pane, "right": right_pane}

        self.manual_window = None

        # runs: controller, status-bar progress
        self.runs = RunController(self)
        self.runs.output.connect(self.on_run_output)
        self.runs.progress.connect(self.on_run_progress)
        self.runs.finished.connect(self.on_run_finished)
        self.runs.samples.connect(self.monitor.update_samples)
        self.last_result = None

        self.queue = RunQueue(workspace_directory() / "queue")
        self.queue_entry = None # id of the queue entry now running, None for a direct run
        self.queue_paused = False
        self.current_study = None # folder path of the parametric study shown in the study view
        self.study_parameter = "Value"

        # geometry preview: built by the solver in its own process, drawn over the case
        self.previews = PreviewController(self)
        self.previews.finished.connect(self.on_preview_finished)
        self.preview_lines = []

        self.refresh_queue()
        self.run_progress = QProgressBar()
        self.run_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.run_progress)

        # two-click stop: the first click arms it for a few seconds, the second stops the run
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.reset_stop_confirm)
        self.stop_button = QToolButton()
        self.stop_button.setIcon(theme.icon("stop"))
        self.stop_button.setIconSize(QSize(14, 14))
        self.stop_button.setMinimumSize(22, 22)
        self.stop_button.setToolTip("Stop the run (click twice)")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.request_stop)
        self.statusBar().addPermanentWidget(self.stop_button)

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
        Ribbon tabs Home (case files), View (camera), Run (solver and studies) and Help.
        """

        self.ribbon = Ribbon()
        self.setMenuWidget(self.ribbon)

        # home: new from a template, open, save
        template_menu = QMenu(self)
        
        for template_path in sorted(template_directory.glob("*.json")):
            action = template_menu.addAction(template_path.stem)
            action.triggered.connect(lambda checked=False, template_path=template_path: self.open_path(template_path, as_template=True))
        
        new_study_action = self.make_action("Start page", theme.icon("new_study"), QKeySequence.New, self.new_study)  # Ctrl+N only, not on the ribbon
        new_action = self.make_action("New", theme.icon("new"), None, None)
        new_action.setMenu(template_menu)
        open_action = self.make_action("Open", theme.icon("open"), QKeySequence.Open, self.open_dialog)
        save_action = self.make_action("Save", theme.icon("save"), QKeySequence.Save, self.save)
        save_as_action = self.make_action("Save as", theme.icon("save_as"), QKeySequence.SaveAs, self.save_as)
        self.make_action("Quit", None, QKeySequence.Quit, self.close)
        self.case_loading_actions = [new_study_action, new_action, open_action]
        home_tab = self.ribbon.add_tab("Home")
        home_tab.add_group("Case", (new_action, open_action, save_action, save_as_action))

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
        self.preview_action = self.make_action("Voxels", theme.icon("voxels"), None, None)
        self.preview_action.setCheckable(True)
        self.preview_action.toggled.connect(self.toggle_preview)
        self.case_loading_actions.append(self.preview_action)
        view_tab.add_group("Geometry", (self.preview_action,))
        reset_layout_action = self.make_action("Reset layout", theme.icon("reset_layout"), 
                                               None, self.reset_layout)
        view_tab.add_group("Window", (reset_layout_action,))

        # run: start / stop (stop asks for a second click), new study
        self.run_action = self.make_action("Run", theme.icon("run"), QKeySequence("F5"), self.start_run)
        self.stop_action = self.make_action("Stop", theme.icon("stop"), QKeySequence("Shift+F5"), self.request_stop)
        self.stop_action.setEnabled(False)
        add_study_action = self.make_action("New study", theme.icon("new_study"), None, self.add_study)
        run_tab = self.ribbon.add_tab("Run")
        run_tab.add_group("Solver", (self.run_action, self.stop_action))
        run_tab.add_group("Study", (add_study_action,))

        # help: the manual
        manual_action = self.make_action("Manual", theme.icon("manual"), QKeySequence.HelpContents, self.show_manual)
        self.ribbon.add_tab("Help").add_group("Help", (manual_action,))

    def show_start(self, visible):
        """
        Show the start page with the ribbon hidden, or return to the normal setup view.
        """

        self.ribbon.setVisible(not visible)
        self.pages.setCurrentIndex(1 if visible else 0)

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
        self.preview_action.setChecked(False)
        self.draft_forms = []
        self.current_study = None
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
        Case (case name, flow, domain, models, geometry parts), Study (one node per study, its sweep below it)
        and Results (monitors, one node per written study).
        """

        self.tree.clear()
        while self.forms.count():
            self.forms.removeWidget(self.forms.widget(0))

        self.saved_forms = {}
        self.study_items = {}
        self.studies = []
        case_file = self.case_file
        case_item = QTreeWidgetItem(self.tree, ["Case"])
        name_form = PropertyForm(case_file, form_choices, ("name",))
        name_form.changed.connect(self.on_changed)
        name_form.rejected.connect(self.on_rejected)
        case_item.setData(0, Qt.UserRole, ("form", self.forms.addWidget(name_form)))
        nodes = (("Flow", case_file.flow), ("Domain", case_file.domain), ("Models", case_file.turbulence))
        for label, target in nodes:
            self.add_node(case_item, label, target, None)

        self.geometry_item = QTreeWidgetItem(case_item, ["Geometry"])
        for spec in case_file.geometry:
            self.add_node(self.geometry_item, spec.name, spec, None, part=True)

        self.study_item = QTreeWidgetItem(self.tree, ["Study"])
        self.study_item.setData(0, kind_role, "study_branch")
        results_item = QTreeWidgetItem(self.tree, ["Results"])
        self.result_items = {}

        # studies already on disk, then the ones not written yet
        for folder in self.study_folders():
            try:
                study = load_study(folder)
                form = self.make_study_form(study["name"], StudySolver(**study["solver"]), Timing(**study["timing"]))
                if study["sweep"] is not None:
                    self.attach_sweep(form, study["sweep"])
            except (OSError, ValueError, KeyError, TypeError):
                continue

            key = str(folder.resolve())
            form.folder = key
            form.show_saved(self.study_state(folder))
            self.saved_forms[key] = form
            self.studies.append(form)
            self.add_study_nodes(form)
            self.study_items[key] = form.tree_item
            result_label = f"{study['name']}: force coefficients" if study["sweep"] is None else f"{study['name']}: sweep plot"
            result_item = QTreeWidgetItem(results_item, [result_label])
            result_item.setData(0, Qt.UserRole, ("study_result", key))
            self.result_items[key] = result_item

        for form in self.draft_forms:
            form.set_case(case_file)
            self.studies.append(form)
            self.add_study_nodes(form)

        self.tree.expandAll()
        self.tree.setCurrentItem(case_item.child(0))

    def add_node(self, parent, label, target, field_names, part=False):
        """
        Add a tree node with its property form on the stack (a part gets only the fields of its kind).
        """

        if part:
            form = PartForm(target, form_choices)
        else:
            form = PropertyForm(target, form_choices, field_names)

        form.changed.connect(self.on_changed)
        form.rejected.connect(self.on_rejected)
        item = QTreeWidgetItem(parent, [label])
        item.setData(0, Qt.UserRole, ("form", self.forms.addWidget(form)))

    def on_rejected(self, text):
        """
        A typed edit was refused: say why in the status bar.
        """

        self.statusBar().showMessage(text, 10000)

    def add_study_node(self, parent, form, label, kind):
        """
        Add a node (kind study or sweep) with its form on the stack.

        Returns the tree item.
        """

        item = QTreeWidgetItem(parent, [label])
        item.setData(0, Qt.UserRole, ("form", self.forms.addWidget(form)))
        item.setData(0, kind_role, kind)

        return item

    def add_study_nodes(self, form):
        """
        Add a study's node under Study, and its sweep node below it when it has one.
        """

        form.tree_item = self.add_study_node(self.study_item, form, form.name_edit.text(), "study")
        if form.sweep_form is not None:
            form.sweep_item = self.add_study_node(form.tree_item, form.sweep_form, "Parametric sweep", "sweep")

    def make_study_form(self, name, solver, timing):
        """
        A study form wired to the window: refused edits, renaming, run and run again.

        Returns the StudyForm.
        """

        form = StudyForm(self.case_file, self.study_throughput, form_choices, name, solver, timing)
        form.rejected.connect(self.on_rejected)
        form.name_changed.connect(lambda text, form=form: self.rename_study(form, text))
        form.run_requested.connect(lambda form=form: self.run_study(form))
        form.settings_changed.connect(self.refresh_summary)

        return form

    def rename_study(self, form, text):
        """
        Show the typed study name on its tree node.
        """

        if form.tree_item is not None:
            form.tree_item.setText(0, text)

    def attach_sweep(self, form, sweep=None):
        """
        Give a study a parametric sweep form (values copied from a sweep dict when given), estimated with the study's timing.
        """

        form.sweep_form = SweepForm(self.case_file, lambda form=form: form.timing, self.study_throughput)
        if sweep is not None:
            form.sweep_form.load_settings(sweep)

    def study_of_item(self, item):
        """
        Returns the study form that owns a tree node (its own node or its sweep node), or None for any other node.
        """

        while item is not None:
            index = self.study_item.indexOfChild(item)
            if index >= 0:
                return self.studies[index]

            item = item.parent()

        return None

    def study_dict(self, form):
        """
        Returns the study as write_study takes it: the form's settings plus its sweep (None when it has none).
        """

        study = form.settings()
        if form.sweep_form is not None:
            study["sweep"] = form.sweep_form.settings()

        return study

    def summary_case_file(self):
        """
        The case as the selected study (else the first study) would run it, so the summary checks the solver and
        timing that will be used; the case itself when there is no study.

        Returns a CaseFile.
        """

        form = self.study_of_item(self.tree.currentItem()) if self.study_item is not None else None
        if form is None and self.studies:
            form = self.studies[0]

        if form is None:
            return self.case_file

        settings = form.settings()

        return apply_study_settings(self.case_file, settings["solver"], settings["timing"])

    def study_throughput(self):
        """
        Returns this machine's throughput in MLUPS, for the study time estimate.
        """

        return self.estimate_inputs()[0]

    def studies_directory(self):
        """
        Where this case's studies live: beside the case file in a project, else the workspace studies folder.

        Returns the Path.
        """

        if is_project(self.path):
            return self.path.parent / "studies"

        return workspace_directory() / "studies"

    def study_folders(self):
        """
        Returns the study folders (with a study.json) of the open case, sorted by name; none for an unsaved case.
        """

        directory = self.studies_directory()
        if self.path is None or not directory.is_dir():
            return []

        return sorted(folder for folder in directory.iterdir() if (folder / "study.json").exists())

    def study_state(self, folder):
        """
        Returns a short state line for a written study, from its queue entries (for example 3 finished, 2 pending).
        """

        counts = {}
        for entry in self.queue.study_entries(str(folder)):
            counts[entry["state"]] = counts.get(entry["state"], 0) + 1

        if not counts:
            return "not in the queue"

        return ", ".join(f"{count} {state}" for state, count in counts.items())

    def add_draft_study(self):
        """
        Add an unwritten study with the next free name Study N, the case's own solver and timing and no sweep, and
        select it.
        """

        if self.case_file is None:
            return

        solver, timing = study_defaults(self.case_file)
        taken = {form.name_edit.text() for form in self.draft_forms}
        form = self.make_study_form(default_study_name(self.studies_directory(), taken), solver, timing)
        self.draft_forms.append(form)
        self.studies.append(form)
        self.add_study_nodes(form)
        self.tree.setCurrentItem(form.tree_item)

    def add_study(self):
        """
        New study from the case's defaults (Study branch menu and the Run tab).
        """

        self.add_draft_study()

    def add_sweep(self, form):
        """
        Give an unwritten study a parametric sweep and select it.
        """

        self.attach_sweep(form)
        self.build_tree()
        self.tree.setCurrentItem(form.sweep_item)

    def remove_sweep(self, form):
        """
        Take the sweep off an unwritten study.
        """

        form.sweep_form = None
        form.sweep_item = None
        self.build_tree()
        self.tree.setCurrentItem(form.tree_item)

    def remove_study(self, form):
        """
        Drop an unwritten study.
        """

        self.draft_forms.remove(form)
        self.build_tree()
        self.tree.setCurrentItem(self.study_item)

    def show_tree_menu(self, position):
        """
        Right-click menu by node: Study branch (new study), unwritten study (add sweep, remove study), its sweep (remove sweep).
        A written study has no menu.
        """

        item = self.tree.itemAt(position)
        if item is None or self.case_file is None:
            return

        kind = item.data(0, kind_role)
        form = self.study_of_item(item)
        menu = QMenu(self)
        handlers = []
        if kind == "study_branch":
            handlers.append((menu.addAction("New study"), self.add_study))
        elif kind == "study" and form.folder is None:
            if form.sweep_form is None:
                handlers.append((menu.addAction("Add parametric sweep"), lambda: self.add_sweep(form)))

            handlers.append((menu.addAction("Remove study"), lambda: self.remove_study(form)))
        elif kind == "sweep" and form.folder is None:
            handlers.append((menu.addAction("Remove parametric sweep"), lambda: self.remove_sweep(form)))

        if not handlers:
            return

        chosen = menu.exec(self.tree.viewport().mapToGlobal(position))
        for action, handler in handlers:
            if chosen == action:
                handler()

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
        self.tree.setCurrentItem(self.geometry_item.child(self.geometry_item.childCount() - 1))
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
        Show the selected node: a form (case part or study) with the viewport, the monitors, or a study plot.
        Branch nodes keep the current view.
        """

        data = item.data(0, Qt.UserRole) if item is not None else None
        if data is None:
            return

        if data[0] == "form":
            self.forms.setCurrentIndex(data[1])
            self.graphics.setCurrentWidget(self.viewport.plotter)
            self.refresh_summary()  # the summary follows the selected study's solver and timing
        elif data[0] == "study_result":
            self.show_study_result(data[1])

    def show_study_result(self, study_path):
        """
        Make a written study current: a sweep shows its table and plot, a single run shows its monitor plot.
        """

        form = self.saved_forms[study_path]
        if form.sweep_form is not None:
            self.current_study = study_path
            self.study_parameter = sweep_labels.get(form.sweep_form.parameter_box.currentData(), "Value")
            self.refresh_queue()
            self.graphics.setCurrentWidget(self.study_view)
            return

        self.current_study = None
        self.refresh_queue()

        # the study running now shows the live monitors
        if any(entry["state"] == "running" for entry in self.queue.study_entries(study_path)):
            self.graphics.setCurrentWidget(self.monitor)
            return

        self.study_monitor.clear()
        entries = self.queue.study_entries(study_path)
        samples = read_samples(entries[0]["run_folder"]) if entries and entries[0]["run_folder"] else None
        if samples is not None:
            try:
                study = load_study(study_path)
                case = part_free_case(load_case_file(Path(study_path) / study["cases"][0]))
            except (OSError, ValueError, KeyError, TypeError):
                case = None

            if case is not None:
                self.study_monitor.set_timeline(case.total_steps(), case.warmup_steps(), case.flow_through_steps(), case.timing.sample_every)
                self.study_monitor.update_samples(samples)

        self.graphics.setCurrentWidget(self.study_monitor)

    def show_monitors(self):
        """
        Show the live force coefficients: select the running study's node under Results (a sweep keeps its plot
        node, so the live curves show directly).
        """

        entry = self.queue.find(self.queue_entry) if self.queue_entry is not None else None
        study_path = entry.get("study") if entry is not None else None
        item = self.result_items.get(study_path) if study_path else None
        if item is not None and entry.get("value") is None:
            self.tree.setCurrentItem(item)
            return

        self.graphics.setCurrentWidget(self.monitor)

    def on_changed(self):
        """
        Any accepted edit: mark unsaved and re-check the case.
        """

        self.preview_action.setChecked(False)
        self.dirty = True
        for form in self.draft_forms:
            form.set_case(self.case_file)

        self.part_reports = self.viewport.draw(self.case_file)
        self.refresh_summary()

    def toggle_preview(self, checked):
        """
        Checked: build the current case's geometry in the background; unchecked: back to the plain view.
        """

        if not checked:
            self.preview_lines = []
            if self.case_file is not None:
                self.part_reports = self.viewport.draw(self.case_file)
                self.refresh_summary()
            return

        if self.case_file is None or not self.case_file.geometry or self.previews.is_running():
            self.preview_action.setChecked(False)
            return

        # the case as edited, saved beside the preview so unsaved changes are included
        preview_directory = workspace_directory() / ".preview"
        preview_directory.mkdir(exist_ok=True)
        save_case_file(self.case_file, preview_directory / "case.json")
        self.previews.start(preview_directory / "case.json", preview_directory / "preview.npz")
        self.set_preview_busy(True)
        self.statusBar().showMessage("Building geometry preview...")

    def on_preview_finished(self, path, error):
        """
        Draw the preview over the case, or say why it failed.
        """

        self.set_preview_busy(False)
        if not path:
            self.preview_action.setChecked(False)
            self.statusBar().showMessage(f"Geometry preview failed: {error}", 10000)
            return

        # an edit while building made this preview stale
        if not self.preview_action.isChecked():
            return

        preview = load_preview(path)
        self.part_reports = self.viewport.draw(self.case_file, part_opacity=0.25)
        self.viewport.show_preview(preview)
        self.preview_lines = preview_report(preview)
        self.statusBar().showMessage("Geometry preview ready", 5000)
        self.refresh_summary()

    def refresh_summary(self):
        """
        Derived numbers (tau, Mach, cells, steps) and the validation verdict, recomputed after every edit.
        """

        case_file = self.summary_case_file()
        case = part_free_case(case_file)

        error = self.validation_error(case_file)
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
        if self.preview_lines:
            lines += ["", "<b>Voxels</b>", *self.preview_lines]
        if self.last_result is not None:
            lines += ["", "<b>Last run</b>", *self.result_lines(self.last_result)]

        self.summary.setText("<br>".join(lines))
        self.update_title()

    def validation_error(self, case_file=None):
        """
        The engine's verdict on a case (the open one by default), without building geometry.

        Returns the error message, or None when the case can run.
        """

        if case_file is None:
            case_file = self.case_file

        case = part_free_case(case_file)
        try:
            case.validate()
            for spec in case_file.geometry:
                check_choice("geometry kind", spec.kind, geometry_choices["kind"])
                check_choice("geometry wall", spec.wall, geometry_choices["wall"])
                if spec.kind == "ahmed" and spec.wall != "staircase":
                    raise ValueError(f"{spec.name}: an ahmed body needs staircase walls")

                fit_error = ahmed_fit_error(spec, case_file.domain) if spec.kind == "ahmed" else None
                if fit_error is not None:
                    raise ValueError(fit_error)
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

        # a project keeps its own copy of every STL, so the folder is complete on its own
        if is_project(self.path):
            try:
                adopt_geometry(self.case_file, self.path)
            except OSError as error:
                QMessageBox.critical(self, "Cannot copy geometry", str(error))
                return False

        save_case_file(self.case_file, self.path)
        self.dirty = False
        self.update_title()
        self.remember_recent(self.path)

        return True

    def save_as(self):
        """
        Ask for a file, starting in the project folder (or the workspace) with the case name filled in.
        A file saved straight into the workspace gets its own project folder.

        Returns True when saved.
        """

        directory = self.path.parent if is_project(self.path) else workspace_directory()
        chosen, _ = QFileDialog.getSaveFileName(self, "Save case as", str(directory / f"{self.case_file.name}.json"), "Case files (*.json)")
        if not chosen:
            return False

        path = Path(chosen)
        if path.suffix.lower() != ".json":
            path = path.with_name(path.name + ".json")

        if path.parent.resolve() == workspace_directory().resolve():
            try:
                path = create_project(workspace_directory(), path.stem)
            except (FileExistsError, ValueError) as error:
                QMessageBox.warning(self, "Cannot create project", sentence_case(str(error)))
                return False

        self.path = path

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

        state = {"window/geometry": self.saveGeometry(), "window/state": self.saveState(layout_version)}
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
            self.restoreState(state["window/state"], layout_version)

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
        self.stop_button.setVisible(running)
        self.reset_stop_confirm()

    def set_preview_busy(self, busy):
        """
        Lock the window like a run while the geometry preview builds, with a moving (indeterminate) progress bar.
        """

        self.forms.setEnabled(not busy)
        for action in self.case_loading_actions:
            action.setEnabled(not busy)

        self.run_action.setEnabled(not busy)
        self.run_progress.setRange(0, 0 if busy else 100)
        self.run_progress.setVisible(busy)
    
    def start_run(self):
        """
        Run the selected study (the first one when the selection is not in a study, a new Study 1 when there is none).
        A study that has run already runs again and replaces its results.
        """

        if self.case_file is None or self.runs.is_running():
            return

        item = self.tree.currentItem()
        form = self.study_of_item(item)
        if form is None and self.studies:
            form = self.studies[0]

        if form is None:
            self.add_study()
            form = self.studies[-1]

        self.run_study(form)

    def discard_study_results(self, form):
        """
        Before a study that has run runs again: confirm, then delete its run folders, queue entries and study folder.

        Returns True when the old results are gone and the study can be written again.
        """

        entries = self.queue.study_entries(form.folder)
        if any(entry["state"] == "running" for entry in entries):
            self.statusBar().showMessage("This study is running", 5000)
            return False

        answer = QMessageBox.question(self, "Run again", f"Running {form.name_edit.text()} again replaces its results. Continue?")
        if answer != QMessageBox.Yes:
            return False

        for entry in entries:
            if entry["run_folder"]:
                shutil.rmtree(entry["run_folder"], ignore_errors=True)
            self.queue.remove(entry["id"])

        shutil.rmtree(form.folder, ignore_errors=True)

        return True

    def run_study(self, form):
        """
        Validate, save the case, write the study folder beside it, queue every case file (one per swept value, or one)
        and start the queue.
        """

        if self.case_file is None or not form.check():
            return

        if form.sweep_form is not None and not form.sweep_form.check():
            return

        study = self.study_dict(form)
        applied = apply_study_settings(self.case_file, study["solver"], study["timing"])
        error = self.validation_error(applied)
        if error is not None:
            QMessageBox.warning(self, "Cannot start study", error)
            return

        if (self.dirty or self.path is None) and not self.save():
            return

        if form.folder is not None and not self.discard_study_results(form):
            return

        folder = self.studies_directory() / study["name"]
        try:
            _, written = write_study(folder, self.case_file, study)
        except (OSError, ValueError) as error:
            form.show_error(sentence_case(str(error)))
            return

        sweep = study["sweep"]
        values = [None] * len(written) if sweep is None else sweep["values"]
        runs_directory = runs_directory_for(self.path, workspace_directory())
        study_path = str(folder.resolve())
        for (variant, _), value in zip(written, values):
            self.queue.add(variant, runs_directory, study=study_path, value=value)

        # the study is on disk now: rebuild the tree so it moves from drafts to written studies
        if form in self.draft_forms:
            self.draft_forms.remove(form)
        self.build_tree()
        self.current_study = None if sweep is None else study_path
        self.study_parameter = "Value" if sweep is None else sweep_labels[sweep["parameter"]]
        self.tree.setCurrentItem(self.study_items[study_path])
        self.refresh_queue()
        self.run_queue()

    def clear_queue(self):
        """
        Drop the open project's waiting runs; the one running carries on.
        """

        self.queue.clear_pending(runs_directory_for(self.path, workspace_directory()))
        self.refresh_queue()

    def pause_queue(self):
        """
        Stop the queue from starting the next entry; the run in progress carries on.
        """

        self.queue_paused = True
        self.statusBar().showMessage("Queue paused: the current run finishes, then the queue waits", 8000)

    def build_queue_toolbar(self):
        """
        Small toolbar above the queue table.

        Returns the QToolBar.
        """

        toolbar = QToolBar()
        buttons = (("Start/Resume", self.run_queue), ("Pause", self.pause_queue), ("Remove", self.remove_selected),
                   ("Move up", lambda: self.move_selected(-1)), ("Move down", lambda: self.move_selected(1)),
                   ("Clear queue", self.clear_queue))
        for text, handler in buttons:
            toolbar.addAction(text).triggered.connect(handler)

        return toolbar

    def selected_queue_id(self):
        """
        Returns the id of the selected queue row, or None when nothing is selected.
        """

        row = self.queue_table.currentRow()
        item = self.queue_table.item(row, 0) if row >= 0 else None

        return item.data(Qt.UserRole) if item is not None else None

    def remove_selected(self):
        """
        Remove the selected queue entry (never the running one).
        """

        identifier = self.selected_queue_id()
        if identifier is not None:
            self.queue.remove(identifier)
            self.refresh_queue()

    def move_selected(self, offset):
        """
        Move the selected queue entry up (-1) or down (+1).
        """

        identifier = self.selected_queue_id()
        if identifier is not None:
            self.queue.move(identifier, offset)
            self.refresh_queue()

    def refresh_queue(self):
        """
        Rebuild the queue table (the open project's waiting and running entries only), the written studies' states
        and the study view.
        """

        project_entries = []
        entries = []
        if self.case_file is not None:
            runs_directory = runs_directory_for(self.path, workspace_directory())
            project_entries = self.queue.project_entries(runs_directory)
            entries = self.queue.active_entries(runs_directory)

        self.queue_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            value = "" if entry.get("value") is None else f"{entry['value']:g}"
            cells = (str(row + 1), entry["name"], value, entry["state"], entry["headline"], entry["run_folder"])
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, entry["id"])
                self.queue_table.setItem(row, column, item)

        self.queue_table.resizeColumnsToContents()
        self.queue_table.horizontalHeader().setStretchLastSection(True)

        # the queue tab appears once this project has queued a parametric sweep
        self.bottom_tabs.setTabVisible(self.queue_tab_index, any(entry.get("value") is not None for entry in project_entries))

        for study_path, form in self.saved_forms.items():
            form.show_state(self.study_state(study_path))

        study_entries = self.queue.study_entries(self.current_study) if self.current_study else []
        self.study_view.show_rows(study_rows(study_entries), self.study_parameter)

    def show_queue_menu(self, position):
        """
        Right-click menu on a queue row: remove, move up, move down (waiting runs only; the running one is fixed).
        """

        item = self.queue_table.itemAt(position)
        if item is None:
            return

        identifier = item.data(Qt.UserRole)
        waiting = self.queue.find(identifier)["state"] == "pending"
        menu = QMenu(self)
        remove_action = menu.addAction("Remove")
        up_action = menu.addAction("Move up")
        down_action = menu.addAction("Move down")
        for action in (remove_action, up_action, down_action):
            action.setEnabled(waiting)
        chosen = menu.exec(self.queue_table.viewport().mapToGlobal(position))

        if chosen == remove_action:
            self.queue.remove(identifier)
        elif chosen == up_action:
            self.queue.move(identifier, -1)
        elif chosen == down_action:
            self.queue.move(identifier, 1)

        self.refresh_queue()

    def run_queue(self):
        """
        Resume the queue: start the next pending entry if nothing is running.
        """

        self.queue_paused = False
        self.start_next_queued()

    def start_next_queued(self):
        """
        Launch the first pending entry through the run controller.
        """

        entry = self.queue.next_pending(runs_directory_for(self.path, workspace_directory()))
        if entry is None or self.runs.is_running() or self.queue_paused:
            return

        self.queue_entry = entry["id"]
        self.queue.mark(entry["id"], "running")
        self.console.clear()
        self.monitor.clear()
        self.last_result = None
        self.run_progress.setValue(0)
        self.set_running(True)
        self.show_monitors()
        self.runs.start(entry["case"], entry["runs"], entry["name"])
        self.statusBar().showMessage(f"Queue: running {entry['name']}")
        self.refresh_queue()

    def request_stop(self):
        """
        Stop needs two clicks within a few seconds: the first arms it ("Stop?" in amber), the second stops the run.
        """

        if not self.runs.is_running():
            return

        if self.stop_timer.isActive():
            self.reset_stop_confirm()
            self.runs.stop()
            return

        self.stop_button.setText("Stop?")
        self.stop_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.stop_button.setStyleSheet(f"color: {theme.amber}")
        self.stop_action.setText("Stop?")
        self.stop_timer.start(stop_confirm_milliseconds)

    def reset_stop_confirm(self):
        """
        Disarm the stop buttons and put their plain look back.
        """

        self.stop_timer.stop()
        self.stop_button.setText("")
        self.stop_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.stop_button.setStyleSheet("")
        self.stop_action.setText("Stop")

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

        if self.queue_entry is not None:
            state = result.get("status", "failed")
            headline = ""
            if result.get("parts"):
                name, statistics = next(iter(result["parts"].items()))
                headline = f"{name} C_x {statistics['x']['mean']:.4f} C_y {statistics['y']['mean']:.4f}"

            self.queue.mark(self.queue_entry, state, result.get("run_folder", ""), headline)

            # stop ends the whole study: its waiting runs are dropped so starting another study does not resume it
            if state == "stopped":
                self.queue_paused = True
                study_path = self.queue.find(self.queue_entry).get("study")
                if study_path:
                    for entry in self.queue.study_entries(study_path):
                        if entry["state"] == "pending":
                            self.queue.remove(entry["id"])

            self.queue_entry = None
            self.refresh_queue()

            QTimer.singleShot(0, self.start_next_queued)

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
    
    def show_manual(self):
        """
        Open the manual window (one at a time; brought to the front if already open).
        """

        if self.manual_window is None:
            self.manual_window = ManualWindow(self)
        self.manual_window.show()
        self.manual_window.raise_()