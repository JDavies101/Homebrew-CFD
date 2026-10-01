# main window: setup tree, property forms, case summary with live validation, file handling and autosave
from pathlib import Path
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)
from src import __version__
from src.run.case import Case, setting_choices, check_choice
from src.run.case_file import load_case_file, save_case_file, geometry_choices
from app.property_form import PropertyForm

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
        splitter.addWidget(self.tree)
        splitter.addWidget(self.forms)
        splitter.addWidget(summary_pane)
        splitter.setSizes([220, 520, 300])
        self.setCentralWidget(splitter)
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
        self.refresh_summary()

    def refresh_summary(self):
        """
        Derived numbers (tau, Mach, cells, steps) and the validation verdict, recomputed after every edit.
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
            verdict = "<span style='color:#2a2'>Ready to run</span>"
        except (ValueError, ZeroDivisionError) as error:
            verdict = f"<span style='color:#d33'>{error}</span>"

        cells = case_file.domain.nx * case_file.domain.ny * case_file.domain.nz
        mach_number = case_file.flow.free_stream_velocity * 3 ** 0.5
        lines = [f"<b>{case_file.name}</b>", verdict, "",
                 f"tau = {case_file.flow.relaxation_time:.5f}", f"Mach = {mach_number:.3f}",
                 f"cells = {cells:,}", f"steps = {case.total_steps():,} (warmup {case.warmup_steps():,})",
                 f"parts = {len(case_file.geometry)}"]
        self.summary.setText("<br>".join(lines))
        self.update_title()

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
        Ask before closing with unsaved edits.
        """

        if self.confirm_discard():
            event.accept()
        else:
            event.ignore()

    def update_title(self):
        """
        Window title: app version, case file name, * when unsaved.
        """

        name = self.path.name if self.path is not None else (self.case_file.name + " (unsaved)" if self.case_file else "")
        self.setWindowTitle(f"Homebrew CFD {__version__} - {name}{' *' if self.dirty else ''}")