# study form: one study's name, type, solver and timing in the tree, editable until it is written, then read-only
from dataclasses import asdict, replace
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from src.run.estimate import run_seconds, format_duration
from src.run.parametric import study_types, check_study_name
from app.property_form import PropertyForm
from app import theme

class StudyForm(QWidget):
    """
    One study: its name and type, the solver and timing settings of its runs, and the time one run takes.
    """

    run_requested = Signal() # Run study clicked on an unwritten study
    name_changed = Signal(str) # the study name as typed
    settings_changed = Signal() # a solver or timing edit was accepted
    rejected = Signal(str) # why a typed edit was refused

    def __init__(self, case_file, throughput_source, choices, name, solver, timing, parent=None):
        """
        Editors for the solver (StudySolver) and timing (Timing) objects, which the form writes into;
        throughput_source returns MLUPS for the time estimate.
        """

        super().__init__(parent)
        self.case_file = case_file
        self.throughput_source = throughput_source
        self.solver = solver
        self.timing = timing

        # set by the main window: tree nodes, the sweep, and the study folder once written
        self.tree_item = None
        self.sweep_form = None
        self.sweep_item = None
        self.folder = None

        self.name_edit = QLineEdit(name)
        self.type_box = QComboBox()
        for key, label in study_types.items():
            self.type_box.addItem(label, key)

        self.estimate_label = QLabel()
        header = QFormLayout()
        header.addRow("Study name", self.name_edit)
        header.addRow("Study type", self.type_box)
        header.addRow("Time per run", self.estimate_label)

        self.solver_form = PropertyForm(solver, choices)
        self.timing_form = PropertyForm(timing, choices)
        self.inputs = QWidget()
        inputs_layout = QVBoxLayout(self.inputs)
        inputs_layout.setContentsMargins(0, 0, 0, 0)
        inputs_layout.addLayout(header)
        for title, form in (("Solver", self.solver_form), ("Timing", self.timing_form)):
            group = QGroupBox(title)
            QVBoxLayout(group).addWidget(form)
            inputs_layout.addWidget(group)

        self.state_label = QLabel()
        self.state_label.setVisible(False)
        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {theme.error}")
        self.error_label.setWordWrap(True)
        self.run_button = QPushButton("Run study")
        self.run_button.clicked.connect(self.run_requested)

        layout = QVBoxLayout(self)
        layout.addWidget(self.inputs)
        layout.addWidget(self.state_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.run_button)
        layout.addStretch()

        self.name_edit.textChanged.connect(self.name_changed)
        for form in (self.solver_form, self.timing_form):
            form.changed.connect(self.on_settings_changed)
            form.rejected.connect(self.rejected)

        self.update_estimate()

    def set_case(self, case_file):
        """
        Follow the open case: refresh the time estimate and the sweep's part choices.
        """

        self.case_file = case_file
        self.update_estimate()
        if self.sweep_form is not None:
            self.sweep_form.set_case(case_file)

    def on_settings_changed(self):
        """
        A solver or timing edit was accepted: refresh the estimates and tell the window.
        """

        self.update_estimate()
        if self.sweep_form is not None:
            self.sweep_form.update_preview()

        self.settings_changed.emit()

    def update_estimate(self):
        """
        Time of one run at the stored throughput, from this study's timing.
        """

        try:
            seconds = run_seconds(replace(self.case_file, timing=self.timing), self.throughput_source())
        except (ValueError, ZeroDivisionError):
            self.estimate_label.setText("-")
            return

        self.estimate_label.setText(f"about {format_duration(seconds)}")

    def check(self):
        """
        Show inline why the study cannot run yet, or clear the message.

        Returns True when the name is usable.
        """

        name = self.name_edit.text().strip()
        try:
            if not name:
                raise ValueError("enter a study name")

            check_study_name(name)
        except ValueError as error:
            text = str(error)
            self.error_label.setText(text[:1].upper() + text[1:])
            return False

        self.error_label.setText("")

        return True

    def show_error(self, text):
        """
        Show a message below the form (for example a refusal from the study writer).
        """

        self.error_label.setText(text)

    def settings(self):
        """
        Returns the study as a dict with name, type, solver, timing and no sweep (the keys study.json uses).
        """

        return {"name": self.name_edit.text().strip(), "type": self.type_box.currentData(), "solver": asdict(self.solver),
                "timing": asdict(self.timing), "sweep": None}

    def show_saved(self, state):
        """
        Show a study already on disk: still editable, with its state; running it again replaces its results.
        """

        self.run_button.setText("Run again")
        self.state_label.setVisible(True)
        self.show_state(state)

    def show_state(self, state):
        """
        Show the state line of a written study (for example 3 of 5 finished).
        """

        self.state_label.setText(f"State: {state}")
