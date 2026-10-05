# new parametric study dialog: pick a part and parameter, give the values, see the run count and time
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QRadioButton, QSpinBox, QVBoxLayout)
from src.run.estimate import run_seconds, format_duration
from src.run.parametric import sweep_choices, sweep_labels, current_value, values_from_range, unique_study_name
from app import theme

class StudyDialog(QDialog):
    """
    One swept parameter on one part: a list of values or a start / end / count range.
    """

    def __init__(self, case_file, throughput, studies_directory, parent=None):
        """
        Fill the part and parameter choices from the case; throughput (MLUPS) feeds the time estimate.
        The default name is the first one not already used in the studies directory.
        """

        super().__init__(parent)
        self.setWindowTitle("New parametric study")
        self.case_file = case_file
        self.throughput = throughput

        self.name_edit = QLineEdit(unique_study_name(studies_directory, f"{case_file.name}_study"))
        self.part_box = QComboBox()
        for spec in case_file.geometry:
            if sweep_choices(spec):
                self.part_box.addItem(spec.name)

        self.parameter_box = QComboBox()
        self.current_label = QLabel()
        self.list_radio = QRadioButton("List")
        self.range_radio = QRadioButton("Range")
        self.list_radio.setChecked(True)
        self.list_edit = QLineEdit()
        self.list_edit.setPlaceholderText("2, 4, 6, 8")
        self.start_box = QDoubleSpinBox()
        self.end_box = QDoubleSpinBox()
        for box in (self.start_box, self.end_box):
            box.setRange(-10000.0, 10000.0)
            box.setDecimals(3)

        self.count_box = QSpinBox()
        self.count_box.setRange(2, 100)
        self.count_box.setValue(5)
        range_row = QHBoxLayout()
        for label, widget in (("start", self.start_box), ("end", self.end_box), ("count", self.count_box)):
            range_row.addWidget(QLabel(label))
            range_row.addWidget(widget)

        self.preview_label = QLabel()
        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {theme.error}")
        self.error_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Study name", self.name_edit)
        form.addRow("Part", self.part_box)
        form.addRow("Parameter", self.parameter_box)
        form.addRow("Present value", self.current_label)
        form.addRow(self.list_radio, self.list_edit)
        form.addRow(self.range_radio, range_row)
        form.addRow("Runs", self.preview_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

        self.part_box.currentIndexChanged.connect(self.fill_parameters)
        self.parameter_box.currentIndexChanged.connect(self.show_current)
        for signal in (self.list_edit.textChanged, self.start_box.valueChanged, self.end_box.valueChanged, self.count_box.valueChanged,
                       self.list_radio.toggled):
            signal.connect(self.update_preview)

        self.fill_parameters()

    def selected_spec(self):
        """
        Returns the GeometrySpec of the chosen part, or None when no part can be swept.
        """

        return next((spec for spec in self.case_file.geometry if spec.name == self.part_box.currentText()), None)

    def fill_parameters(self):
        """
        List the parameters valid for the chosen part.
        """

        spec = self.selected_spec()
        self.parameter_box.clear()
        if spec is not None:
            for parameter in sweep_choices(spec):
                self.parameter_box.addItem(sweep_labels[parameter], parameter)

        self.show_current()

    def show_current(self):
        """
        Show the part's present value and start the range around it.
        """

        spec = self.selected_spec()
        parameter = self.parameter_box.currentData()
        if spec is None or parameter is None:
            return

        value = current_value(spec, parameter)
        self.current_label.setText(f"{value:g}")
        self.start_box.setValue(value)
        self.end_box.setValue(value + 4)
        self.update_preview()

    def parsed_values(self):
        """
        The values as typed: the list, or the range.

        Returns a list of floats; raises ValueError for text that is not numbers.
        """

        if self.range_radio.isChecked():
            return values_from_range(self.start_box.value(), self.end_box.value(), self.count_box.value())

        values = [float(piece) for piece in self.list_edit.text().replace(";", ",").split(",") if piece.strip()]
        if not values:
            raise ValueError("enter at least one value")

        return values

    def update_preview(self):
        """
        Run count and total time at the stored throughput (same domain and timing for every variant).
        """

        try:
            count = len(self.parsed_values())
        except ValueError:
            self.preview_label.setText("-")
            return

        seconds = count * run_seconds(self.case_file, self.throughput)
        self.preview_label.setText(f"{count} runs, about {format_duration(seconds)}")

    def validate_and_accept(self):
        """
        Accept when the name, part and values are usable; else show why inline.
        """

        try:
            if not self.name_edit.text().strip():
                raise ValueError("enter a study name")

            if self.selected_spec() is None:
                raise ValueError("no part can be swept")

            self.parsed_values()
        except ValueError as error:
            text = str(error)
            self.error_label.setText(text[:1].upper() + text[1:])
            return

        self.accept()

    def result_values(self):
        """
        Returns (study_name, part_name, parameter, values).
        """

        return (self.name_edit.text().strip(), self.part_box.currentText(), self.parameter_box.currentData(), self.parsed_values())
