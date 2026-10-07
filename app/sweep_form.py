# sweep form: one parametric sweep's part, parameter and values in the tree, editable until its study is written
from dataclasses import replace
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton, QSpinBox,
                               QVBoxLayout, QWidget)
from src.run.estimate import run_seconds, format_duration
from src.run.parametric import sweep_choices, sweep_labels, current_value, values_from_range
from app import theme

class SweepForm(QWidget):
    """
    One swept parameter on one part: a list of values or a start / end / count range, with the run count and time.
    """

    def __init__(self, case_file, timing_source, throughput_source, parent=None):
        """
        Fill the part and parameter choices from the case; timing_source returns the study's Timing and
        throughput_source returns MLUPS, for the time estimate.
        """

        super().__init__(parent)
        self.case_file = case_file
        self.timing_source = timing_source
        self.throughput_source = throughput_source
        self.part_names = []

        self.part_box = QComboBox()
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
        form = QFormLayout()
        form.addRow("Part", self.part_box)
        form.addRow("Parameter", self.parameter_box)
        form.addRow("Present value", self.current_label)
        form.addRow(self.list_radio, self.list_edit)
        form.addRow(self.range_radio, range_row)
        form.addRow("Runs", self.preview_label)
        self.inputs = QWidget()
        self.inputs.setLayout(form)

        self.error_label = QLabel()
        self.error_label.setStyleSheet(f"color: {theme.error}")
        self.error_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.inputs)
        layout.addWidget(self.error_label)
        layout.addStretch()

        self.part_box.currentIndexChanged.connect(self.fill_parameters)
        self.parameter_box.currentIndexChanged.connect(self.show_current)
        for signal in (self.list_edit.textChanged, self.start_box.valueChanged, self.end_box.valueChanged, self.count_box.valueChanged,
                       self.list_radio.toggled):
            signal.connect(self.update_preview)

        self.set_case(case_file)

    def sweepable_names(self):
        """
        Returns the names of the case's parts that have at least one sweep parameter.
        """

        return [spec.name for spec in self.case_file.geometry if sweep_choices(spec)]

    def selected_spec(self):
        """
        Returns the GeometrySpec of the chosen part, or None when no part can be swept.
        """

        return next((spec for spec in self.case_file.geometry if spec.name == self.part_box.currentText()), None)

    def set_case(self, case_file):
        """
        Follow the open case: same sweepable parts only refresh the present value, else the choices are rebuilt.
        """

        self.case_file = case_file
        names = self.sweepable_names()
        if names == self.part_names:
            self.update_present()
            self.update_preview()
            return

        chosen = self.part_box.currentText()
        parameter = self.parameter_box.currentData()
        self.part_names = names
        self.part_box.blockSignals(True)
        self.part_box.clear()
        self.part_box.addItems(names)
        self.part_box.setCurrentIndex(max(self.part_box.findText(chosen), 0))
        self.part_box.blockSignals(False)
        self.fill_parameters()

        index = self.parameter_box.findData(parameter)
        if index >= 0:
            self.parameter_box.setCurrentIndex(index)

    def fill_parameters(self):
        """
        List the parameters valid for the chosen part.
        """

        spec = self.selected_spec()
        self.parameter_box.blockSignals(True)
        self.parameter_box.clear()
        if spec is not None:
            for parameter in sweep_choices(spec):
                self.parameter_box.addItem(sweep_labels[parameter], parameter)

        self.parameter_box.blockSignals(False)
        self.show_current()

    def show_current(self):
        """
        Show the part's present value and start the range around it.
        """

        self.update_present()
        spec = self.selected_spec()
        parameter = self.parameter_box.currentData()
        if spec is not None and parameter is not None:
            value = current_value(spec, parameter)
            self.start_box.setValue(value)
            self.end_box.setValue(value + 4)

        self.update_preview()

    def update_present(self):
        """
        Show the part's present value of the chosen parameter.
        """

        spec = self.selected_spec()
        parameter = self.parameter_box.currentData()
        if spec is None or parameter is None:
            self.current_label.setText("-")
            return

        self.current_label.setText(f"{current_value(spec, parameter):g}")

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
        Run count and total time at the stored throughput (the study's timing, the same for every value).
        """

        try:
            count = len(self.parsed_values())
            seconds = count * run_seconds(replace(self.case_file, timing=self.timing_source()), self.throughput_source())
        except (ValueError, ZeroDivisionError):
            self.preview_label.setText("-")
            return

        self.preview_label.setText(f"{count} runs, about {format_duration(seconds)}")

    def check(self):
        """
        Show inline why the sweep cannot run yet, or clear the message.

        Returns True when the part and values are usable.
        """

        try:
            if self.selected_spec() is None:
                raise ValueError("no part can be swept")

            self.parsed_values()
        except ValueError as error:
            text = str(error)
            self.error_label.setText(text[:1].upper() + text[1:])
            return False

        self.error_label.setText("")

        return True

    def settings(self):
        """
        Returns the sweep as a dict with part, parameter and values (the keys study.json uses).
        """

        return {"part": self.part_box.currentText(), "parameter": self.parameter_box.currentData(), "values": self.parsed_values()}

    def load_settings(self, sweep):
        """
        Copy part, parameter and values from a sweep dict, as far as this case still has them.
        """

        index = self.part_box.findText(sweep["part"])
        if index >= 0:
            self.part_box.setCurrentIndex(index)

        index = self.parameter_box.findData(sweep["parameter"])
        if index >= 0:
            self.parameter_box.setCurrentIndex(index)

        self.list_radio.setChecked(True)
        self.list_edit.setText(", ".join(f"{value:.10g}" for value in sweep["values"]))

