# study view: result table and C_x / C_y against the swept value for one parametric study
import json
from pathlib import Path
import numpy as np
import pyqtgraph
from PySide6.QtWidgets import QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from app import theme

columns = ("Value", "State", "C_x", "C_y", "SE")

def study_rows(entries):
    """
    One row per study entry: value, state and the first part's mean coefficients from its run folder's result.json.

    Returns a list of dicts with value, state, cx, cy, se (None until the run has a result).
    """

    rows = []
    for entry in entries:
        row = {"value": entry["value"], "state": entry["state"], "cx": None, "cy": None, "se": None}
        result_path = Path(entry["run_folder"]) / "result.json" if entry["run_folder"] else None
        if result_path is not None and result_path.exists():
            parts = json.loads(result_path.read_text()).get("parts", {})
            if parts:
                statistics = next(iter(parts.values()))
                row["cx"] = statistics["x"]["mean"]
                row["cy"] = statistics["y"]["mean"]
                row["se"] = statistics["y"]["se"]

        rows.append(row)

    return rows

class StudyView(QWidget):
    """
    Table above a plot of the force coefficients against the swept value.
    """

    def __init__(self, parent=None):
        """
        Empty table and plot.
        """

        super().__init__(parent)
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(320)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.plot = pyqtgraph.PlotWidget(background=theme.panel)
        for side in ("left", "bottom"):
            self.plot.getAxis(side).enableAutoSIPrefix(False)
            self.plot.getAxis(side).setPen(theme.grid)
            self.plot.getAxis(side).setTextPen(theme.muted)

        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Coefficient")
        self.plot.setMinimumHeight(250)
        self.legend = self.plot.addLegend()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(self.plot)
        QVBoxLayout(self).addWidget(splitter)

    def show_rows(self, rows, parameter_label):
        """
        Rebuild the table and redraw the curves (finished runs only) for these rows.
        """

        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            cells = [f"{row['value']:g}", row["state"]] + ["" if row[key] is None else f"{row[key]:.4f}" for key in ("cx", "cy", "se")]
            for column, text in enumerate(cells):
                self.table.setItem(index, column, QTableWidgetItem(text))

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.plot.clear()
        self.legend.clear()
        self.plot.setLabel("bottom", parameter_label or "Value")
        done = sorted((row for row in rows if row["cx"] is not None), key=lambda row: row["value"])
        if not done:
            return

        values = np.array([row["value"] for row in done])
        for key, name, color in (("cx", "C_x", theme.curve_colors[0]), ("cy", "C_y", theme.curve_colors[1])):
            coefficients = np.array([row[key] for row in done])
            self.plot.plot(values, coefficients, name=name, pen=pyqtgraph.mkPen(color, width=2), symbol="o", symbolBrush=color)

        errors = np.array([row["se"] for row in done])
        self.plot.addItem(pyqtgraph.ErrorBarItem(x=values, y=np.array([row["cy"] for row in done]), height=2 * errors,
                                                 pen=pyqtgraph.mkPen(theme.curve_colors[1])))
