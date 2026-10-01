# monitor plot: live force-coefficient histories from the running solver
import pyqtgraph
from PySide6.QtWidgets import QVBoxLayout, QWidget

class MonitorPlot(QWidget):
    """
    One curve per part and force axis (x and y; z is ~0 in the current cases and is left out to keep the plot readable).
    """

    def __init__(self, parent=None):
        """
        Empty plot with grid, legend and axis labels.
        """

        super().__init__(parent)
        self.plot = pyqtgraph.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Sample (after warmup)")
        self.plot.setLabel("left", "Force coefficient")
        self.legend = self.plot.addLegend()
        self.curves = {}
        QVBoxLayout(self).addWidget(self.plot)

    def clear(self):
        """
        Remove all curves before a new run.
        """

        self.plot.clear()
        self.legend.clear()
        self.curves = {}

    def update_samples(self, samples):
        """
        Redraw every curve from the full sample array (cheap: a few thousand points).
        """

        header, values = samples
        for column, name in enumerate(header):
            if name.endswith("_z"):
                continue
            if name not in self.curves:
                part_name, axis = name.rsplit("_", 1)
                pen = pyqtgraph.mkPen(pyqtgraph.intColor(len(self.curves), hues=6), width=2)
                self.curves[name] = self.plot.plot(name=f"{part_name} C_{axis}", pen=pen)
            self.curves[name].setData(values[:, column])