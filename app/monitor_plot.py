# monitor plot: live force-coefficient histories from the running solver
import pyqtgraph
from PySide6.QtWidgets import QVBoxLayout, QWidget
import numpy as np

class MonitorPlot(QWidget):
    """
    One curve per part and force axis (x and y; z is ~0 in the current 
    cases and is left out to keep the plot readable).
    """

    def __init__(self, parent=None):
        """
        Empty plot with grid, legend and axis labels.
        """

        super().__init__(parent)
        self.plot = pyqtgraph.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Flow-throughs")
        self.plot.setLabel("left", "Force coefficient")
        self.legend = self.plot.addLegend()
        self.curves = {}
        self.mean_curves = {}
        self.timeline = None # (warmup, flow-through, sample interval) in steps, from the solver's progress.json
        self.warmup_region = None
        QVBoxLayout(self).addWidget(self.plot)

    def clear(self):
        """
        Remove all curves before a new run.
        """

        self.plot.clear()
        self.legend.clear()
        self.curves = {}
        self.mean_curves = {}
        self.timeline = None
        self.warmup_region = None

    def set_timeline(self, steps, warmup, flow_through_steps, sample_every):
        """
        Fix the x-range to the whole run in flow-throughs and shade the warmup (set once per run).
        """

        if self.timeline is not None:
            return

        self.timeline = (warmup, flow_through_steps, sample_every)
        self.plot.setXRange(0, steps / flow_through_steps, padding=0)
        self.warmup_region = pyqtgraph.LinearRegionItem((0, warmup / flow_through_steps), 
                                                        movable=False, brush=(128, 128, 128, 40))
        self.plot.addItem(self.warmup_region)        

    def update_samples(self, samples):
        """
        Redraw every curve and its running mean from the full sample array (cheap: a few thousand points).
        """

        header, values = samples
        if self.timeline is None or len(values) == 0:
            return
        
        warmup, flow_through_steps, sample_every = self.timeline
        time = (warmup + sample_every * np.arange(len(values))) / flow_through_steps
        counts = np.arange(1, len(values) + 1)

        for column, name in enumerate(header):
            if name.endswith("_z"):
                continue
            if name not in self.curves:
                part_name, axis = name.rsplit("_", 1)
                color = pyqtgraph.intColor(len(self.curves), hues=6)
                faint = pyqtgraph.mkColor(color)
                faint.setAlpha(70)
                self.curves[name] = self.plot.plot(pen=pyqtgraph.mkPen(faint, width=1))
                self.mean_curves[name] = self.plot.plot(name=f"{part_name} C_{axis} (mean)",
                                                        pen=pyqtgraph.mkPen(color, width=3))
            self.curves[name].setData(time, values[:, column])
            self.mean_curves[name].setData(time, np.cumsum(values[:, column]) / counts)

        shown = values[:, [column for column, name in enumerate(header) if not name.endswith("_z")]]
        low, high = np.percentile(shown, [1, 99])
        padding = max(0.1 * (high - low), 1e-3)
        self.plot.setYRange(low - padding, high + padding, padding=0)