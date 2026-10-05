# study page: full-window COMSOL-style start page, problem/study/flow/geometry steps, live preview, recent cases
from pathlib import Path
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtWidgets import (QWidget, QSplitter, QListWidget, QListWidgetItem, QLabel, QVBoxLayout, QHBoxLayout,
                               QStackedWidget, QScrollArea, QToolButton, QButtonGroup, QPushButton, QFileDialog, QSizePolicy)
from src.run.case import setting_choices
from src.run.case_file import part_free_case
from src.run.study import StudyAnswers, study_choices, physical_reynolds_number, case_from_answers, add_stl_part
from app.property_form import PropertyForm
from app.viewport import CaseViewport, read_part_mesh, estimate_lines
from app import theme

wizard_choices = {**study_choices, "sgs": setting_choices["sgs"]}

# what the user reads: field names and choice values
wizard_labels = {
    "problem": "Problem",
    "free_stream": "Free stream (body in open air)",
    "ground_static": "Ground effect, static floor",
    "ground_moving": "Ground effect, moving floor (road)",
    "wheel": "Rotating wheel on moving ground",
    "thin_span": "Thin span (quasi-2D)",
    "name": "Case name",
    "study": "Study",
    "averaged": "Averaged forces (steady)",
    "unsteady": "Unsteady (vortex shedding)",
    "speed": "Speed (m/s)",
    "body_length": "Body length (m)",
    "fluid": "Fluid",
    "air": "Air, 20 °C",
    "water": "Water, 20 °C",
    "resolution": "Resolution",
    "coarse": "Coarse (40 cells per body length)",
    "medium": "Medium (64 cells per body length)",
    "fine": "Fine (96 cells per body length)",
    "lattice_velocity": "Lattice velocity U",
    "reynolds_number": "Reynolds number (blank: from the flow)",
    "length_multiple": "Domain length (body lengths)",
    "height_multiple": "Domain height (body lengths, blank: automatic)",
    "width_multiple": "Domain width (body lengths)",
    "sgs": "Turbulence model",
    "none": "None",
    "wale": "WALE",
    "smagorinsky": "Smagorinsky",
    "allow_below_floor": "Keep the physical Re below the tau floor (LES carries it)",
}

# short card titles, shown above the one-line description
card_titles = {
    "free_stream": "Free stream",
    "ground_static": "Ground effect, static floor",
    "ground_moving": "Ground effect, moving floor",
    "wheel": "Rotating wheel",
    "averaged": "Averaged forces",
    "unsteady": "Unsteady",
}

# one-line description under each card title
card_descriptions = {
    "free_stream": "Body in open air, far from walls",
    "ground_static": "Wind-tunnel floor that does not move",
    "ground_moving": "Road moving under a car or wing",
    "wheel": "Spinning wheel on a moving road",
    "averaged": "Mean forces after the flow settles",
    "unsteady": "Time history for vortex shedding and Strouhal",
}

# step titles and subtitles, in rail order
step_titles = (
    ("Problem", "What are you simulating?"),
    ("Study", "What do you want from the run?"),
    ("Flow", "The real flow; the grid and lattice numbers follow from it."),
    ("Geometry", "Optional: an STL now, or parts later from Home > Geometry."),
)

# fields shown as cards (problem, study) and as property forms (flow), for building the steps and the label test
card_fields = ("problem", "study")
problem_optional_fields = ("thin_span", "name")
flow_required_fields = ("speed", "body_length", "fluid", "resolution")
flow_optional_fields = ("lattice_velocity", "reynolds_number", "length_multiple", "height_multiple", "width_multiple",
                        "sgs", "allow_below_floor")

class StudyPage(QWidget):
    """
    Full-window start page: step rail and recent cases on the left, the current step in the middle,
    a live preview and verdict on the right; Create on the last step builds the case.
    """

    created = Signal(object)
    open_requested = Signal(str)
    blank_requested = Signal()
    template_requested = Signal(str)

    def __init__(self, stl_cache, start_directory_function, recent_paths_function, estimate_function, preview=None, parent=None):
        """
        Build the rail, the step stack, the bottom bar and the preview; start on the first step.
        estimate_function returns (throughput in MLUPS, measured, total GPU memory in GB or None).
        """

        super().__init__(parent)
        self.stl_cache = stl_cache
        self.start_directory_function = start_directory_function
        self.recent_paths_function = recent_paths_function
        self.estimate_function = estimate_function
        self.answers = StudyAnswers()
        self.stl_path = None
        self.stl_inspection = None
        self.error = None
        self.max_visited = 0
        self.first_draw = True
        self.preview = preview if preview is not None else CaseViewport(self)

        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setAlignment(Qt.AlignTop)

        # left: step rail, then recent cases
        self.rail = QListWidget()
        self.rail.currentRowChanged.connect(self.on_rail_row)
        self.recent_list = QListWidget()
        self.recent_list.itemActivated.connect(self.on_recent_activated)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.rail)
        left_layout.addWidget(QLabel("Recent cases"))
        left_layout.addWidget(self.recent_list)

        # middle: heading, subtitle, the current step scrolled so More options never squeezes rows
        self.heading = QLabel()
        self.heading.setFont(theme.heading_font(16))
        self.subtitle = QLabel()
        self.subtitle.setWordWrap(True)
        self.step_stack = QStackedWidget()
        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.addWidget(self.heading)
        middle_layout.addWidget(self.subtitle)
        middle_layout.addWidget(self.step_stack)

        # right: live preview above the verdict
        right = QSplitter(Qt.Vertical)
        if hasattr(self.preview, "plotter"):
            right.addWidget(self.preview.plotter)
        right.addWidget(self.verdict)
        right.setSizes([560, 160])

        content = QSplitter()
        content.addWidget(left)
        content.addWidget(middle)
        content.addWidget(right)
        content.setSizes([220, 460, 400])

        # bottom bar: open or start blank on the left, step navigation on the right
        self.open_button = QPushButton("Open case...")
        self.open_button.clicked.connect(lambda: self.open_requested.emit(""))
        self.blank_button = QPushButton("Blank case")
        self.blank_button.clicked.connect(self.blank_requested.emit)
        self.back_button = QPushButton("< Back")
        self.back_button.clicked.connect(self.go_back)
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(self.go_next)
        bottom = QHBoxLayout()
        bottom.addWidget(self.open_button)
        bottom.addWidget(self.blank_button)
        bottom.addStretch()
        bottom.addWidget(self.back_button)
        bottom.addWidget(self.next_button)

        layout = QVBoxLayout(self)
        layout.addWidget(content)
        layout.addLayout(bottom)

        self.build_steps()
        self.set_recent(self.recent_paths_function())

        self.template_list = QListWidget()
        self.template_list.itemActivated.connect(lambda item: self.template_requested.emit(item.data(Qt.UserRole)))
        left_layout.addWidget(QLabel("Templates"))
        left_layout.addWidget(self.template_list)

        self.go_to(0)
        self.refresh()

    def make_cards(self, field_name, values):
        """
        A column of checkable cards, one per value, writing the choice into the answers on click.

        Returns the QWidget holding them.
        """

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QButtonGroup(container)
        group.setExclusive(True)
        current = getattr(self.answers, field_name)
        for value in values:
            button = QToolButton()
            button.setObjectName("card")
            button.setCheckable(True)
            button.setChecked(value == current)
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setIconSize(QSize(40, 40))
            button.setIcon(theme.icon(value))
            button.setText(f"{card_titles[value]}\n{card_descriptions[value]}")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            button.clicked.connect(lambda checked=False, field_name=field_name, value=value: self.choose(field_name, value))
            group.addButton(button)
            layout.addWidget(button)
        layout.addStretch()

        return container

    def choose(self, field_name, value):
        """
        Write a card's value into the answers and refresh the preview and verdict.
        """

        setattr(self.answers, field_name, value)
        self.refresh()

    def add_more_options(self, layout, fields):
        """
        A folded "More options" form beneath the required fields, opened on click; does nothing when fields is empty.
        """

        if not fields:
            return

        more_button = QToolButton()
        more_button.setText("More options")
        more_button.setCheckable(True)
        more_button.setArrowType(Qt.RightArrow)
        more_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        more_form = PropertyForm(self.answers, wizard_choices, fields, labels=wizard_labels)
        more_form.changed.connect(self.refresh)
        more_form.setVisible(False)
        more_button.toggled.connect(more_form.setVisible)
        more_button.toggled.connect(lambda opened, more_button=more_button: more_button.setArrowType(Qt.DownArrow if opened else Qt.RightArrow))
        layout.addWidget(more_button)
        layout.addWidget(more_form)

    def build_step_problem(self):
        """
        Problem cards, then the optional thin-span and case-name fields.

        Returns the step's QWidget.
        """

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.make_cards("problem", study_choices["problem"]))
        self.add_more_options(layout, problem_optional_fields)
        layout.addStretch()

        return widget

    def build_step_study(self):
        """
        Study cards, no more options.

        Returns the step's QWidget.
        """

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.make_cards("study", study_choices["study"]))
        layout.addStretch()

        return widget

    def build_step_flow(self):
        """
        The required flow fields, then the optional lattice and domain-sizing fields.

        Returns the step's QWidget.
        """

        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = PropertyForm(self.answers, wizard_choices, flow_required_fields, labels=wizard_labels)
        form.changed.connect(self.refresh)
        layout.addWidget(form)
        self.add_more_options(layout, flow_optional_fields)
        layout.addStretch()

        return widget

    def build_step_geometry(self):
        """
        An optional STL import, with a report on what was read and a way to remove it.

        Returns the step's QWidget.
        """

        widget = QWidget()
        layout = QVBoxLayout(widget)
        buttons = QHBoxLayout()
        import_button = QPushButton("Import STL...")
        import_button.clicked.connect(self.choose_stl)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.remove_stl)
        buttons.addWidget(import_button)
        buttons.addWidget(remove_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.geometry_report = QLabel()
        self.geometry_report.setWordWrap(True)
        layout.addWidget(self.geometry_report)
        layout.addStretch()

        return widget

    def build_steps(self):
        """
        Build every step's widget inside a scroll area and refresh the rail.
        """

        while self.step_stack.count():
            widget = self.step_stack.widget(0)
            self.step_stack.removeWidget(widget)
            widget.deleteLater()

        for builder in (self.build_step_problem, self.build_step_study, self.build_step_flow, self.build_step_geometry):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(builder())
            self.step_stack.addWidget(scroll)

        self.update_rail()

    def update_rail(self):
        """
        Rail labels: numbered step names, a check mark while the case validates, disabled past the last visited step.
        """

        current = self.step_stack.currentIndex()
        self.rail.blockSignals(True)
        self.rail.clear()
        for index, (title, _) in enumerate(step_titles):
            mark = " ✓" if self.error is None and index <= self.max_visited else ""
            item = QListWidgetItem(f"{index + 1} {title}{mark}")
            if index > self.max_visited:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.rail.addItem(item)
        self.rail.setCurrentRow(current)
        self.rail.blockSignals(False)

    def on_rail_row(self, row):
        """
        Follow a rail click to a visited step; revert the selection when the step is not open yet.
        """

        if row < 0:
            return
        if row > self.max_visited:
            self.rail.blockSignals(True)
            self.rail.setCurrentRow(self.step_stack.currentIndex())
            self.rail.blockSignals(False)
            return
        self.go_to(row)

    def go_to(self, index):
        """
        Show one step: heading, subtitle, rail selection and the Back / Next buttons.
        """

        self.step_stack.setCurrentIndex(index)
        title, subtitle = step_titles[index]
        self.heading.setText(title)
        self.subtitle.setText(subtitle)
        self.rail.setCurrentRow(index)
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("Create" if index == len(step_titles) - 1 else "Next >")
        self.next_button.setEnabled(self.error is None)

    def go_back(self):
        """
        Step back one, if not already on the first step.
        """

        index = self.step_stack.currentIndex()
        if index > 0:
            self.go_to(index - 1)

    def go_next(self):
        """
        Step forward one, or emit the finished case on the last step; does nothing while the case has an error.
        """

        if self.error is not None:
            return

        index = self.step_stack.currentIndex()
        if index == len(step_titles) - 1:
            self.created.emit(self.case_file())
            return

        self.max_visited = max(self.max_visited, index + 1)
        self.update_rail()
        self.go_to(index + 1)

    def refresh(self):
        """
        Rebuild the case from the answers, show why it cannot run or what it runs at, and redraw the preview.
        """

        try:
            case = part_free_case(case_from_answers(self.answers))
            case.validate()
        except (ValueError, ZeroDivisionError) as error:
            self.error = str(error)
            self.verdict.setText(f"<span style='color:{theme.error}'>{self.error[:1].upper() + self.error[1:]}</span>")
            self.next_button.setEnabled(False)
            self.update_rail()
            return

        self.error = None
        flow = case.flow
        domain = case.domain
        physical = physical_reynolds_number(self.answers)
        resolved_note = " (capped where tau reaches its floor)" if flow.reynolds_number < physical else ""
        lines = [f"<span style='color:{theme.ok}'>Ready</span>",
                 f"physical Re = {physical:,.0f}",
                 f"resolved Re = {flow.reynolds_number:,.0f}{resolved_note}",
                 f"tau = {flow.relaxation_time:.5f}",
                 f"Mach = {flow.free_stream_velocity * 3 ** 0.5:.3f}",
                 f"grid = {domain.nx} x {domain.ny} x {domain.nz} ({domain.nx * domain.ny * domain.nz:,} cells)"]
        lines += estimate_lines(self.case_file(), *self.estimate_function())

        # a bad STL must never break the preview or the verdict
        try:
            reports = self.preview.draw(self.case_file(), reset_camera=self.first_draw)
            self.first_draw = False
            if self.stl_path is not None and reports:
                lines += ["", "Parts", *reports]
        except (OSError, ValueError):
            pass

        self.verdict.setText("<br>".join(lines))
        self.next_button.setEnabled(True)
        self.update_rail()

    def choose_stl(self):
        """
        Pick an STL and show its inspection; it becomes the first part on Create.
        """

        path, _ = QFileDialog.getOpenFileName(self, "Import STL", str(self.start_directory_function()), "STL files (*.stl)")
        if not path:
            return
        try:
            _, inspection = read_part_mesh(path, self.stl_cache)
        except (OSError, ValueError) as error:
            self.geometry_report.setText(f"<span style='color:{theme.error}'>Cannot read {Path(path).name} ({error})</span>")
            return

        self.stl_path = path
        self.stl_inspection = inspection
        size = inspection["size"]
        closure = "watertight" if inspection["watertight"] else f"<span style='color:{theme.error}'>not watertight</span>"
        self.geometry_report.setText(f"{Path(path).name}: {size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} units, {closure}; "
                                     f"scaled so its length in x is the body length in cells")
        self.refresh()

    def remove_stl(self):
        """
        Clear the chosen STL; the case goes back to part-free.
        """

        self.stl_path = None
        self.stl_inspection = None
        self.geometry_report.setText("")
        self.refresh()

    def case_file(self):
        """
        The case the answers describe, with the chosen STL as its first part.

        Returns a CaseFile.
        """

        case_file = case_from_answers(self.answers)
        if self.stl_path is not None:
            add_stl_part(case_file, self.stl_path, self.stl_inspection, Path(self.stl_path).stem)

        return case_file

    def reset(self):
        """
        Start a fresh set of answers with no STL, rebuild the steps, and go back to the first one.
        """

        self.answers = StudyAnswers()
        self.stl_path = None
        self.stl_inspection = None
        self.error = None
        self.max_visited = 0
        self.first_draw = True
        self.build_steps()
        self.go_to(0)
        self.refresh()

    def set_recent(self, paths):
        """
        Fill the recent-cases list, file name shown and the full path as the tooltip and item data.
        """

        self.recent_list.clear()
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            self.recent_list.addItem(item)

    def set_templates(self, paths):
        """
        Fill the templates list: name shown (file stem, spaced), full path as item data.
        """

        self.template_list.clear()
        for path in paths:
            item = QListWidgetItem(Path(path).stem.replace("_", " "))
            item.setData(Qt.UserRole, str(path))
            self.template_list.addItem(item)

    def on_recent_activated(self, item):
        """
        Open the case a recent-list entry points to.
        """

        self.open_requested.emit(item.data(Qt.UserRole))

    def close_preview(self):
        """
        Release the preview's render window before Qt destroys the widget.
        """

        self.preview.close()
