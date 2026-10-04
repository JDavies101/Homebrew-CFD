# property forms: text, drop-down and optional fields write typed values back to the object
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window system needed; must precede the Qt import
from PySide6.QtWidgets import QApplication
from src.run.case import Flow, Domain, Timing
from app.property_form import PropertyForm, invalid_style

application = QApplication.instance() or QApplication([])
choices = {"floor": ("static", "moving")}

# test 1: a valid text edit writes the parsed float and announces it
def test_text_edit_writes_float():

    flow = Flow()
    form = PropertyForm(flow, choices)
    signals = []
    form.changed.connect(lambda: signals.append(True))
    form.editors["reynolds_number"].setText("10000")
    form.editors["reynolds_number"].editingFinished.emit()

    assert flow.reynolds_number == 10000.0
    assert isinstance(flow.reynolds_number, float)
    assert signals == [True]

# test 2: text that does not fit the type is refused and marked
def test_bad_text_refused():

    flow = Flow()
    form = PropertyForm(flow, choices)
    form.editors["reynolds_number"].setText("ten thousand")
    form.editors["reynolds_number"].editingFinished.emit()

    assert flow.reynolds_number == 5000.0
    assert form.editors["reynolds_number"].styleSheet() == invalid_style

# test 3: a drop-down writes the chosen string
def test_choice_writes_string():

    domain = Domain()
    form = PropertyForm(domain, choices)
    form.editors["floor"].setCurrentText("moving")

    assert domain.floor == "moving"

# test 4: an optional integer accepts a number and an empty field (None)
def test_optional_int():

    timing = Timing()
    form = PropertyForm(timing, choices)
    form.editors["steps_override"].setText("60000")
    form.editors["steps_override"].editingFinished.emit()
    steps_after_number = timing.steps_override
    form.editors["steps_override"].setText("")
    form.editors["steps_override"].editingFinished.emit()

    assert steps_after_number == 60000
    assert timing.steps_override is None

# test 5: field_names limits the form to the listed fields
def test_field_subset():

    form = PropertyForm(Flow(), choices, field_names=("reynolds_number",))

    assert list(form.editors) == ["reynolds_number"]

# test 6: labelled choices show the label and write the real value
def test_labelled_choice_writes_value():

    domain = Domain()
    labels = {"floor": "Floor", "static": "Static floor", "moving": "Moving floor (road)"}
    form = PropertyForm(domain, choices, labels=labels)
    editor = form.editors["floor"]
    editor.setCurrentIndex(editor.findData("moving"))

    assert domain.floor == "moving"
    assert editor.currentText() == "Moving floor (road)"
    assert form.layout().labelForField(editor).text() == "Floor"