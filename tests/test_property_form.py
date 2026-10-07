# property forms: text, drop-down and optional fields write typed values back to the object
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window system needed; must precede the Qt import
from PySide6.QtWidgets import QApplication
import dataclasses
from src.run.case import Flow, Domain, Timing
from src.run.case_file import GeometrySpec
from app.property_form import PropertyForm, PartForm, invalid_style, part_common_fields, part_kind_fields, part_field_names

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

# test 7: every geometry field is common or used by at least one kind
def test_part_fields_all_covered():

    used = set(part_common_fields)
    for names in part_kind_fields.values():
        used.update(names)

    assert {item.name for item in dataclasses.fields(GeometrySpec)} == used

# test 8: a naca part lists neither the radius nor the nose
def test_naca_fields_exclude_other_kinds():

    names = part_field_names("naca")

    assert "radius" not in names
    assert "nose" not in names
    assert "chord" in names
    assert names[:4] == part_common_fields

# test 9: a part form shows only the fields of the part's kind
def test_part_form_rows_follow_kind():

    spec = GeometrySpec(name="ball", kind="sphere", reference_area=300.0)
    form = PartForm(spec, choices)

    assert set(form.editors) == set(part_field_names("sphere"))
    assert "chord" not in form.editors

# test 10: switching a part to ahmed sets staircase walls, switching to another kind leaves them
def test_ahmed_kind_sets_staircase_walls():

    part_choices = {"kind": ("sphere", "ahmed"), "wall": ("bouzidi", "staircase")}
    ahmed_spec = GeometrySpec(name="body", kind="sphere", reference_area=300.0)
    ahmed_form = PartForm(ahmed_spec, part_choices)
    signals = []
    ahmed_form.changed.connect(lambda: signals.append(True))
    ahmed_form.editors["kind"].setCurrentIndex(ahmed_form.editors["kind"].findData("ahmed"))
    sphere_spec = GeometrySpec(name="ball", kind="ahmed", reference_area=300.0, wall="bouzidi")
    sphere_form = PartForm(sphere_spec, part_choices)
    sphere_form.editors["kind"].setCurrentIndex(sphere_form.editors["kind"].findData("sphere"))

    assert ahmed_spec.kind == "ahmed"
    assert ahmed_spec.wall == "staircase"
    assert signals == [True]
    assert sphere_spec.kind == "sphere"
    assert sphere_spec.wall == "bouzidi"

# test 11: a refused text edit announces why, naming the field
def test_bad_text_announces_reason():

    flow = Flow()
    form = PropertyForm(flow, choices)
    reasons = []
    form.rejected.connect(reasons.append)
    form.editors["reynolds_number"].setText("ten thousand")
    form.editors["reynolds_number"].editingFinished.emit()

    assert len(reasons) == 1
    assert reasons[0].startswith("Reynolds number not changed")
    assert "ten thousand" in reasons[0]