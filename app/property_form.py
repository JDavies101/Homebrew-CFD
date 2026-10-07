# property form: one editor per dataclass field, writing accepted edits straight back into the object
import dataclasses
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QWidget
from app import theme

invalid_style = f"border: 1px solid {theme.error}"

# geometry part fields: the common ones always, the rest by kind
part_common_fields = ("name", "kind", "reference_area", "wall")
part_kind_fields = {"stl": ("path", "cells_per_unit", "offset"),
                    "naca": ("section", "chord", "angle_degrees", "leading_edge"),
                    "sphere": ("center", "radius"),
                    "cylinder": ("center", "radius", "spin_ratio"),
                    "ahmed": ("x_start", "body_height", "slant_angle", "nose")}

def part_field_names(kind):
    """
    The geometry fields a part of this kind uses: the common fields, then the kind's own.

    Returns a tuple of field names (the common ones only for an unknown kind).
    """

    return part_common_fields + part_kind_fields.get(kind, ())

def format_value(value):
    """
    Text for a field value: lists as "a, b, c", None as empty, numbers in their shortest exact form.

    Returns the string.
    """

    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    
    return str(value)

def parse_value(text, annotation):
    """
    Turn editor text back into the field's type, read from its annotation ("float | None", "<class 'int'>", ...).

    Returns the parsed value; raises ValueError on text that does not fit the type.
    """

    text = text.strip()
    if text == "" and "None" in annotation:
        return None
    if "list" in annotation:
        return [float(item) for item in text.split(",")]
    if "float" in annotation:
        return float(text)
    if "int" in annotation:
        return int(text)
    
    return text

class PropertyForm(QWidget):
    """
    Editors for chosen fields of one dataclass instance; emits changed after every accepted edit.
    """

    changed = Signal()
    rejected = Signal(str) # why a typed edit was refused

    def __init__(self, target, choices, field_names=None, parent=None, labels=None):
        """
        Build one row per field: drop-down for fields with fixed choices, check box for booleans, text otherwise;
        labels maps field names and choice values to display text (default: the name with spaces).
        """

        super().__init__(parent)
        self.target = target
        self.labels = labels or {}
        self.choices = choices
        self.editors = {}
        self.layout_rows = QFormLayout(self)
        self.build(field_names)

    def build(self, field_names):
        """
        Replace the rows with one per chosen field (all fields when field_names is None).
        """

        while self.layout_rows.rowCount():
            self.layout_rows.removeRow(0)

        self.editors = {}
        for item in dataclasses.fields(self.target):
            if field_names is not None and item.name not in field_names:
                continue

            editor = self.editor_for(item, self.choices)
            self.editors[item.name] = editor
            self.layout_rows.addRow(self.labels.get(item.name, item.name.replace("_", " ")), editor)

    def editor_for(self, item, choices):
        """
        Make and connect the editor for one field.

        Returns the editor widget.
        """

        name = item.name
        value = getattr(self.target, name)
        if name in choices:
            editor = QComboBox()
            for choice in choices[name]:
                editor.addItem(self.labels.get(choice, choice), choice)
            editor.setCurrentIndex(editor.findData(value))
            editor.currentIndexChanged.connect(lambda index, editor=editor, name=name: self.write(name, editor.itemData(index)))
        elif isinstance(value, bool):
            editor = QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(lambda checked, name=name: self.write(name, checked))
        else:
            editor = QLineEdit(format_value(value))
            annotation = str(item.type)
            editor.editingFinished.connect(lambda editor=editor, name=name, annotation=annotation: self.commit_text(editor, name, annotation))

        return editor
    
    def commit_text(self, editor, name, annotation):
        """
        Parse a finished text edit; write it if it fits the type, else mark the editor, say why and keep the old value.
        """

        try:
            value = parse_value(editor.text(), annotation)

        except ValueError as error:
            editor.setStyleSheet(invalid_style)
            editor.setToolTip(f"not a valid {annotation}")
            label = name.replace("_", " ")
            self.rejected.emit(f"{label[:1].upper() + label[1:]} not changed: {error}")
            return
        
        editor.setStyleSheet("")
        editor.setToolTip("")
        self.write(name, value)

    def write(self, name, value):
        """
        Store the value on the target and announce the change.
        """

        setattr(self.target, name, value)
        self.changed.emit()

class PartForm(PropertyForm):
    """
    The form of one geometry part: only the fields its kind uses, rebuilt when the kind changes.
    """

    def __init__(self, target, choices, parent=None):
        """
        Rows for the part's present kind.
        """

        super().__init__(target, choices, part_field_names(target.kind), parent)

    def write(self, name, value):
        """
        Store the value; a new kind rebuilds the rows once the drop-down has finished its signal.
        An ahmed body has staircase walls only, so that kind sets them.
        """

        if name == "kind" and value == "ahmed":
            self.target.wall = "staircase"

        super().write(name, value)
        if name == "kind":
            QTimer.singleShot(0, lambda: self.build(part_field_names(self.target.kind)))