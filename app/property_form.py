# property form: one editor per dataclass field, writing accepted edits straight back into the object
import dataclasses
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QWidget
from app import theme

invalid_style = f"border: 1px solid {theme.error}"

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

    def __init__(self, target, choices, field_names=None, parent=None):
        """
        Build one row per field: drop-down for fields with fixed choices, check box for booleans, text otherwise.
        """

        super().__init__(parent)
        self.target = target
        self.editors = {}
        layout = QFormLayout(self)
        for item in dataclasses.fields(target):
            if field_names is not None and item.name not in field_names:
                continue

            editor = self.editor_for(item, choices)
            self.editors[item.name] = editor
            layout.addRow(item.name.replace("_", " "), editor)

    def editor_for(self, item, choices):
        """
        Make and connect the editor for one field.

        Returns the editor widget.
        """

        name = item.name
        value = getattr(self.target, name)
        if name in choices:
            editor = QComboBox()
            editor.addItems(choices[name])
            editor.setCurrentText(str(value))
            editor.currentTextChanged.connect(lambda text, name=name: self.write(name, text))
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
        Parse a finished text edit; write it if it fits the type, else mark the editor and keep the old value.
        """

        try:
            value = parse_value(editor.text(), annotation)
        
        except ValueError:
            editor.setStyleSheet(invalid_style)
            editor.setToolTip(f"not a valid {annotation}")
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