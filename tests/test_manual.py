# manual: every page listed in order with its heading, the window renders a page, pages name only real features
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no window system needed; must precede the Qt import
import re
from PySide6.QtWidgets import QApplication
from app.manual import manual_pages, manual_directory, ManualWindow
from src.run.case_file import geometry_choices
from src.run.case import setting_choices

application = QApplication.instance() or QApplication([])

# test 1: the pages are listed in number order, each titled by its first heading
def test_pages_in_order():

    pages = manual_pages()
    titles = [title for title, _ in pages]

    assert len(pages) >= 5
    assert titles[0] == "Introduction"
    assert [path.name[:2] for _, path in pages] == sorted(path.name[:2] for _, path in pages)

# test 2: the window lists every page and renders the first
def test_window_renders():

    window = ManualWindow()

    assert window.page_list.count() == len(manual_pages())
    assert "Homebrew CFD" in window.browser.toPlainText()

# test 3: the reference names every geometry kind and every choice of the solver settings it documents
def test_reference_covers_choices():

    reference = (manual_directory / "05_reference.md").read_text(encoding="utf-8")
    documented = ["collision", "inlet", "sgs", "x_boundary", "y_boundary", "side_walls", "layer_kind"]
    missing = [kind for kind in geometry_choices["kind"] if f"| {kind} |" not in reference]
    missing += [value for name in documented for value in setting_choices[name] if not re.search(rf"\b{value}\b", reference)]

    assert missing == []