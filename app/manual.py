# manual: the Markdown pages of docs/manual shown in a window, page list on the left
import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QTextBrowser

# source build: the repository; packaged app: beside the executable's bundled files
bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
manual_directory = bundle_root / "docs" / "manual"

def manual_pages(directory=manual_directory):
    """
    The manual's pages in reading order (file names start with their number).

    Returns a list of (title, path), the title taken from the page's first heading.
    """

    pages = []
    for path in sorted(Path(directory).glob("*.md")):
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        pages.append((first_line.lstrip("# ").strip(), path))

    return pages

class ManualWindow(QDialog):
    """
    Non-modal window: page titles on the left, the selected page rendered on the right.
    """

    def __init__(self, parent=None, directory=manual_directory):
        """
        List the pages and show the first.
        """

        super().__init__(parent)
        self.setWindowTitle("Homebrew CFD Manual")
        self.resize(980, 720)
        self.page_list = QListWidget()
        self.page_list.setMaximumWidth(240)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout = QHBoxLayout(self)
        layout.addWidget(self.page_list)
        layout.addWidget(self.browser)

        for title, path in manual_pages(directory):
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, str(path))
            self.page_list.addItem(item)

        self.page_list.currentItemChanged.connect(self.show_page)
        self.page_list.setCurrentRow(0)

    def show_page(self, item, previous):
        """
        Render the selected page.
        """

        if item is not None:
            self.browser.setMarkdown(Path(item.data(Qt.UserRole)).read_text(encoding="utf-8"))