# desktop app entry: python -m app [case.json]
import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow
from pathlib import Path
from PySide6.QtGui import QIcon

def main():
    """
    Start the application; with --solve as the first argument, run the solver instead (the packaged app's solver entry).
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--solve":
        from src.run.__main__ import main as solve

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        solve()
        return

    application = QApplication(sys.argv)
    application.setApplicationName("Homebrew CFD")
    application.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "icon.png")))
    window = MainWindow()
    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])

    window.show()
    sys.exit(application.exec())

if __name__ == "__main__":
    main()