# desktop app entry: python -m app [case.json]
import sys
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow

def main():
    """
    Start the application, optionally opening a case file given on the command line.
    """

    application = QApplication(sys.argv)
    application.setApplicationName("Homebrew CFD")
    window = MainWindow()
    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])

    window.show()
    sys.exit(application.exec())

if __name__ == "__main__":
    main()