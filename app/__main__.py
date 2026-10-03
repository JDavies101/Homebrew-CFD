# desktop app entry: python -m app [case.json]
import sys
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap, QColor, QIcon
from PySide6.QtCore import Qt
from pathlib import Path
from src import __version__

SPLASH_WIDTH = 720

def show_splash(application):
    """
    Show the splash screen at SPLASH_WIDTH logical pixels with the version and a loading message.

    Returns the splash screen.
    """

    ratio = application.primaryScreen().devicePixelRatio()
    pixmap = QPixmap(str(Path(__file__).resolve().parent / "splash.png"))
    pixmap = pixmap.scaledToWidth(int(SPLASH_WIDTH * ratio),
                                  Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)

    # transparent rounded corners
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    splash.show()
    splash.showMessage(f"   Version {__version__}   ·   Loading\n", Qt.AlignmentFlag.AlignBottom |
                       Qt.AlignmentFlag.AlignLeft, QColor(150, 160, 178))
    application.processEvents()

    return splash

def main():
    """
    Start the application; with --solve as the first argument, 
    run the solver instead (the packaged app's solver entry).
    """

    if len(sys.argv) > 1 and sys.argv[1] == "--solve":
        from src.run.__main__ import main as solve

        sys.argv = [sys.argv[0]] + sys.argv[2:]
        solve()
        return

    application = QApplication(sys.argv)
    application.setApplicationName("Homebrew CFD")
    application.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "icon.png")))
    splash = show_splash(application)

    # heavy import (widgets, PyVista, case code) after the splash is up
    from app.main_window import MainWindow
    window = MainWindow()
    if len(sys.argv) > 1:
        window.open_path(sys.argv[1])

    window.show()
    splash.finish(window)
    sys.exit(application.exec())

if __name__ == "__main__":
    main()