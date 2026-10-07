# theme: dark navy palette from the icon, Fusion style, Saira headings; 
# every UI color comes from here
from pathlib import Path
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette, QIcon

# palette, sampled from the icon
navy = "#0f1c2c" # window background (icon background)
panel = "#13243a" # inputs, trees, console, plots
raised = "#1a2d45" # ribbon, dock titles, headers
grid = "#283e5a" # borders, separators, splitter handles (icon grid)
hover = "#22385a"
text = "#d6d9df" # wordmark silver
muted = "#96a0b2" # captions, axis labels, tagline gray
disabled = "#5c6a80"
accent = "#1e78ff" # selection, focus (streamline blue)
accent_hover = "#4a93ff"
violet = "#7d3cf0"
orange_red = "#ff551e"
amber = "#ffaa00" # Run, warnings (CFD amber)
silver = "#c4c9d2" # geometry
ok = "#4cd38a"
error = "#ff5a3c"
viewport_background = "#ffffff"
viewport_outline = grid # navy-gray domain box, visible on white
viewport_part = "#7d8696" # darker steel so parts read on white
# curve colors for monitors, in order of first appearance
curve_colors = (accent, amber, violet, orange_red, silver, "#7fb6ff")

fonts_directory = Path(__file__).resolve().parent.parent / "packaging" / "fonts"
icons_directory = Path(__file__).resolve().parent / "icons"

stylesheet = f"""
QSplitter::handle {{ background: {grid}; }}
QSplitter::handle:hover {{ background: {accent}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QTabBar::tab {{ background: {navy}; color: {muted}; padding: 6px 16px; border: none; }}
QTabBar::tab:selected {{ background: {raised}; color: {text}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QToolButton {{ border: 1px solid transparent; border-radius: 4px; padding: 2px; }}
QToolButton:hover {{ background: {hover}; border-color: {grid}; }}
QToolButton:pressed {{ background: {grid}; }}
QDockWidget::title {{ background: {raised}; padding: 4px 8px; }}
QToolTip {{ background: {raised}; color: {text}; border: 1px solid {grid}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ border: 1px solid {grid}; border-radius: 3px; padding: 2px 4px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {accent}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px; border-left: 1px solid {grid}; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; border-left: 1px solid {grid}; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {hover}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{icons_directory.as_posix()}/spin_up.svg"); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{icons_directory.as_posix()}/spin_down.svg"); width: 8px; height: 8px; }}
QToolButton#card {{ text-align: left; padding: 8px; border: 1px solid {grid}; border-radius: 6px; }}
QToolButton#card:checked {{ border-color: {accent}; background: {hover}; }}
"""

def heading_font(point_size):
    """
    Saira semi-expanded italic SemiBold for tab names and headings.

    Returns the QFont.
    """

    font = QFont("Saira", point_size, QFont.Weight.DemiBold, True)
    font.setVariableAxis(QFont.Tag("wdth"), 112.5)

    return font

def icon(name):
    """
    A ribbon icon from app/icons.

    Returns the QIcon.
    """

    return QIcon(str(icons_directory / f"{name}.svg"))

def apply_theme(application):
    """
    Fusion style, dark navy palette, the stylesheet, and the Saira fonts registered.
    """

    application.setStyle("Fusion")

    # register Saira (roman and italic variable fonts)
    for font_file in fonts_directory.glob("Saira*.ttf"):
        QFontDatabase.addApplicationFont(str(font_file))

    palette = QPalette()
    roles = {QPalette.Window: navy, QPalette.WindowText: text, QPalette.Base: panel, QPalette.AlternateBase: raised,
             QPalette.Text: text, QPalette.Button: raised, QPalette.ButtonText: text, QPalette.ToolTipBase: raised,
             QPalette.ToolTipText: text, QPalette.Highlight: accent, QPalette.HighlightedText: "#ffffff",
             QPalette.Link: accent_hover, QPalette.PlaceholderText: disabled, QPalette.BrightText: amber}
    
    for role, color in roles.items():
        palette.setColor(role, QColor(color))

    # disabled widgets (frozen inputs during a run, grayed group captions)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(disabled))

    application.setPalette(palette)
    application.setStyleSheet(stylesheet)