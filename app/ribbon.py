# ribbon: Fluent-style command bar, one tab per area, each tab a row of 
# titled groups of large buttons bound to actions
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QTabWidget, QToolButton, QVBoxLayout, QWidget

class RibbonTab(QWidget):
    """
    One ribbon page: groups left to right, separated by thin vertical lines.
    """

    def __init__(self):
        """
        Empty row; groups are inserted before the closing stretch.
        """

        super().__init__()
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(4, 2, 4, 2)
        self.row.setSpacing(4)
        self.row.addStretch()

    def add_group(self, title, actions):
        """
        A titled group of large buttons, one per action (an action with a menu opens it on click).
        """

        if self.row.count() > 1:
            separator = QFrame()
            separator.setFrameShape(QFrame.VLine)
            separator.setFrameShadow(QFrame.Sunken)
            self.row.insertWidget(self.row.count() - 1, separator)

        group = QWidget()
        column = QVBoxLayout(group)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)

        for action in actions:
            button = QToolButton()
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(28, 28))
            button.setAutoRaise(True)

            if action.menu() is not None:
                button.setPopupMode(QToolButton.InstantPopup)

            buttons.addWidget(button)

        column.addLayout(buttons)

        # group caption under the buttons, greyed
        caption = QLabel(title)
        caption.setAlignment(Qt.AlignHCenter)
        caption.setEnabled(False)
        column.addWidget(caption)
        self.row.insertWidget(self.row.count() - 1, group)

class Ribbon(QTabWidget):
    """
    The window's command bar (replaces the menu bar and toolbars).
    """

    def __init__(self):
        """
        Flat tab bar, only as tall as its buttons.
        """

        super().__init__()
        self.setDocumentMode(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def add_tab(self, name):
        """
        Append an empty tab.

        Returns the RibbonTab to fill with groups.
        """

        tab = RibbonTab()
        self.addTab(tab, name)

        return tab