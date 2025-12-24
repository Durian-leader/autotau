"""
AutoTau GUI application entry point.

Usage:
    python -m autotau.gui
    # or after installation:
    autotau-gui
"""

import sys


def main():
    """Main entry point for the AutoTau GUI application."""
    # Import here to avoid loading PyQt5 when not needed
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AutoTau")
    app.setApplicationVersion("0.5.0")
    app.setOrganizationName("AutoTau")

    # Set style
    app.setStyle('Fusion')

    # Optional: Apply dark theme
    # from PyQt5.QtGui import QPalette, QColor
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # palette.setColor(QPalette.WindowText, Qt.white)
    # palette.setColor(QPalette.Base, QColor(25, 25, 25))
    # palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    # palette.setColor(QPalette.ToolTipBase, Qt.white)
    # palette.setColor(QPalette.ToolTipText, Qt.white)
    # palette.setColor(QPalette.Text, Qt.white)
    # palette.setColor(QPalette.Button, QColor(53, 53, 53))
    # palette.setColor(QPalette.ButtonText, Qt.white)
    # palette.setColor(QPalette.BrightText, Qt.red)
    # palette.setColor(QPalette.Link, QColor(42, 130, 218))
    # palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    # palette.setColor(QPalette.HighlightedText, Qt.black)
    # app.setPalette(palette)

    from .main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
