import sys

from PySide6.QtWidgets import QApplication

from geopotential.ui.main_window import GeoPotentialMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GeoPotential Mapper")
    window = GeoPotentialMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
