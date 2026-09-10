import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geopotential.processing.gridding import grid_scattered
from geopotential.visualization.palettes import PALETTES, get_palette


class MapCanvas(FigureCanvas):
    def __init__(self):
        self.figure, self.ax = plt.subplots(figsize=(9, 7))
        super().__init__(self.figure)
        self.colorbar = None

    def plot_grid(self, work, x_col, y_col, value_col, xx, yy, zz, palette_name, reverse=False):
        self.ax.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

        colors = [color for _, color in get_palette(palette_name, reverse)]
        cmap = LinearSegmentedColormap.from_list(palette_name, colors, N=256)
        finite = np.asarray(zz)[np.isfinite(zz)]
        if finite.size == 0:
            raise ValueError('The interpolation produced no finite grid values.')

        levels = np.linspace(float(np.nanmin(finite)), float(np.nanmax(finite)), 30)
        if np.allclose(levels[0], levels[-1]):
            levels = 20

        contour = self.ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
        self.ax.contour(xx, yy, zz, levels=levels, colors='black', linewidths=0.2, alpha=0.35)
        self.ax.scatter(work[x_col], work[y_col], s=10, c='black', marker='o', label='Stations')
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_title(value_col)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend(loc='upper right')
        self.colorbar = self.figure.colorbar(contour, ax=self.ax, pad=0.02)
        self.colorbar.set_label(value_col)
        self.figure.tight_layout()
        self.draw()


class GeoPotentialDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoPotential Mapper')
        self.resize(1450, 900)
        self.df = None
        self.current_grid = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(320)
        controls.setMaximumWidth(390)
        form = QFormLayout(controls)

        self.open_button = QPushButton('Open survey file')
        self.open_button.clicked.connect(self.open_file)
        form.addRow(self.open_button)

        self.file_label = QLabel('No file loaded')
        self.file_label.setWordWrap(True)
        form.addRow('File', self.file_label)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.value_combo = QComboBox()
        form.addRow('X / Easting', self.x_combo)
        form.addRow('Y / Northing', self.y_combo)
        form.addRow('Value', self.value_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            'Linear',
            'Nearest',
            'Cubic',
            'Ordinary Kriging',
            'Minimum Curvature',
        ])
        form.addRow('Gridding', self.method_combo)

        self.variogram_combo = QComboBox()
        self.variogram_combo.addItems(['Spherical', 'Exponential', 'Gaussian', 'Linear', 'Power'])
        form.addRow('Kriging variogram', self.variogram_combo)

        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(30, 600)
        self.nx_spin.setValue(180)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(30, 600)
        self.ny_spin.setValue(180)
        form.addRow('Grid cells X', self.nx_spin)
        form.addRow('Grid cells Y', self.ny_spin)

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.reverse_check = QCheckBox('Reverse palette')
        form.addRow('Color palette', self.palette_combo)
        form.addRow('', self.reverse_check)

        self.grid_button = QPushButton('Generate map')
        self.grid_button.clicked.connect(self.generate_map)
        form.addRow(self.grid_button)

        self.export_button = QPushButton('Export grid CSV')
        self.export_button.clicked.connect(self.export_grid)
        form.addRow(self.export_button)

        self.status_label = QLabel('Ready')
        self.status_label.setWordWrap(True)
        form.addRow('Status', self.status_label)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = MapCanvas()
        self.table = QTableWidget()
        self.table.setMaximumHeight(220)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.table)

        splitter.addWidget(controls)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open survey data',
            '',
            'Survey files (*.csv *.xyz *.txt *.dat);;All files (*.*)',
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path, sep=None, engine='python')
        except Exception as exc:
            QMessageBox.critical(self, 'Import error', str(exc))
            return

        self.file_label.setText(path)
        columns = [str(c) for c in self.df.columns]
        for combo in (self.x_combo, self.y_combo, self.value_combo):
            combo.clear()
            combo.addItems(columns)

        lowered = {str(c).lower(): str(c) for c in self.df.columns}
        self._select_guess(self.x_combo, lowered, ['x', 'easting', 'east', 'utm_x'])
        self._select_guess(self.y_combo, lowered, ['y', 'northing', 'north', 'utm_y'])
        self._select_guess(self.value_combo, lowered, ['bouguer', 'gravity', 'cba', 'tmi', 'magnetic', 'magnetics'])
        self.populate_table()
        self.status_label.setText(f'{len(self.df)} rows loaded')

    @staticmethod
    def _select_guess(combo, lowered, candidates):
        for key in candidates:
            if key in lowered:
                combo.setCurrentText(lowered[key])
                return

    def populate_table(self):
        preview = self.df.head(20)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def generate_map(self):
        if self.df is None:
            QMessageBox.information(self, 'No data', 'Load a survey file first.')
            return

        method = self.method_combo.currentText()
        self.status_label.setText(f'Running {method}...')
        QApplication.processEvents()
        try:
            work, xx, yy, zz, variance = grid_scattered(
                self.df,
                self.x_combo.currentText(),
                self.y_combo.currentText(),
                self.value_combo.currentText(),
                method=method,
                nx=self.nx_spin.value(),
                ny=self.ny_spin.value(),
                variogram_model=self.variogram_combo.currentText().lower(),
            )
            self.canvas.plot_grid(
                work,
                self.x_combo.currentText(),
                self.y_combo.currentText(),
                self.value_combo.currentText(),
                xx,
                yy,
                zz,
                self.palette_combo.currentText(),
                reverse=self.reverse_check.isChecked(),
            )
            self.current_grid = (xx, yy, zz)
            note = ' Kriging variance was also calculated.' if variance is not None else ''
            self.status_label.setText(f'{method} complete.{note}')
        except Exception as exc:
            self.status_label.setText('Gridding failed')
            QMessageBox.critical(self, 'Gridding error', str(exc))

    def export_grid(self):
        if self.current_grid is None:
            QMessageBox.information(self, 'No grid', 'Generate a grid first.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export grid', 'grid.csv', 'CSV files (*.csv)')
        if not path:
            return
        xx, yy, zz = self.current_grid
        pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'value': np.asarray(zz).ravel()}).dropna().to_csv(path, index=False)
        self.status_label.setText(f'Grid exported to {path}')


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('GeoPotential Mapper')
    window = GeoPotentialDesktop()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
