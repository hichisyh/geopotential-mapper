import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
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


class ColorButton(QPushButton):
    def __init__(self, text, color):
        super().__init__(text)
        self.color = color
        self.clicked.connect(self.choose_color)
        self.refresh()

    def choose_color(self):
        selected = QColorDialog.getColor(QColor(self.color), self, f'Choose {self.text()}')
        if selected.isValid():
            self.color = selected.name()
            self.refresh()

    def refresh(self):
        self.setStyleSheet(f'background-color: {self.color}; color: black; font-weight: 600;')


class MapCanvas(FigureCanvas):
    def __init__(self):
        self.figure, self.ax = plt.subplots(figsize=(9, 7))
        super().__init__(self.figure)
        self.colorbar = None

    def plot_grid(
        self,
        work,
        x_col,
        y_col,
        value_col,
        xx,
        yy,
        zz,
        palette_name,
        reverse=False,
        custom_colors=None,
        levels_count=30,
        manual_range=None,
        show_stations=True,
        show_contours=True,
    ):
        self.ax.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None

        if custom_colors:
            colors = custom_colors
            cmap_name = 'Custom'
        else:
            colors = [color for _, color in get_palette(palette_name, reverse)]
            cmap_name = palette_name
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

        finite = np.asarray(zz)[np.isfinite(zz)]
        if finite.size == 0:
            raise ValueError('The interpolation produced no finite grid values.')

        if manual_range is None:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = manual_range
            if vmax <= vmin:
                raise ValueError('Manual map maximum must be greater than minimum.')

        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        levels = np.linspace(vmin, vmax, max(3, int(levels_count)))

        contour = self.ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, extend='both')
        if show_contours:
            self.ax.contour(xx, yy, zz, levels=levels, colors='black', linewidths=0.25, alpha=0.40)
        if show_stations:
            self.ax.scatter(work[x_col], work[y_col], s=10, c='black', marker='o', label='Stations')
            self.ax.legend(loc='upper right')

        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_title(value_col)
        self.ax.set_aspect('equal', adjustable='box')
        self.colorbar = self.figure.colorbar(contour, ax=self.ax, pad=0.02)
        self.colorbar.set_label(value_col)
        self.figure.tight_layout()
        self.draw()


class GeoPotentialDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoPotential Mapper')
        self.resize(1500, 920)
        self.df = None
        self.current_grid = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(335)
        controls.setMaximumWidth(410)
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
        self.method_combo.addItems(['Linear', 'Nearest', 'Cubic', 'Ordinary Kriging', 'Minimum Curvature'])
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
        self.custom_check = QCheckBox('Use custom 3-color palette')
        form.addRow('Color palette', self.palette_combo)
        form.addRow('', self.reverse_check)
        form.addRow('', self.custom_check)

        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self.low_color = ColorButton('Low', '#174a9c')
        self.mid_color = ColorButton('Mid', '#ffffff')
        self.high_color = ColorButton('High', '#b51f2e')
        color_layout.addWidget(self.low_color)
        color_layout.addWidget(self.mid_color)
        color_layout.addWidget(self.high_color)
        form.addRow('Custom colors', color_row)

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(3, 100)
        self.levels_spin.setValue(30)
        form.addRow('Contour levels', self.levels_spin)

        self.manual_range_check = QCheckBox('Use manual color range')
        form.addRow('', self.manual_range_check)
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(-1e12, 1e12)
        self.range_min.setDecimals(4)
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(-1e12, 1e12)
        self.range_max.setDecimals(4)
        form.addRow('Color minimum', self.range_min)
        form.addRow('Color maximum', self.range_max)

        self.show_stations = QCheckBox('Show stations')
        self.show_stations.setChecked(True)
        self.show_contours = QCheckBox('Show contour lines')
        self.show_contours.setChecked(True)
        form.addRow('', self.show_stations)
        form.addRow('', self.show_contours)

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
            self, 'Open survey data', '', 'Survey files (*.csv *.xyz *.txt *.dat);;All files (*.*)'
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

            finite = np.asarray(zz)[np.isfinite(zz)]
            if finite.size and not self.manual_range_check.isChecked():
                self.range_min.setValue(float(np.nanmin(finite)))
                self.range_max.setValue(float(np.nanmax(finite)))

            custom_colors = None
            if self.custom_check.isChecked():
                custom_colors = [self.low_color.color, self.mid_color.color, self.high_color.color]
                if self.reverse_check.isChecked():
                    custom_colors.reverse()

            manual_range = None
            if self.manual_range_check.isChecked():
                manual_range = (self.range_min.value(), self.range_max.value())

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
                custom_colors=custom_colors,
                levels_count=self.levels_spin.value(),
                manual_range=manual_range,
                show_stations=self.show_stations.isChecked(),
                show_contours=self.show_contours.isChecked(),
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
