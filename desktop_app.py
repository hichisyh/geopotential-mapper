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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geopotential.processing.gridding import grid_scattered
from geopotential.processing.gravity import (
    polynomial_regional,
    residual_from_regional,
    total_horizontal_gradient,
    upward_continuation,
    vertical_derivative,
)
from geopotential.project.layers import GridLayer, ProjectLayers
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

    def plot_layer(
        self,
        layer,
        palette_name,
        reverse=False,
        custom_colors=None,
        levels_count=30,
        manual_range=None,
        show_contours=True,
        stations=None,
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

        zz = np.asarray(layer.data, dtype=float)
        finite = zz[np.isfinite(zz)]
        if finite.size == 0:
            raise ValueError('The selected layer contains no finite grid values.')

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

        contour = self.ax.contourf(layer.x, layer.y, zz, levels=levels, cmap=cmap, extend='both')
        if show_contours:
            self.ax.contour(layer.x, layer.y, zz, levels=levels, colors='black', linewidths=0.25, alpha=0.40)
        if stations is not None:
            sx, sy = stations
            self.ax.scatter(sx, sy, s=10, c='black', marker='o', label='Stations')
            self.ax.legend(loc='upper right')

        self.ax.set_xlabel('X / Easting')
        self.ax.set_ylabel('Y / Northing')
        subtitle = f' — {layer.operation}' if layer.operation else ''
        self.ax.set_title(f'{layer.name}{subtitle}')
        self.ax.set_aspect('equal', adjustable='box')
        self.colorbar = self.figure.colorbar(contour, ax=self.ax, pad=0.02)
        self.colorbar.set_label(layer.unit or layer.name)
        self.figure.tight_layout()
        self.draw()


class GeoPotentialDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoPotential Mapper')
        self.resize(1600, 940)
        self.df = None
        self.work = None
        self.project = ProjectLayers()
        self.active_layer_id = None
        self.tree_items = {}

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(345)
        controls.setMaximumWidth(430)
        controls_layout = QVBoxLayout(controls)

        import_box = QGroupBox('Data & Gridding')
        import_form = QFormLayout(import_box)
        self.open_button = QPushButton('Open survey file')
        self.open_button.clicked.connect(self.open_file)
        import_form.addRow(self.open_button)
        self.file_label = QLabel('No file loaded')
        self.file_label.setWordWrap(True)
        import_form.addRow('File', self.file_label)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.value_combo = QComboBox()
        import_form.addRow('X / Easting', self.x_combo)
        import_form.addRow('Y / Northing', self.y_combo)
        import_form.addRow('Value', self.value_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItems(['Linear', 'Nearest', 'Cubic', 'Ordinary Kriging', 'Minimum Curvature'])
        import_form.addRow('Gridding', self.method_combo)
        self.variogram_combo = QComboBox()
        self.variogram_combo.addItems(['Spherical', 'Exponential', 'Gaussian', 'Linear', 'Power'])
        import_form.addRow('Kriging variogram', self.variogram_combo)

        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(30, 600)
        self.nx_spin.setValue(180)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(30, 600)
        self.ny_spin.setValue(180)
        import_form.addRow('Grid cells X', self.nx_spin)
        import_form.addRow('Grid cells Y', self.ny_spin)

        self.grid_button = QPushButton('Generate base grid')
        self.grid_button.clicked.connect(self.generate_map)
        import_form.addRow(self.grid_button)
        controls_layout.addWidget(import_box)

        gravity_box = QGroupBox('Gravity Processing')
        gravity_form = QFormLayout(gravity_box)
        self.gravity_operation = QComboBox()
        self.gravity_operation.addItems([
            'Polynomial Regional',
            'Residual (Polynomial)',
            'Upward Continuation',
            'First Vertical Derivative (1VD)',
            'Total Horizontal Gradient (THG)',
        ])
        gravity_form.addRow('Operation', self.gravity_operation)

        self.poly_degree = QSpinBox()
        self.poly_degree.setRange(1, 3)
        self.poly_degree.setValue(1)
        gravity_form.addRow('Polynomial degree', self.poly_degree)

        self.cont_height = QDoubleSpinBox()
        self.cont_height.setRange(0.0, 1000000.0)
        self.cont_height.setDecimals(2)
        self.cont_height.setValue(100.0)
        self.cont_height.setSuffix(' m')
        gravity_form.addRow('Continuation height', self.cont_height)

        self.process_button = QPushButton('Create processed layer')
        self.process_button.clicked.connect(self.process_gravity)
        gravity_form.addRow(self.process_button)
        controls_layout.addWidget(gravity_box)

        display_box = QGroupBox('Map Display')
        display_form = QFormLayout(display_box)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.reverse_check = QCheckBox('Reverse palette')
        self.custom_check = QCheckBox('Use custom 3-color palette')
        display_form.addRow('Color palette', self.palette_combo)
        display_form.addRow('', self.reverse_check)
        display_form.addRow('', self.custom_check)

        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self.low_color = ColorButton('Low', '#174a9c')
        self.mid_color = ColorButton('Mid', '#ffffff')
        self.high_color = ColorButton('High', '#b51f2e')
        color_layout.addWidget(self.low_color)
        color_layout.addWidget(self.mid_color)
        color_layout.addWidget(self.high_color)
        display_form.addRow('Custom colors', color_row)

        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(3, 100)
        self.levels_spin.setValue(30)
        display_form.addRow('Contour levels', self.levels_spin)

        self.manual_range_check = QCheckBox('Use manual color range')
        display_form.addRow('', self.manual_range_check)
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(-1e12, 1e12)
        self.range_min.setDecimals(4)
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(-1e12, 1e12)
        self.range_max.setDecimals(4)
        display_form.addRow('Color minimum', self.range_min)
        display_form.addRow('Color maximum', self.range_max)

        self.show_stations = QCheckBox('Show stations')
        self.show_stations.setChecked(True)
        self.show_contours = QCheckBox('Show contour lines')
        self.show_contours.setChecked(True)
        display_form.addRow('', self.show_stations)
        display_form.addRow('', self.show_contours)

        self.refresh_button = QPushButton('Refresh active layer')
        self.refresh_button.clicked.connect(self.refresh_active_layer)
        display_form.addRow(self.refresh_button)

        self.export_button = QPushButton('Export active grid CSV')
        self.export_button.clicked.connect(self.export_grid)
        display_form.addRow(self.export_button)
        controls_layout.addWidget(display_box)

        self.status_label = QLabel('Ready')
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        project_box = QGroupBox('Project Explorer / Layers')
        project_layout = QVBoxLayout(project_box)
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(['Layer', 'Operation'])
        self.layer_tree.itemSelectionChanged.connect(self.on_layer_selected)
        project_layout.addWidget(self.layer_tree)
        center_layout.addWidget(project_box)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = MapCanvas()
        self.table = QTableWidget()
        self.table.setMaximumHeight(190)
        right_layout.addWidget(self.canvas, stretch=1)
        right_layout.addWidget(self.table)

        splitter.addWidget(controls)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([390, 320, 890])

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
        self.status_label.setText(f'{len(self.df)} rows loaded. Generate a base grid to begin processing.')

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
            self.work, xx, yy, zz, variance = grid_scattered(
                self.df,
                self.x_combo.currentText(),
                self.y_combo.currentText(),
                self.value_combo.currentText(),
                method=method,
                nx=self.nx_spin.value(),
                ny=self.ny_spin.value(),
                variogram_model=self.variogram_combo.currentText().lower(),
            )
            layer = GridLayer(
                name=self.value_combo.currentText(),
                x=xx,
                y=yy,
                data=zz,
                unit='',
                operation=f'{method} grid',
                params={'method': method, 'nx': self.nx_spin.value(), 'ny': self.ny_spin.value()},
            )
            self.project.add(layer)
            self.add_layer_to_tree(layer)
            self.set_active_layer(layer)
            note = ' Kriging variance was also calculated.' if variance is not None else ''
            self.status_label.setText(f'Base grid created as a project layer.{note}')
        except Exception as exc:
            self.status_label.setText('Gridding failed')
            QMessageBox.critical(self, 'Gridding error', str(exc))

    def add_layer_to_tree(self, layer):
        item = QTreeWidgetItem([layer.name, layer.operation])
        item.setData(0, Qt.UserRole, layer.id)
        parent_item = self.tree_items.get(layer.parent_id)
        if parent_item is None:
            self.layer_tree.addTopLevelItem(item)
        else:
            parent_item.addChild(item)
            parent_item.setExpanded(True)
        self.tree_items[layer.id] = item
        self.layer_tree.setCurrentItem(item)

    def on_layer_selected(self):
        items = self.layer_tree.selectedItems()
        if not items:
            return
        layer_id = items[0].data(0, Qt.UserRole)
        layer = self.project.get(layer_id)
        if layer is not None:
            self.set_active_layer(layer)

    def set_active_layer(self, layer):
        self.active_layer_id = layer.id
        finite = np.asarray(layer.data)[np.isfinite(layer.data)]
        if finite.size and not self.manual_range_check.isChecked():
            self.range_min.setValue(float(np.nanmin(finite)))
            self.range_max.setValue(float(np.nanmax(finite)))
        self.refresh_active_layer()

    def active_layer(self):
        return self.project.get(self.active_layer_id) if self.active_layer_id else None

    def display_settings(self):
        custom_colors = None
        if self.custom_check.isChecked():
            custom_colors = [self.low_color.color, self.mid_color.color, self.high_color.color]
            if self.reverse_check.isChecked():
                custom_colors.reverse()

        manual_range = None
        if self.manual_range_check.isChecked():
            manual_range = (self.range_min.value(), self.range_max.value())
        return custom_colors, manual_range

    def refresh_active_layer(self):
        layer = self.active_layer()
        if layer is None:
            return
        try:
            custom_colors, manual_range = self.display_settings()
            stations = None
            if self.show_stations.isChecked() and self.work is not None:
                stations = (
                    self.work[self.x_combo.currentText()].to_numpy(dtype=float),
                    self.work[self.y_combo.currentText()].to_numpy(dtype=float),
                )
            self.canvas.plot_layer(
                layer,
                self.palette_combo.currentText(),
                reverse=self.reverse_check.isChecked(),
                custom_colors=custom_colors,
                levels_count=self.levels_spin.value(),
                manual_range=manual_range,
                show_contours=self.show_contours.isChecked(),
                stations=stations,
            )
        except Exception as exc:
            QMessageBox.critical(self, 'Display error', str(exc))

    def process_gravity(self):
        source = self.active_layer()
        if source is None:
            QMessageBox.information(self, 'No layer', 'Generate or select a grid layer first.')
            return

        operation = self.gravity_operation.currentText()
        self.status_label.setText(f'Processing {operation}...')
        QApplication.processEvents()
        try:
            if operation == 'Polynomial Regional':
                degree = self.poly_degree.value()
                result = polynomial_regional(source.x, source.y, source.data, degree=degree)
                name = f'{source.name} Regional P{degree}'
                params = {'degree': degree}
                unit = source.unit
            elif operation == 'Residual (Polynomial)':
                degree = self.poly_degree.value()
                regional = polynomial_regional(source.x, source.y, source.data, degree=degree)
                result = residual_from_regional(source.data, regional)
                name = f'{source.name} Residual P{degree}'
                params = {'degree': degree}
                unit = source.unit
            elif operation == 'Upward Continuation':
                height = self.cont_height.value()
                result = upward_continuation(source.x, source.y, source.data, height=height)
                name = f'{source.name} Up {height:g}m'
                params = {'height_m': height}
                unit = source.unit
            elif operation == 'First Vertical Derivative (1VD)':
                result = vertical_derivative(source.x, source.y, source.data)
                name = f'{source.name} 1VD'
                params = {}
                unit = f'{source.unit}/m' if source.unit else 'value/m'
            elif operation == 'Total Horizontal Gradient (THG)':
                result = total_horizontal_gradient(source.x, source.y, source.data)
                name = f'{source.name} THG'
                params = {}
                unit = f'{source.unit}/m' if source.unit else 'value/m'
            else:
                raise ValueError(f'Unsupported operation: {operation}')

            layer = GridLayer(
                name=name,
                x=source.x.copy(),
                y=source.y.copy(),
                data=result,
                unit=unit,
                operation=operation,
                params=params,
                parent_id=source.id,
            )
            self.project.add(layer)
            self.add_layer_to_tree(layer)
            self.set_active_layer(layer)
            self.status_label.setText(f'Created layer: {name}')
        except Exception as exc:
            self.status_label.setText('Gravity processing failed')
            QMessageBox.critical(self, 'Processing error', str(exc))

    def export_grid(self):
        layer = self.active_layer()
        if layer is None:
            QMessageBox.information(self, 'No grid', 'Generate or select a grid layer first.')
            return
        default_name = layer.name.replace(' ', '_').replace('/', '_') + '.csv'
        path, _ = QFileDialog.getSaveFileName(self, 'Export grid', default_name, 'CSV files (*.csv)')
        if not path:
            return
        pd.DataFrame({
            'x': layer.x.ravel(),
            'y': layer.y.ravel(),
            'value': np.asarray(layer.data).ravel(),
        }).dropna().to_csv(path, index=False)
        self.status_label.setText(f'Layer exported to {path}')


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('GeoPotential Mapper')
    window = GeoPotentialDesktop()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
