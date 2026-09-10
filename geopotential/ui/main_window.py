import json

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDockWidget,
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
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolBox,
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
from geopotential.ui.map_canvas import MapCanvas
from geopotential.visualization.palettes import PALETTES


class ColorButton(QPushButton):
    def __init__(self, text, color):
        super().__init__(text)
        self.color = color
        self.clicked.connect(self.choose_color)
        self.refresh()

    def choose_color(self):
        selected = QColorDialog.getColor(QColor(self.color), self, f"Choose {self.text()} color")
        if selected.isValid():
            self.color = selected.name()
            self.refresh()

    def refresh(self):
        self.setStyleSheet(
            f"background-color:{self.color}; color:black; font-weight:600; padding:5px;"
        )


class GeoPotentialMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoPotential Mapper")
        self.resize(1500, 900)
        self.setMinimumSize(980, 650)

        self.df = None
        self.work = None
        self.project = ProjectLayers()
        self.active_layer_id = None
        self.tree_items = {}

        self._build_toolbar()
        self._build_central_layout()
        self._build_data_dock()
        self._build_status_bar()
        self._apply_app_style()

    # ---------- UI construction ----------
    def _build_toolbar(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.open_action = QAction("Open", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.open_file)

        self.generate_action = QAction("Generate", self)
        self.generate_action.setShortcut("Ctrl+G")
        self.generate_action.triggered.connect(self.generate_map)

        self.refresh_action = QAction("Refresh", self)
        self.refresh_action.setShortcut("F5")
        self.refresh_action.triggered.connect(self.refresh_active_layer)

        self.reset_action = QAction("Reset", self)
        self.reset_action.triggered.connect(self.reset_project)

        self.save_action = QAction("Save Project", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.triggered.connect(self.save_project)

        self.export_action = QAction("Export Grid", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self.export_grid)

        self.table_action = QAction("Data Table", self)
        self.table_action.setCheckable(True)
        self.table_action.toggled.connect(self._toggle_data_dock)

        for action in (
            self.open_action,
            self.generate_action,
            self.refresh_action,
            self.reset_action,
            self.save_action,
            self.export_action,
        ):
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addAction(self.table_action)

    def _build_central_layout(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        sidebar = QWidget()
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(390)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        project_box = QGroupBox("Project / Layers")
        project_layout = QVBoxLayout(project_box)
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabels(["Layer", "Operation"])
        self.layer_tree.setAlternatingRowColors(True)
        self.layer_tree.setMinimumHeight(180)
        self.layer_tree.itemSelectionChanged.connect(self.on_layer_selected)
        project_layout.addWidget(self.layer_tree)
        sidebar_layout.addWidget(project_box)

        self.options = QToolBox()
        self.options.addItem(self._make_data_page(), "Data Options")
        self.options.addItem(self._make_gridding_page(), "Gridding")
        self.options.addItem(self._make_processing_page(), "Gravity Processing")
        self.options.addItem(self._make_style_page(), "Map Style")
        self.options.addItem(self._make_color_page(), "Color Scale Settings")
        sidebar_layout.addWidget(self.options, stretch=1)

        self.canvas = MapCanvas(self)
        layout.addWidget(sidebar, stretch=0)
        layout.addWidget(self.canvas, stretch=1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)

    def _make_data_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.value_combo = QComboBox()
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["", "mGal", "nT", "SI"])
        form.addRow("File", self.file_label)
        form.addRow("X / Easting", self.x_combo)
        form.addRow("Y / Northing", self.y_combo)
        form.addRow("Value", self.value_combo)
        form.addRow("Unit", self.unit_combo)
        return page

    def _make_gridding_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["Linear", "Nearest", "Cubic", "Ordinary Kriging", "Minimum Curvature"]
        )
        self.method_combo.currentTextChanged.connect(self._update_gridding_controls)
        self.variogram_combo = QComboBox()
        self.variogram_combo.addItems(["Spherical", "Exponential", "Gaussian", "Linear", "Power"])
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(30, 600)
        self.nx_spin.setValue(180)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(30, 600)
        self.ny_spin.setValue(180)
        form.addRow("Method", self.method_combo)
        form.addRow("Variogram", self.variogram_combo)
        form.addRow("Grid cells X", self.nx_spin)
        form.addRow("Grid cells Y", self.ny_spin)
        self._update_gridding_controls(self.method_combo.currentText())
        return page

    def _make_processing_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.gravity_operation = QComboBox()
        self.gravity_operation.addItems(
            [
                "Polynomial Regional",
                "Residual (Polynomial)",
                "Upward Continuation",
                "First Vertical Derivative (1VD)",
                "Total Horizontal Gradient (THG)",
            ]
        )
        self.gravity_operation.currentTextChanged.connect(self._update_processing_controls)
        self.poly_degree = QSpinBox()
        self.poly_degree.setRange(1, 3)
        self.poly_degree.setValue(1)
        self.cont_height = QDoubleSpinBox()
        self.cont_height.setRange(0.0, 1000000.0)
        self.cont_height.setDecimals(2)
        self.cont_height.setValue(100.0)
        self.cont_height.setSuffix(" m")
        self.process_button = QPushButton("Create processed layer")
        self.process_button.clicked.connect(self.process_gravity)
        form.addRow("Operation", self.gravity_operation)
        form.addRow("Polynomial degree", self.poly_degree)
        form.addRow("Continuation height", self.cont_height)
        form.addRow(self.process_button)
        self._update_processing_controls(self.gravity_operation.currentText())
        return page

    def _make_style_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.station_mode = QComboBox()
        self.station_mode.addItems(["Points", "None"])
        self.contour_mode = QComboBox()
        self.contour_mode.addItems(["Fill + Lines", "Fill only"])
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(3, 100)
        self.levels_spin.setValue(30)
        form.addRow("Stations", self.station_mode)
        form.addRow("Contours", self.contour_mode)
        form.addRow("Contour levels", self.levels_spin)
        return page

    def _make_color_page(self):
        page = QWidget()
        form = QFormLayout(page)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(list(PALETTES.keys()))
        self.palette_direction = QComboBox()
        self.palette_direction.addItems(["Normal", "Reverse"])
        self.palette_mode = QComboBox()
        self.palette_mode.addItems(["Preset palette", "Custom 3-color"])
        self.range_mode = QComboBox()
        self.range_mode.addItems(["Automatic", "Manual", "Symmetric around zero"])
        self.range_mode.currentTextChanged.connect(self._update_range_controls)
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(-1e12, 1e12)
        self.range_min.setDecimals(4)
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(-1e12, 1e12)
        self.range_max.setDecimals(4)

        color_row = QWidget()
        row = QHBoxLayout(color_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.low_color = ColorButton("Low", "#174a9c")
        self.mid_color = ColorButton("Mid", "#ffffff")
        self.high_color = ColorButton("High", "#b51f2e")
        row.addWidget(self.low_color)
        row.addWidget(self.mid_color)
        row.addWidget(self.high_color)

        form.addRow("Palette", self.palette_combo)
        form.addRow("Direction", self.palette_direction)
        form.addRow("Palette type", self.palette_mode)
        form.addRow("Range", self.range_mode)
        form.addRow("Minimum", self.range_min)
        form.addRow("Maximum", self.range_max)
        form.addRow("Custom colors", color_row)
        self._update_range_controls(self.range_mode.currentText())
        return page

    def _build_data_dock(self):
        self.data_dock = QDockWidget("Survey Data", self)
        self.data_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.table = QTableWidget()
        self.data_dock.setWidget(self.table)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.data_dock)
        self.data_dock.hide()
        self.data_dock.visibilityChanged.connect(self.table_action.setChecked)

    def _build_status_bar(self):
        self.statusBar().showMessage("Ready")
        self.meta_label = QLabel("No active grid")
        self.statusBar().addPermanentWidget(self.meta_label)

    def _apply_app_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6f8; }
            QToolBar { spacing: 6px; padding: 5px; background: #ffffff; border-bottom: 1px solid #d7dce2; }
            QToolButton { padding: 6px 10px; }
            QGroupBox { font-weight: 600; border: 1px solid #d7dce2; border-radius: 6px; margin-top: 8px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QToolBox::tab { padding: 8px; font-weight: 600; background: #e9edf2; border-radius: 4px; }
            QToolBox::tab:selected { background: #ffffff; }
            QComboBox, QSpinBox, QDoubleSpinBox { min-height: 25px; padding: 2px 5px; }
            QPushButton { min-height: 28px; padding: 4px 8px; }
            QTreeWidget, QTableWidget { background: #ffffff; border: 1px solid #d7dce2; }
            """
        )

    # ---------- contextual controls ----------
    def _update_gridding_controls(self, method):
        self.variogram_combo.setEnabled(method == "Ordinary Kriging")

    def _update_processing_controls(self, operation):
        poly = "Polynomial" in operation
        self.poly_degree.setEnabled(poly)
        self.cont_height.setEnabled(operation == "Upward Continuation")

    def _update_range_controls(self, mode):
        manual = mode == "Manual"
        self.range_min.setEnabled(manual)
        self.range_max.setEnabled(manual)

    # ---------- data actions ----------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open survey data",
            "",
            "Survey files (*.csv *.xyz *.txt *.dat);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))
            return

        self.file_label.setText(path)
        columns = [str(c) for c in self.df.columns]
        for combo in (self.x_combo, self.y_combo, self.value_combo):
            combo.clear()
            combo.addItems(columns)

        lowered = {str(c).lower(): str(c) for c in self.df.columns}
        self._select_guess(self.x_combo, lowered, ["x", "easting", "east", "utm_x"])
        self._select_guess(self.y_combo, lowered, ["y", "northing", "north", "utm_y"])
        self._select_guess(
            self.value_combo,
            lowered,
            ["bouguer", "gravity", "cba", "tmi", "magnetic", "magnetics"],
        )
        self.populate_table()
        self.statusBar().showMessage(f"Loaded {len(self.df)} rows")

    @staticmethod
    def _select_guess(combo, lowered, candidates):
        for key in candidates:
            if key in lowered:
                combo.setCurrentText(lowered[key])
                return

    def populate_table(self):
        if self.df is None:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        preview = self.df.head(200)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for row_index, (_, row) in enumerate(preview.iterrows()):
            for col_index, value in enumerate(row):
                self.table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def generate_map(self):
        if self.df is None:
            QMessageBox.information(self, "No data", "Load a survey file first.")
            return
        method = self.method_combo.currentText()
        self.statusBar().showMessage(f"Running {method}...")
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
                unit=self.unit_combo.currentText(),
                operation=f"{method} grid",
                params={"method": method, "nx": self.nx_spin.value(), "ny": self.ny_spin.value()},
            )
            self.project.add(layer)
            self.add_layer_to_tree(layer)
            self.set_active_layer(layer)
            note = " Kriging variance calculated." if variance is not None else ""
            self.statusBar().showMessage(f"Grid generated.{note}")
        except Exception as exc:
            self.statusBar().showMessage("Gridding failed")
            QMessageBox.critical(self, "Gridding error", str(exc))

    # ---------- project/layer actions ----------
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
        layer = self.project.get(items[0].data(0, Qt.UserRole))
        if layer is not None:
            self.set_active_layer(layer)

    def active_layer(self):
        return self.project.get(self.active_layer_id) if self.active_layer_id else None

    def set_active_layer(self, layer):
        self.active_layer_id = layer.id
        finite = np.asarray(layer.data)[np.isfinite(layer.data)]
        if finite.size:
            self.range_min.setValue(float(np.nanmin(finite)))
            self.range_max.setValue(float(np.nanmax(finite)))
        self.meta_label.setText(f"{layer.name} | {layer.data.shape[1]} × {layer.data.shape[0]}")
        self.refresh_active_layer()

    def process_gravity(self):
        source = self.active_layer()
        if source is None:
            QMessageBox.information(self, "No layer", "Generate or select a grid layer first.")
            return

        op = self.gravity_operation.currentText()
        self.statusBar().showMessage(f"Processing {op}...")
        QApplication.processEvents()
        try:
            params = {}
            if op == "Polynomial Regional":
                degree = self.poly_degree.value()
                result = polynomial_regional(source.x, source.y, source.data, degree=degree)
                name = f"{source.name} Regional P{degree}"
                params = {"degree": degree}
            elif op == "Residual (Polynomial)":
                degree = self.poly_degree.value()
                regional = polynomial_regional(source.x, source.y, source.data, degree=degree)
                result = residual_from_regional(source.data, regional)
                name = f"{source.name} Residual P{degree}"
                params = {"degree": degree}
            elif op == "Upward Continuation":
                height = self.cont_height.value()
                result = upward_continuation(source.x, source.y, source.data, height)
                name = f"{source.name} Up {height:g}m"
                params = {"height_m": height}
            elif op == "First Vertical Derivative (1VD)":
                result = vertical_derivative(source.x, source.y, source.data)
                name = f"{source.name} 1VD"
            elif op == "Total Horizontal Gradient (THG)":
                result = total_horizontal_gradient(source.x, source.y, source.data)
                name = f"{source.name} THG"
            else:
                raise ValueError(f"Unsupported operation: {op}")

            layer = GridLayer(
                name=name,
                x=source.x,
                y=source.y,
                data=result,
                unit=source.unit,
                operation=op,
                params=params,
                parent_id=source.id,
            )
            self.project.add(layer)
            self.add_layer_to_tree(layer)
            self.set_active_layer(layer)
            self.statusBar().showMessage(f"Created {name}")
        except Exception as exc:
            self.statusBar().showMessage("Processing failed")
            QMessageBox.critical(self, "Processing error", str(exc))

    # ---------- rendering ----------
    def _display_settings(self, layer):
        reverse = self.palette_direction.currentText() == "Reverse"
        custom = None
        if self.palette_mode.currentText() == "Custom 3-color":
            custom = [self.low_color.color, self.mid_color.color, self.high_color.color]
            if reverse:
                custom.reverse()

        finite = np.asarray(layer.data)[np.isfinite(layer.data)]
        mode = self.range_mode.currentText()
        manual_range = None
        if mode == "Manual":
            manual_range = (self.range_min.value(), self.range_max.value())
        elif mode == "Symmetric around zero" and finite.size:
            bound = float(np.nanmax(np.abs(finite)))
            manual_range = (-bound, bound)
        return reverse, custom, manual_range

    def refresh_active_layer(self):
        layer = self.active_layer()
        if layer is None:
            return
        try:
            reverse, custom, manual_range = self._display_settings(layer)
            stations = None
            if self.station_mode.currentText() == "Points" and self.work is not None:
                stations = (
                    self.work[self.x_combo.currentText()].to_numpy(dtype=float),
                    self.work[self.y_combo.currentText()].to_numpy(dtype=float),
                )
            self.canvas.plot_layer(
                layer,
                self.palette_combo.currentText(),
                reverse=reverse,
                custom_colors=custom,
                levels_count=self.levels_spin.value(),
                manual_range=manual_range,
                show_contours=self.contour_mode.currentText() == "Fill + Lines",
                stations=stations,
            )
            self.statusBar().showMessage("View refreshed")
        except Exception as exc:
            QMessageBox.critical(self, "Display error", str(exc))

    # ---------- toolbar actions ----------
    def reset_project(self):
        self.project.clear()
        self.layer_tree.clear()
        self.tree_items.clear()
        self.active_layer_id = None
        self.work = None
        self.canvas.clear_map()
        self.meta_label.setText("No active grid")
        self.statusBar().showMessage("Project layers reset")

    def export_grid(self):
        layer = self.active_layer()
        if layer is None:
            QMessageBox.information(self, "No grid", "Generate or select a grid first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export grid", "grid.csv", "CSV files (*.csv)")
        if not path:
            return
        pd.DataFrame(
            {"x": layer.x.ravel(), "y": layer.y.ravel(), "value": np.asarray(layer.data).ravel()}
        ).dropna().to_csv(path, index=False)
        self.statusBar().showMessage(f"Exported {path}")

    def save_project(self):
        if not self.project.all():
            QMessageBox.information(self, "Empty project", "There are no grid layers to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "geopotential_project.npz", "GeoPotential project (*.npz)"
        )
        if not path:
            return
        if not path.lower().endswith(".npz"):
            path += ".npz"

        arrays = {}
        metadata = []
        for i, layer in enumerate(self.project.all()):
            prefix = f"layer_{i}"
            arrays[f"{prefix}_x"] = np.asarray(layer.x)
            arrays[f"{prefix}_y"] = np.asarray(layer.y)
            arrays[f"{prefix}_data"] = np.asarray(layer.data)
            metadata.append(
                {
                    "prefix": prefix,
                    "id": layer.id,
                    "name": layer.name,
                    "unit": layer.unit,
                    "operation": layer.operation,
                    "params": layer.params,
                    "parent_id": layer.parent_id,
                }
            )
        arrays["metadata_json"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
        self.statusBar().showMessage(f"Project saved to {path}")

    def _toggle_data_dock(self, visible):
        self.data_dock.setVisible(visible)
