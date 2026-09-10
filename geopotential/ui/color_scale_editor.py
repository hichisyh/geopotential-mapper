from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from geopotential.visualization.color_scale import ColorScaleModel


class ColorStopDialog(QDialog):
    def __init__(self, parent=None, value=0.0, color='#ffffff', vmin=-1e12, vmax=1e12, title='Color stop'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._color = color
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(float(vmin), float(vmax))
        self.value_spin.setDecimals(6)
        self.value_spin.setValue(float(value))
        self.color_button = QPushButton()
        self.color_button.clicked.connect(self.choose_color)
        self._refresh_color_button()
        form.addRow('Value', self.value_spin)
        form.addRow('Color', self.color_button)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def color(self):
        return self._color

    def choose_color(self):
        selected = QColorDialog.getColor(QColor(self._color), self, 'Choose color')
        if selected.isValid():
            self._color = selected.name()
            self._refresh_color_button()

    def _refresh_color_button(self):
        self.color_button.setText(self._color)
        self.color_button.setStyleSheet(
            f'background:{self._color}; color:{"black" if QColor(self._color).lightness() > 130 else "white"}; padding:6px;'
        )


class ColorScaleBar(QWidget):
    selectionChanged = Signal(object)
    scaleChanged = Signal()
    editRequested = Signal(str)

    def __init__(self, model: ColorScaleModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.selected_id = None
        self.dragging = False
        self.setMinimumWidth(120)
        self.setMinimumHeight(280)
        self.setMouseTracking(True)

    def set_model(self, model):
        self.model = model
        self.selected_id = None
        self.update()

    def _bar_rect(self):
        return QRectF(40, 18, 30, max(80, self.height() - 36))

    def _value_to_y(self, value):
        rect = self._bar_rect()
        span = self.model.vmax - self.model.vmin
        if span <= 0:
            return rect.bottom()
        t = (float(value) - self.model.vmin) / span
        return rect.bottom() - min(1.0, max(0.0, t)) * rect.height()

    def _y_to_value(self, y):
        rect = self._bar_rect()
        t = (rect.bottom() - y) / rect.height()
        t = min(1.0, max(0.0, t))
        return self.model.vmin + t * (self.model.vmax - self.model.vmin)

    def _hit_stop(self, pos):
        for stop in reversed(self.model.stops):
            y = self._value_to_y(stop.value)
            if abs(pos.y() - y) <= 9 and 20 <= pos.x() <= self.width() - 4:
                return stop
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self._bar_rect()

        gradient = QLinearGradient(rect.left(), rect.bottom(), rect.left(), rect.top())
        for t, color in self.model.normalized_stops():
            gradient.setColorAt(t, QColor(color))
        painter.fillRect(rect, gradient)
        painter.setPen(QPen(QColor('#5b6570'), 1))
        painter.drawRect(rect)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        for stop in self.model.stops:
            y = self._value_to_y(stop.value)
            selected = stop.id == self.selected_id
            marker_color = QColor('#1f6feb' if selected else '#343a40')
            painter.setPen(QPen(marker_color, 1.5))
            painter.setBrush(QColor(stop.color))
            marker = QPolygonF([
                QPointF(rect.right() + 4, y),
                QPointF(rect.right() + 13, y - 6),
                QPointF(rect.right() + 13, y + 6),
            ])
            painter.drawPolygon(marker)
            painter.drawText(QRectF(rect.right() + 17, y - 10, max(45, self.width() - rect.right() - 20), 20), Qt.AlignVCenter, f'{stop.value:.4g}')

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        stop = self._hit_stop(event.position())
        if stop is not None:
            self.selected_id = stop.id
            self.dragging = True
            self.selectionChanged.emit(stop.id)
            self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging or not self.selected_id:
            return
        self.model.update_stop(self.selected_id, value=self._y_to_value(event.position().y()))
        self.scaleChanged.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def mouseDoubleClickEvent(self, event):
        stop = self._hit_stop(event.position())
        if stop is not None:
            self.selected_id = stop.id
            self.selectionChanged.emit(stop.id)
            self.editRequested.emit(stop.id)
            self.update()


class ColorScaleEditor(QWidget):
    scaleChanged = Signal()

    def __init__(self, model: ColorScaleModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.setMinimumWidth(150)
        self.setMaximumWidth(185)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel('Color Scale')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-weight:700;')
        layout.addWidget(title)

        self.bar = ColorScaleBar(self.model, self)
        self.bar.scaleChanged.connect(self.scaleChanged)
        self.bar.editRequested.connect(self.modify_selected)
        layout.addWidget(self.bar, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(4)
        self.add_button = QPushButton('+')
        self.add_button.setToolTip('Add color stop')
        self.delete_button = QPushButton('−')
        self.delete_button.setToolTip('Delete selected color stop')
        self.modify_button = QPushButton('Edit')
        self.modify_button.setToolTip('Modify selected color stop')
        controls.addWidget(self.add_button)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.modify_button)
        layout.addLayout(controls)

        self.add_button.clicked.connect(self.add_stop)
        self.delete_button.clicked.connect(self.delete_selected)
        self.modify_button.clicked.connect(self.modify_selected)

    def set_model(self, model):
        self.model = model
        self.bar.set_model(model)

    def add_stop(self):
        default = (self.model.vmin + self.model.vmax) / 2.0
        dialog = ColorStopDialog(
            self,
            value=default,
            color='#ffffff',
            vmin=self.model.vmin,
            vmax=self.model.vmax,
            title='Add color stop',
        )
        if dialog.exec() == QDialog.Accepted:
            stop = self.model.add_stop(dialog.value_spin.value(), dialog.color)
            self.bar.selected_id = stop.id
            self.bar.update()
            self.scaleChanged.emit()

    def delete_selected(self):
        if not self.bar.selected_id:
            QMessageBox.information(self, 'Color scale', 'Select a color stop first.')
            return
        try:
            if self.model.remove_stop(self.bar.selected_id):
                self.bar.selected_id = None
                self.bar.update()
                self.scaleChanged.emit()
        except ValueError as exc:
            QMessageBox.information(self, 'Color scale', str(exc))

    def modify_selected(self, stop_id=None):
        if isinstance(stop_id, bool):
            stop_id = None
        selected_id = stop_id or self.bar.selected_id
        if not selected_id:
            QMessageBox.information(self, 'Color scale', 'Select a color stop first.')
            return
        stop = self.model.get(selected_id)
        if stop is None:
            return
        dialog = ColorStopDialog(
            self,
            value=stop.value,
            color=stop.color,
            vmin=self.model.vmin,
            vmax=self.model.vmax,
            title='Modify color stop',
        )
        if dialog.exec() == QDialog.Accepted:
            self.model.update_stop(selected_id, dialog.value_spin.value(), dialog.color)
            self.bar.update()
            self.scaleChanged.emit()
