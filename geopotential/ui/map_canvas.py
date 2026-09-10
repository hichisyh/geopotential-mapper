import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy


class MapCanvas(FigureCanvas):
    """Responsive map-only Matplotlib canvas.

    The color scale is intentionally rendered by a separate Qt widget. This
    keeps Matplotlib from allocating or reallocating map space for a colorbar
    during repeated Generate/Refresh actions.
    """

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 7), constrained_layout=False)
        self.ax = self.figure.add_axes([0.09, 0.09, 0.88, 0.84])
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._last_mappable = None

    def clear_map(self):
        self.ax.clear()
        self._last_mappable = None
        self.draw_idle()

    def plot_layer(
        self,
        layer,
        color_scale,
        levels_count=30,
        show_contours=True,
        stations=None,
    ):
        self.ax.clear()

        zz = np.asarray(layer.data, dtype=float)
        finite = zz[np.isfinite(zz)]
        if finite.size == 0:
            raise ValueError('The selected layer contains no finite grid values.')

        vmin = float(color_scale.vmin)
        vmax = float(color_scale.vmax)
        if vmax <= vmin:
            raise ValueError('Color scale maximum must be greater than minimum.')

        cmap = color_scale.to_colormap()
        levels = np.linspace(vmin, vmax, max(3, int(levels_count)))
        contour = self.ax.contourf(
            layer.x,
            layer.y,
            zz,
            levels=levels,
            cmap=cmap,
            extend='both',
        )
        self._last_mappable = contour

        if show_contours:
            self.ax.contour(
                layer.x,
                layer.y,
                zz,
                levels=levels,
                colors='black',
                linewidths=0.25,
                alpha=0.40,
            )

        if stations is not None:
            sx, sy = stations
            self.ax.scatter(sx, sy, s=10, c='black', marker='o', label='Stations')
            self.ax.legend(loc='upper right')

        self.ax.set_xlabel('X / Easting')
        self.ax.set_ylabel('Y / Northing')
        subtitle = f' — {layer.operation}' if layer.operation else ''
        self.ax.set_title(f'{layer.name}{subtitle}')
        self.ax.set_aspect('equal', adjustable='box')
        self.draw_idle()
