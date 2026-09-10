import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

from geopotential.visualization.palettes import get_palette


class MapCanvas(FigureCanvas):
    """Responsive Matplotlib canvas with a permanent colorbar axis.

    The map axis and colorbar axis are created once and reused on every redraw,
    preventing repeated Figure.colorbar() calls from progressively shrinking the map.
    """

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(10, 7), constrained_layout=False)
        grid = self.figure.add_gridspec(
            1,
            2,
            width_ratios=[32, 1],
            left=0.075,
            right=0.965,
            bottom=0.08,
            top=0.94,
            wspace=0.10,
        )
        self.ax = self.figure.add_subplot(grid[0, 0])
        self.cax = self.figure.add_subplot(grid[0, 1])
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self._last_mappable = None

    def clear_map(self):
        self.ax.clear()
        self.cax.clear()
        self._last_mappable = None
        self.draw_idle()

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
        self.cax.clear()

        if custom_colors:
            colors = custom_colors
            cmap_name = "Custom"
        else:
            colors = [color for _, color in get_palette(palette_name, reverse)]
            cmap_name = palette_name
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

        zz = np.asarray(layer.data, dtype=float)
        finite = zz[np.isfinite(zz)]
        if finite.size == 0:
            raise ValueError("The selected layer contains no finite grid values.")

        if manual_range is None:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = manual_range
            if vmax <= vmin:
                raise ValueError("Manual map maximum must be greater than minimum.")

        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9

        levels = np.linspace(vmin, vmax, max(3, int(levels_count)))
        contour = self.ax.contourf(
            layer.x,
            layer.y,
            zz,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        self._last_mappable = contour

        if show_contours:
            self.ax.contour(
                layer.x,
                layer.y,
                zz,
                levels=levels,
                colors="black",
                linewidths=0.25,
                alpha=0.40,
            )

        if stations is not None:
            sx, sy = stations
            self.ax.scatter(sx, sy, s=10, c="black", marker="o", label="Stations")
            self.ax.legend(loc="upper right")

        self.ax.set_xlabel("X / Easting")
        self.ax.set_ylabel("Y / Northing")
        subtitle = f" — {layer.operation}" if layer.operation else ""
        self.ax.set_title(f"{layer.name}{subtitle}")
        self.ax.set_aspect("equal", adjustable="box")

        colorbar = self.figure.colorbar(contour, cax=self.cax)
        colorbar.set_label(layer.unit or layer.name)

        self.draw_idle()
