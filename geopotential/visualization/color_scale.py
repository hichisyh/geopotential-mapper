from dataclasses import dataclass, field
from uuid import uuid4

from matplotlib.colors import LinearSegmentedColormap

from geopotential.visualization.palettes import PALETTES


@dataclass
class ColorStop:
    value: float
    color: str
    id: str = field(default_factory=lambda: uuid4().hex)


class ColorScaleModel:
    """Value-based color scale independent from Qt and Matplotlib widgets."""

    def __init__(self, stops=None, vmin=0.0, vmax=1.0):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.stops = list(stops or [])
        self._normalize_bounds()
        self.sort()

    def _normalize_bounds(self):
        if self.vmax <= self.vmin:
            self.vmax = self.vmin + 1.0

    def set_range(self, vmin, vmax, preserve_relative=False):
        old_min, old_max = self.vmin, self.vmax
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._normalize_bounds()
        if preserve_relative and self.stops and old_max > old_min:
            old_span = old_max - old_min
            new_span = self.vmax - self.vmin
            for stop in self.stops:
                t = (stop.value - old_min) / old_span
                stop.value = self.vmin + t * new_span
        self.clamp()
        self.sort()

    def from_palette(self, palette_name, vmin, vmax, reverse=False):
        palette = PALETTES.get(palette_name, PALETTES['Geophysics Classic'])
        if reverse:
            palette = [[1.0 - p, c] for p, c in reversed(palette)]
        self.vmin, self.vmax = float(vmin), float(vmax)
        self._normalize_bounds()
        span = self.vmax - self.vmin
        self.stops = [ColorStop(self.vmin + float(p) * span, color) for p, color in palette]
        self.sort()

    def add_stop(self, value, color):
        stop = ColorStop(float(value), str(color))
        self.stops.append(stop)
        self.clamp()
        self.sort()
        return stop

    def remove_stop(self, stop_id):
        if len(self.stops) <= 2:
            raise ValueError('A color scale requires at least two stops.')
        before = len(self.stops)
        self.stops = [s for s in self.stops if s.id != stop_id]
        return len(self.stops) != before

    def update_stop(self, stop_id, value=None, color=None):
        stop = self.get(stop_id)
        if stop is None:
            return None
        if value is not None:
            stop.value = float(value)
        if color is not None:
            stop.color = str(color)
        self.clamp()
        self.sort()
        return stop

    def get(self, stop_id):
        for stop in self.stops:
            if stop.id == stop_id:
                return stop
        return None

    def sort(self):
        self.stops.sort(key=lambda s: s.value)

    def clamp(self):
        for stop in self.stops:
            stop.value = min(self.vmax, max(self.vmin, float(stop.value)))

    def normalized_stops(self):
        if not self.stops:
            return [(0.0, '#000000'), (1.0, '#ffffff')]
        span = self.vmax - self.vmin
        if span <= 0:
            span = 1.0
        normalized = []
        for stop in sorted(self.stops, key=lambda s: s.value):
            t = (stop.value - self.vmin) / span
            normalized.append((min(1.0, max(0.0, t)), stop.color))
        if normalized[0][0] > 0.0:
            normalized.insert(0, (0.0, normalized[0][1]))
        if normalized[-1][0] < 1.0:
            normalized.append((1.0, normalized[-1][1]))
        return normalized

    def to_colormap(self, name='InteractiveColorScale'):
        return LinearSegmentedColormap.from_list(name, self.normalized_stops(), N=256)

    def copy(self):
        return ColorScaleModel(
            stops=[ColorStop(s.value, s.color, s.id) for s in self.stops],
            vmin=self.vmin,
            vmax=self.vmax,
        )
