from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


@dataclass
class GridLayer:
    name: str
    x: np.ndarray
    y: np.ndarray
    data: np.ndarray
    unit: str = ''
    operation: str = 'Imported/Grid'
    params: dict = field(default_factory=dict)
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)


class ProjectLayers:
    def __init__(self):
        self._layers: list[GridLayer] = []

    def add(self, layer: GridLayer) -> GridLayer:
        self._layers.append(layer)
        return layer

    def get(self, layer_id: str) -> GridLayer | None:
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None

    def all(self) -> list[GridLayer]:
        return list(self._layers)

    def clear(self):
        self._layers.clear()
