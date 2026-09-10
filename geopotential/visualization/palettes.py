PALETTES = {
    'Geophysics Classic': [
        [0.00, '#1b2a97'],
        [0.16, '#1857d8'],
        [0.32, '#16b7e5'],
        [0.48, '#2bc66d'],
        [0.64, '#e2df32'],
        [0.80, '#f58b26'],
        [1.00, '#c81d25'],
    ],
    'Gravity Diverging': [
        [0.00, '#223b8f'],
        [0.20, '#3e7bc4'],
        [0.40, '#a6cee3'],
        [0.50, '#f7f7f7'],
        [0.60, '#fdbb84'],
        [0.80, '#e34a33'],
        [1.00, '#8c1d18'],
    ],
    'Magnetic Spectrum': [
        [0.00, '#172a88'],
        [0.18, '#1464d2'],
        [0.36, '#00b8d4'],
        [0.52, '#34c759'],
        [0.68, '#d6e229'],
        [0.84, '#ff9f0a'],
        [1.00, '#d7263d'],
    ],
    'Terrain': [
        [0.00, '#1d4f91'],
        [0.25, '#4fa3a5'],
        [0.45, '#72b35a'],
        [0.62, '#c9c56d'],
        [0.78, '#9a6f43'],
        [1.00, '#f0f0f0'],
    ],
    'Grayscale': [[0.00, '#111111'], [1.00, '#f4f4f4']],
    'Blue-White-Red': [[0.00, '#174a9c'], [0.50, '#ffffff'], [1.00, '#b51f2e']],
}


def get_palette(name, reverse=False):
    palette = PALETTES.get(name, PALETTES['Geophysics Classic'])
    if not reverse:
        return palette
    return [[1.0 - stop, color] for stop, color in reversed(palette)]
