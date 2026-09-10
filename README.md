# GeoPotential Mapper

**GeoPotential Mapper is a focused Python desktop tool for the visualization, gridding, and processing of gravity and magnetic geophysical data.**

It is intended for potential-field workflows: importing survey observations, creating grids, visualizing gravity/magnetic anomaly maps, adjusting geophysical color scales, generating derived processing layers, and exporting results.

> **Scope:** this project is a gravity and magnetic data visualization/processing tool. It is **not** intended to be a general GIS platform, a complete replacement for Oasis montaj, or a full geophysical inversion suite.

## Current capabilities

- Native PySide6/Qt desktop application
- CSV / XYZ / TXT / DAT survey-data import
- Gravity and magnetic data workflows
- Automatic/manual coordinate and value-column mapping
- Linear, nearest-neighbour and cubic interpolation
- Ordinary Kriging with selectable variogram model
- Minimum-curvature-style thin-plate spline interpolation
- Project/layer explorer for base and derived grids
- Geophysics-oriented color palettes
- Interactive value-based color scale
- Add, delete, edit and drag individual color stops
- Live color-scale updates without rerunning gridding
- Per-layer color scales
- Configurable contour levels
- Station and contour overlays
- Polynomial regional field
- Polynomial residual field
- Upward continuation
- First vertical derivative (1VD)
- Total horizontal gradient (THG)
- CSV grid export
- Compressed project saving

> The open minimum-curvature implementation uses a thin-plate spline that minimizes surface bending. It is an open scientific analogue and is not claimed to be numerically identical to Geosoft/Oasis montaj's proprietary minimum-curvature implementation.

## Run the desktop application

```bash
git clone https://github.com/hichisyh/geopotential-mapper.git
cd geopotential-mapper
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python desktop_app.py
```

The application opens as a native desktop window; no browser is required.

## Interactive color scale

The map uses a dedicated Qt color-scale editor rather than a Matplotlib colorbar embedded in the plot layout. This keeps the legend area stable during repeated map generation and allows direct manipulation of geophysical color breaks.

Each color stop is represented by a value and a color. Stops can be added, deleted, modified, or dragged directly on the vertical scale. Changes redraw the active map immediately without recalculating the underlying grid.

Included starting palettes:

- Geophysics Classic
- Gravity Diverging
- Magnetic Spectrum
- Terrain
- Grayscale
- Blue-White-Red

These palettes are original project palettes inspired by common potential-field display conventions; they are not copies of proprietary Geosoft `.tbl` files.

## Gridding

### Ordinary Kriging

Uses PyKrige. Available variogram choices include spherical, exponential, Gaussian, linear and power.

### Minimum Curvature

Uses SciPy thin-plate radial basis interpolation as a minimum-curvature-style open implementation.

## Gravity processing

The current gravity-processing foundation includes:

- Polynomial regional separation (degrees 1–3)
- Residual calculation
- Upward continuation
- First vertical derivative (1VD)
- Total horizontal gradient (THG)

Processing is non-destructive: derived results are created as new child layers rather than overwriting the source grid.

## Magnetic processing

Magnetic visualization uses the same import, gridding, layer-management, mapping and interactive color-scale architecture. Dedicated magnetic processing operations are the next major processing milestone.

Planned magnetic operations include scientifically appropriate TMI/IGRF workflows, RTP/RTE where applicable, derivatives, analytic signal and related potential-field interpretation products.

## Project structure

```text
app.py                         # Streamlit prototype
desktop_app.py                 # native desktop launcher
geopotential/
  io/
    importer.py
  processing/
    gridding.py
    gravity.py
  project/
    layers.py
  ui/
    main_window.py
    map_canvas.py
    color_scale_editor.py
  visualization/
    color_scale.py
    maps.py
    palettes.py
sample_data/
  gravity_sample.csv
requirements.txt
```

## Project direction

Development remains deliberately centered on **gravity and magnetic visualization and processing**. Near-term work includes stronger FFT edge handling, additional gravity filters, magnetic processing, profiles, spectrum/depth tools, Euler solutions, map/export improvements, and Windows packaging.

Advanced modelling or inversion may be explored separately in the future, but it is not the primary purpose of GeoPotential Mapper.

## License

GeoPotential Mapper is free and open-source software licensed under the **GNU General Public License v3.0 (GPL-3.0)**. You may use, study, modify, and redistribute the software under the terms of the GPL-3.0. Distributed modified/derivative versions covered by the GPL must preserve the applicable license and source-code obligations.

See the [`LICENSE`](LICENSE) file for the complete license terms.
