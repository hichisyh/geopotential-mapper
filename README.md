# GeoPotential Mapper

Open-source gravity and magnetic processing, mapping and inversion toolkit powered by Python and pyGIMLi.

## Current V1 foundation

The project now provides two front ends that share the same processing code:

- **Desktop Python app** with PySide6/Qt (`desktop_app.py`)
- **Streamlit web prototype** (`app.py`)

Current capabilities:

- CSV / XYZ / TXT / DAT import
- gravity and magnetic survey modes
- automatic/manual column mapping
- basic QC
- Linear interpolation
- Nearest-neighbour interpolation
- Cubic interpolation
- Ordinary Kriging with selectable variogram model
- Minimum-curvature-style thin-plate spline interpolation
- geophysics-oriented color palettes
- custom low/mid/high desktop colors
- palette reversal
- manual color-value range
- configurable contour levels
- station and contour overlays
- CSV grid export
- pyGIMLi environment detection

> Note: the open minimum-curvature implementation uses a thin-plate spline, which minimizes surface bending. It is designed as an open scientific analogue and is not numerically identical to Geosoft/Oasis montaj's proprietary minimum-curvature implementation.

## Run the desktop Python application

```bash
git clone https://github.com/hichisyh/geopotential-mapper.git
cd geopotential-mapper
git checkout feature/v1-foundation
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python desktop_app.py
```

The application opens in its own Windows desktop window; no browser is required.

## Run the Streamlit prototype

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Color system

The application includes geophysics-focused palettes inspired by common potential-field display conventions rather than copies of proprietary Geosoft `.tbl` files:

- Geophysics Classic
- Gravity Diverging
- Magnetic Spectrum
- Terrain
- Grayscale
- Blue-White-Red

The desktop app also allows a custom three-color palette, reversing the palette, changing contour levels, manually setting the displayed data range, and toggling stations/contours.

## Gridding

### Ordinary Kriging

Uses PyKrige. Variogram choices currently include spherical, exponential, Gaussian, linear and power.

### Minimum Curvature

Uses SciPy's thin-plate radial basis interpolation. Future versions will expose more minimum-curvature controls such as smoothing/tension-like behavior, blanking distance and data constraints.

## Project structure

```text
app.py                  # browser-based prototype
desktop_app.py          # native PySide6 desktop GUI
geopotential/
  io/
    importer.py
  processing/
    gridding.py
  visualization/
    maps.py
    palettes.py
sample_data/
  gravity_sample.csv
requirements.txt
```

## Roadmap

Next stages:

1. Project/layer manager and saved projects
2. Bouguer, regional and residual processing
3. Upward continuation and derivatives (1VD, horizontal gradient, analytic signal, tilt)
4. Magnetic processing including RTP/RTE where scientifically appropriate
5. Profiles, power spectrum and depth estimation
6. Euler solutions and structural interpretation
7. 2D/3D forward modelling and inversion with pyGIMLi
8. Windows packaging (`.exe`) and installer
