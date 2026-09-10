# GeoPotential Mapper

Open-source gravity and magnetic processing, mapping and inversion toolkit powered by Python and pyGIMLi.

## V1 foundation

The first version provides:

- CSV / XYZ / TXT / DAT import
- gravity and magnetic survey modes
- automatic and manual column mapping
- basic QC for invalid values and duplicate coordinates
- linear, nearest-neighbour and cubic interpolation
- interactive anomaly contour maps with station overlays
- CSV grid export
- pyGIMLi environment detection

## Run locally

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
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal, normally `http://localhost:8501`.

## Project structure

```text
app.py
geopotential/
  io/
    importer.py
  processing/
    gridding.py
  visualization/
    maps.py
sample_data/
  gravity_sample.csv
requirements.txt
```

## Roadmap

Next stages will add Bouguer/regional/residual processing, gravity and magnetic filters and derivatives, profiles and spectral analysis, Euler solutions, and finally 2D/3D forward modelling and inversion with pyGIMLi.
