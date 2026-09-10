import io
import pandas as pd


def read_table(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), sep=None, engine='python')


def guess_columns(columns):
    cols = {str(c).lower().strip(): c for c in columns}
    aliases = {
        'x': ['x', 'easting', 'east', 'utm_x', 'longitude', 'lon'],
        'y': ['y', 'northing', 'north', 'utm_y', 'latitude', 'lat'],
        'z': ['z', 'elevation', 'altitude', 'height', 'rl'],
        'gravity': ['gravity', 'bouguer', 'bouguer_anomaly', 'g', 'mgal'],
        'magnetic': ['magnetic', 'tmi', 'total_field', 'total_magnetic_intensity', 'nt'],
    }
    return {
        key: next((cols[name] for name in names if name in cols), None)
        for key, names in aliases.items()
    }
