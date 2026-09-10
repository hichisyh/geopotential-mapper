import numpy as np
import pandas as pd
import streamlit as st

from geopotential.io.importer import read_table, guess_columns
from geopotential.processing.gridding import grid_scattered
from geopotential.visualization.maps import anomaly_map
from geopotential.visualization.palettes import PALETTES, get_palette

st.set_page_config(page_title='GeoPotential Mapper', layout='wide')
st.title('GeoPotential Mapper')
st.caption('Gravity and magnetic mapping toolkit powered by Python, SciPy, PyKrige and pyGIMLi.')

with st.sidebar:
    survey_type = st.radio('Survey type', ['Gravity', 'Magnetics'])
    uploaded = st.file_uploader('Upload survey data', type=['csv', 'xyz', 'txt', 'dat'])

if uploaded is None:
    st.info('Upload a CSV, XYZ, TXT or DAT file to begin.')
    st.stop()

try:
    df = read_table(uploaded)
except Exception as exc:
    st.error(f'Could not read this file: {exc}')
    st.stop()

st.subheader('1. Data preview')
st.dataframe(df.head(25), use_container_width=True)

guesses = guess_columns(df.columns)
columns = list(df.columns)

def index_for(name):
    guessed = guesses.get(name)
    return columns.index(guessed) if guessed in columns else 0

c1, c2, c3 = st.columns(3)
with c1:
    x_col = st.selectbox('X / Easting', columns, index=index_for('x'))
with c2:
    y_col = st.selectbox('Y / Northing', columns, index=index_for('y'))
with c3:
    value_key = 'gravity' if survey_type == 'Gravity' else 'magnetic'
    value_col = st.selectbox('Measured/anomaly value', columns, index=index_for(value_key))

st.subheader('2. Quality control')
qc = df[[x_col, y_col, value_col]].apply(pd.to_numeric, errors='coerce')
missing = int(qc.isna().any(axis=1).sum())
duplicates = int(qc.duplicated(subset=[x_col, y_col]).sum())
valid = int(len(qc) - missing)

m1, m2, m3, m4 = st.columns(4)
m1.metric('Rows', len(df))
m2.metric('Valid rows', valid)
m3.metric('Rows with missing/non-numeric values', missing)
m4.metric('Duplicate coordinates', duplicates)

if valid:
    st.dataframe(qc[value_col].describe().to_frame('Value statistics'), use_container_width=True)

st.subheader('3. Gridding & map style')
g1, g2, g3, g4 = st.columns(4)
with g1:
    method = st.selectbox('Interpolation', ['Linear', 'Nearest', 'Cubic', 'Ordinary Kriging', 'Minimum Curvature'])
with g2:
    nx = st.slider('Grid cells X', 40, 400, 150, 10)
with g3:
    ny = st.slider('Grid cells Y', 40, 400, 150, 10)
with g4:
    palette_name = st.selectbox('Color palette', list(PALETTES.keys()))

reverse_palette = st.checkbox('Reverse color palette', value=False)
variogram = 'spherical'
smoothing = 0.0
if method == 'Ordinary Kriging':
    variogram = st.selectbox('Kriging variogram', ['spherical', 'exponential', 'gaussian', 'linear', 'power'])
elif method == 'Minimum Curvature':
    smoothing = st.number_input('Minimum-curvature smoothing', min_value=0.0, value=0.0, step=0.01)

try:
    work, xx, yy, zz, variance = grid_scattered(
        df,
        x_col,
        y_col,
        value_col,
        method,
        nx,
        ny,
        variogram_model=variogram,
        smoothing=smoothing,
    )
except Exception as exc:
    st.error(f'Gridding failed: {exc}')
    st.stop()

units = 'mGal' if survey_type == 'Gravity' else 'nT'
title = f'{survey_type} anomaly map — {value_col} ({units})'
fig = anomaly_map(
    work,
    x_col,
    y_col,
    value_col,
    xx,
    yy,
    zz,
    title,
    colorscale=get_palette(palette_name, reverse_palette),
)
st.plotly_chart(fig, use_container_width=True)

if variance is not None:
    st.caption('Ordinary Kriging variance has been calculated and will be exposed as a dedicated uncertainty layer in a later UI update.')

st.subheader('4. Export interpolated grid')
export = pd.DataFrame({
    'x': xx.ravel(),
    'y': yy.ravel(),
    value_col: np.asarray(zz).ravel(),
}).dropna()
st.download_button(
    'Download grid as CSV',
    data=export.to_csv(index=False),
    file_name=f'{survey_type.lower()}_grid.csv',
    mime='text/csv',
)

st.divider()
try:
    import pygimli as pg
    st.success(f'pyGIMLi detected: {pg.__version__}. Inversion engine will be connected in a later stage.')
except Exception:
    st.warning('pyGIMLi is not available in this Python environment yet. Mapping still works without it.')
