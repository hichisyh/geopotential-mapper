import numpy as np
from scipy.interpolate import griddata


def clean_xyz(df, x_col, y_col, value_col):
    work = df[[x_col, y_col, value_col]].copy()
    for col in work.columns:
        work[col] = np.asarray(work[col], dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.drop_duplicates(subset=[x_col, y_col])
    return work


def grid_scattered(df, x_col, y_col, value_col, method='linear', nx=150, ny=150):
    work = clean_xyz(df, x_col, y_col, value_col)
    if len(work) < 3:
        raise ValueError('At least three valid stations are required.')

    x = work[x_col].to_numpy()
    y = work[y_col].to_numpy()
    z = work[value_col].to_numpy()

    gx = np.linspace(x.min(), x.max(), int(nx))
    gy = np.linspace(y.min(), y.max(), int(ny))
    xx, yy = np.meshgrid(gx, gy)
    zz = griddata((x, y), z, (xx, yy), method=method)
    return work, xx, yy, zz
