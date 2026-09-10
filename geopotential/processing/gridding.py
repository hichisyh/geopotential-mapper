import numpy as np
from scipy.interpolate import RBFInterpolator, griddata


def clean_xyz(df, x_col, y_col, value_col):
    work = df[[x_col, y_col, value_col]].copy()
    for col in work.columns:
        work[col] = np.asarray(work[col], dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.groupby([x_col, y_col], as_index=False)[value_col].mean()
    return work


def _grid_axes(x, y, nx, ny):
    gx = np.linspace(float(np.min(x)), float(np.max(x)), int(nx))
    gy = np.linspace(float(np.min(y)), float(np.max(y)), int(ny))
    return gx, gy, np.meshgrid(gx, gy)


def _scipy_grid(x, y, z, xx, yy, method):
    return griddata((x, y), z, (xx, yy), method=method)


def _ordinary_kriging(x, y, z, gx, gy, variogram_model='spherical'):
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as exc:
        raise ImportError('Ordinary Kriging requires PyKrige. Install with: pip install pykrige') from exc

    model = OrdinaryKriging(
        x,
        y,
        z,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
        coordinates_type='euclidean',
    )
    zz, variance = model.execute('grid', gx, gy)
    return np.asarray(zz, dtype=float), np.asarray(variance, dtype=float)


def _minimum_curvature(x, y, z, xx, yy, smoothing=0.0):
    """Minimum-curvature-style interpolation using a thin-plate spline.

    A thin-plate spline minimizes bending energy and is a close open-source
    analogue to classic minimum-curvature gridding. It is not intended to be
    numerically identical to Geosoft/Oasis montaj's proprietary implementation.
    """
    points = np.column_stack([x, y])
    targets = np.column_stack([xx.ravel(), yy.ravel()])

    # Scale coordinates to improve numerical conditioning for UTM-sized values.
    origin = points.mean(axis=0)
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1.0
    points_n = (points - origin) / scale
    targets_n = (targets - origin) / scale

    rbf = RBFInterpolator(
        points_n,
        z,
        kernel='thin_plate_spline',
        smoothing=float(smoothing),
    )
    return rbf(targets_n).reshape(xx.shape)


def grid_scattered(
    df,
    x_col,
    y_col,
    value_col,
    method='linear',
    nx=150,
    ny=150,
    variogram_model='spherical',
    smoothing=0.0,
):
    work = clean_xyz(df, x_col, y_col, value_col)
    if len(work) < 3:
        raise ValueError('At least three valid stations are required.')

    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)
    z = work[value_col].to_numpy(dtype=float)

    gx, gy, (xx, yy) = _grid_axes(x, y, nx, ny)
    method_key = method.strip().lower().replace('_', ' ')
    variance = None

    if method_key in {'linear', 'nearest', 'cubic'}:
        zz = _scipy_grid(x, y, z, xx, yy, method_key)
    elif method_key in {'kriging', 'ordinary kriging'}:
        zz, variance = _ordinary_kriging(x, y, z, gx, gy, variogram_model)
    elif method_key in {'minimum curvature', 'min curvature', 'minimum-curvature'}:
        zz = _minimum_curvature(x, y, z, xx, yy, smoothing=smoothing)
    else:
        raise ValueError(f'Unknown gridding method: {method}')

    return work, xx, yy, zz, variance
