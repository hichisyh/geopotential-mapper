import numpy as np
from scipy.interpolate import griddata


def _spacing(xx, yy):
    dx = float(np.nanmedian(np.diff(xx[0, :])))
    dy = float(np.nanmedian(np.diff(yy[:, 0])))
    if dx == 0 or dy == 0:
        raise ValueError('Grid spacing must be non-zero.')
    return abs(dx), abs(dy)


def _fill_nans(xx, yy, zz):
    data = np.asarray(zz, dtype=float)
    finite = np.isfinite(data)
    if finite.all():
        return data.copy(), finite
    if finite.sum() < 3:
        raise ValueError('At least three finite grid cells are required.')

    points = np.column_stack([xx[finite], yy[finite]])
    values = data[finite]
    targets = np.column_stack([xx.ravel(), yy.ravel()])
    filled = griddata(points, values, targets, method='nearest').reshape(data.shape)
    return filled, finite


def _restore_mask(result, finite_mask):
    out = np.asarray(result, dtype=float).copy()
    out[~finite_mask] = np.nan
    return out


def polynomial_regional(xx, yy, zz, degree=1):
    degree = int(degree)
    if degree not in (1, 2, 3):
        raise ValueError('Polynomial regional degree must be 1, 2 or 3.')

    data = np.asarray(zz, dtype=float)
    mask = np.isfinite(data)
    if mask.sum() < 6:
        raise ValueError('Not enough finite cells for polynomial regional fitting.')

    x = xx[mask].astype(float)
    y = yy[mask].astype(float)
    z = data[mask]
    x0, y0 = np.mean(x), np.mean(y)
    xs = np.std(x) or 1.0
    ys = np.std(y) or 1.0
    xn = (x - x0) / xs
    yn = (y - y0) / ys

    terms = [np.ones_like(xn), xn, yn]
    if degree >= 2:
        terms += [xn**2, xn * yn, yn**2]
    if degree >= 3:
        terms += [xn**3, xn**2 * yn, xn * yn**2, yn**3]

    design = np.column_stack(terms)
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)

    X = (xx - x0) / xs
    Y = (yy - y0) / ys
    grid_terms = [np.ones_like(X), X, Y]
    if degree >= 2:
        grid_terms += [X**2, X * Y, Y**2]
    if degree >= 3:
        grid_terms += [X**3, X**2 * Y, X * Y**2, Y**3]

    regional = np.zeros_like(X, dtype=float)
    for c, term in zip(coeffs, grid_terms):
        regional += c * term
    regional[~mask] = np.nan
    return regional


def residual_from_regional(zz, regional):
    return np.asarray(zz, dtype=float) - np.asarray(regional, dtype=float)


def _fft_operator(xx, yy, zz, multiplier):
    filled, finite_mask = _fill_nans(xx, yy, zz)
    dx, dy = _spacing(xx, yy)
    ny, nx = filled.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    transformed = np.fft.fft2(filled)
    result = np.real(np.fft.ifft2(transformed * multiplier(KX, KY, K)))
    return _restore_mask(result, finite_mask)


def upward_continuation(xx, yy, zz, height):
    height = float(height)
    if height < 0:
        raise ValueError('Continuation height must be zero or positive.')
    return _fft_operator(xx, yy, zz, lambda _kx, _ky, k: np.exp(-height * k))


def vertical_derivative(xx, yy, zz):
    return _fft_operator(xx, yy, zz, lambda _kx, _ky, k: k)


def horizontal_derivatives(xx, yy, zz):
    dx_grid = _fft_operator(xx, yy, zz, lambda kx, _ky, _k: 1j * kx)
    dy_grid = _fft_operator(xx, yy, zz, lambda _kx, ky, _k: 1j * ky)
    return dx_grid, dy_grid


def total_horizontal_gradient(xx, yy, zz):
    dx_grid, dy_grid = horizontal_derivatives(xx, yy, zz)
    return np.sqrt(dx_grid**2 + dy_grid**2)
