import numpy as np

from geopotential.processing.gravity import (
    polynomial_regional,
    residual_from_regional,
    total_horizontal_gradient,
    upward_continuation,
    vertical_derivative,
)


def make_grid(nx=41, ny=31, dx=25.0, dy=25.0):
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    return np.meshgrid(x, y)


def test_polynomial_regional_recovers_plane():
    xx, yy = make_grid()
    zz = 10.0 + 0.002 * xx - 0.003 * yy
    regional = polynomial_regional(xx, yy, zz, degree=1)
    assert np.allclose(regional, zz, atol=1e-9)


def test_residual_removes_plane():
    xx, yy = make_grid()
    zz = 5.0 + 0.001 * xx + 0.002 * yy
    regional = polynomial_regional(xx, yy, zz, degree=1)
    residual = residual_from_regional(zz, regional)
    assert np.nanmax(np.abs(residual)) < 1e-9


def test_upward_continuation_preserves_shape_and_reduces_variance():
    xx, yy = make_grid()
    zz = np.sin(xx / 120.0) + 0.5 * np.cos(yy / 90.0)
    up = upward_continuation(xx, yy, zz, height=100.0)
    assert up.shape == zz.shape
    assert np.nanstd(up) < np.nanstd(zz)


def test_derivative_outputs_have_expected_shape():
    xx, yy = make_grid()
    zz = np.sin(xx / 150.0) * np.cos(yy / 120.0)
    vd = vertical_derivative(xx, yy, zz)
    thg = total_horizontal_gradient(xx, yy, zz)
    assert vd.shape == zz.shape
    assert thg.shape == zz.shape
    assert np.nanmin(thg) >= 0.0
