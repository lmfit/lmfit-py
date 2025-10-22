"""Tests for the SHGO global minimization algorithm."""

import numpy as np
from numpy.testing import assert_allclose
import scipy

import lmfit


def eggholder(x):
    return (-(x[1] + 47.0) * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0)))))


def eggholder_lmfit(params):
    x0 = params['x0'].value
    x1 = params['x1'].value

    return (-(x1 + 47.0) * np.sin(np.sqrt(abs(x0/2.0 + (x1 + 47.0))))
            - x0 * np.sin(np.sqrt(abs(x0 - (x1 + 47.0)))))


def test_direct_scipy_vs_lmfit():
    """Test DIRECT algorithm in lmfit versus SciPy."""
    bounds = [(-512, 512), (-512, 512)]
    result_scipy = scipy.optimize.direct(eggholder, bounds)

    pars = lmfit.Parameters()
    pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
    mini = lmfit.Minimizer(eggholder_lmfit, pars)
    result = mini.minimize(method='direct')

    assert_allclose(result_scipy.fun, result.residual)
    assert_allclose(result_scipy.x, result.direct_x)
    assert_allclose(result_scipy.nfev, result.direct_nfev)


# correct result for Alpine02 function
global_optimum = [7.91705268, 4.81584232]
fglob = -6.12950


def test_direct_Alpine02(minimizer_Alpine02):
    """Test DIRECT algorithm on Alpine02 function."""
    out = minimizer_Alpine02.minimize(method='direct')

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out.direct_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out.direct_x), max(global_optimum), rtol=1e-3)
    assert out.method == 'direct'


def test_direct_bounds(minimizer_Alpine02):
    """Test DIRECT algorithm with bounds."""
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    out = minimizer_Alpine02.minimize(params=pars_bounds, method='direct')
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5


def styblinski_tang(pos):
    x, y = pos
    return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)


def styblinkski_tang_lmfit(params):
    x = params['x'].value
    y = params['y'].value

    return 0.5 * (x**4 - 16*x**2 + 5*x + y**4 - 16*y**2 + 5*y)


def test_direct_scipy_docstring_example():
    """Test DIRECT algorithm on example from docstring."""
    bounds = scipy.optimize.Bounds([-4., -4.], [4., 4.])
    result_scipy = scipy.optimize.direct(styblinski_tang, bounds)

    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x', 1., True, -4., 4.),
                         ('y', 1., True, -4., 4.))

    mini = lmfit.Minimizer(styblinkski_tang_lmfit, pars_bounds)
    result = mini.minimize(params=pars_bounds, method='direct')

    # correct results - see scipy.optimize.direct docstring
    global_xy = [-2.90321597, -2.90321597]
    fglob = -78.3323279095383
    funevals = 2011

    assert_allclose(result_scipy.fun, fglob)
    assert_allclose(result_scipy.fun, result.residual)

    assert_allclose(result_scipy.x, global_xy)
    assert_allclose(result_scipy.x, result.direct_x)

    assert_allclose(result_scipy.nfev, funevals)
    assert_allclose(result_scipy.nfev, result.direct_nfev)

    # specify ``len_tol`` to stop minimization earlier
    global_xy_len_tol = [-2.9044353, -2.9044353]
    fglob_len_tol = -78.33230330754142
    funevals_len_tol = 207

    result_scipy_len_tol = scipy.optimize.direct(styblinski_tang, bounds,
                                                 len_tol=1e-3)
    assert_allclose(result_scipy_len_tol.fun, fglob_len_tol)
    assert_allclose(result_scipy_len_tol.x, global_xy_len_tol)
    assert_allclose(result_scipy_len_tol.nfev, funevals_len_tol)

    result = mini.minimize(params=pars_bounds, method="direct", len_tol=1e-3)
    assert_allclose(result.call_kws["len_tol"], 1.e-3)
