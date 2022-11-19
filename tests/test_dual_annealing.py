"""Tests for the Dual Annealing algorithm."""

import numpy as np
from numpy.testing import assert_allclose
import scipy
from scipy import __version__ as scipy_version

import lmfit


def eggholder(x):
    return (-(x[1] + 47.0) * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0)))))


def eggholder_lmfit(params):
    x0 = params['x0'].value
    x1 = params['x1'].value

    return (-(x1 + 47.0) * np.sin(np.sqrt(abs(x0/2.0 + (x1 + 47.0))))
            - x0 * np.sin(np.sqrt(abs(x0 - (x1 + 47.0)))))


def test_da_scipy_vs_lmfit():
    """Test DA algorithm in lmfit versus SciPy."""
    bounds = [(-512, 512), (-512, 512)]
    result_scipy = scipy.optimize.dual_annealing(eggholder, bounds, seed=7)

    pars = lmfit.Parameters()
    pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
    mini = lmfit.Minimizer(eggholder_lmfit, pars)
    result = mini.minimize(method='dual_annealing', seed=7)
    out_x = np.array([result.params['x0'].value, result.params['x1'].value])

    assert_allclose(result_scipy.fun, result.residual)
    assert_allclose(result_scipy.x, out_x)


# TODO: add scipy example from docstring after the reproducibility issue in
# https://github.com/scipy/scipy/issues/9732 is resolved.

# correct result for Alpine02 function
global_optimum = [7.91705268, 4.81584232]
fglob = -6.12950


def test_da_Alpine02(minimizer_Alpine02):
    """Test dual_annealing algorithm on Alpine02 function."""
    out = minimizer_Alpine02.minimize(method='dual_annealing')
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert out.method == 'dual_annealing'

    # FIXME: update when SciPy requirement is >= 1.8
    # ``local_search_options`` deprecated in favor of ``minimizer_kwargs``
    if int(scipy_version.split('.')[1]) >= 8:
        assert 'minimizer_kwargs' in out.call_kws
    else:
        assert 'local_search_options' in out.call_kws


def test_da_bounds(minimizer_Alpine02):
    """Test dual_annealing algorithm with bounds."""
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    out = minimizer_Alpine02.minimize(params=pars_bounds,
                                      method='dual_annealing')
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5
