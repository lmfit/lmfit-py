"""Tests for the SHGO global minimization algorithm."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
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


def test_shgo_scipy_vs_lmfit():
    """Test SHGO algorithm in lmfit versus SciPy."""
    bounds = [(-512, 512), (-512, 512)]
    result_scipy = scipy.optimize.shgo(eggholder, bounds, n=30,
                                       sampling_method='sobol')

    pars = lmfit.Parameters()
    pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
    mini = lmfit.Minimizer(eggholder_lmfit, pars)
    result = mini.minimize(method='shgo', n=30, sampling_method='sobol')
    out_x = np.array([result.params['x0'].value, result.params['x1'].value])

    assert_allclose(result_scipy.fun, result.residual)
    assert_allclose(result_scipy.funl, result.shgo_funl)
    assert_allclose(result_scipy.xl, result.shgo_xl)
    assert_allclose(result.shgo_x, out_x)


def test_shgo_scipy_vs_lmfit_2():
    """Test SHGO algorithm in lmfit versus SciPy."""
    bounds = [(-512, 512), (-512, 512)]
    result_scipy = scipy.optimize.shgo(eggholder, bounds, n=60, iters=5,
                                       sampling_method='sobol')

    pars = lmfit.Parameters()
    pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
    mini = lmfit.Minimizer(eggholder_lmfit, pars)
    result = mini.minimize(method='shgo', n=60, iters=5,
                           sampling_method='sobol')
    assert_allclose(result_scipy.fun, result.residual)
    assert_allclose(result_scipy.xl, result.shgo_xl)
    assert_allclose(result_scipy.funl, result.shgo_funl)


# correct result for Alpine02 function
global_optimum = [7.91705268, 4.81584232]
fglob = -6.12950


def test_shgo_simplicial_Alpine02(minimizer_Alpine02):
    """Test SHGO algorithm on Alpine02 function."""
    # sampling_method 'simplicial' fails with iters=1
    out = minimizer_Alpine02.minimize(method='shgo', iters=5)
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert out.method == 'shgo'


def test_shgo_sobol_Alpine02(minimizer_Alpine02):
    """Test SHGO algorithm on Alpine02 function."""
    out = minimizer_Alpine02.minimize(method='shgo', sampling_method='sobol')
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)

    # FIXME: update when SciPy requirement is >= 1.7
    if int(scipy_version.split('.')[1]) >= 7:
        assert out.call_kws['n'] is None
    else:
        assert out.call_kws['n'] == 100


def test_shgo_bounds(minimizer_Alpine02):
    """Test SHGO algorithm with bounds."""
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    out = minimizer_Alpine02.minimize(params=pars_bounds, method='shgo')
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5


def abandoned_test_shgo_disp_true(minimizer_Alpine02, capsys):
    """Test SHGO algorithm with disp is True."""
    kws = {'disp': True}
    minimizer_Alpine02.minimize(method='shgo', options=kws)
    captured = capsys.readouterr()
    assert 'Splitting first generation' in captured.out


def test_shgo_local_solver(minimizer_Alpine02):
    """Test SHGO algorithm with local solver."""
    min_kws = {'method': 'unknown'}
    with pytest.raises(KeyError, match=r'unknown'):
        minimizer_Alpine02.minimize(method='shgo', minimizer_kwargs=min_kws)
