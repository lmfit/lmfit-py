"""Tests for the basinhopping minimization algorithm."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import __version__ as scipy_version
from scipy.optimize import basinhopping

import lmfit


def test_basinhopping_lmfit_vs_scipy():
    """Test basinhopping in lmfit versus scipy."""
    # SciPy
    def func(x):
        return np.cos(14.5*x - 0.3) + (x+0.2) * x

    minimizer_kwargs = {'method': 'L-BFGS-B'}
    x0 = [1.]

    ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, seed=7)

    # lmfit
    def residual(params):
        x = params['x'].value
        return np.cos(14.5*x - 0.3) + (x+0.2) * x

    pars = lmfit.Parameters()
    pars.add_many(('x', 1.))
    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    mini = lmfit.Minimizer(residual, pars)
    out = mini.minimize(method='basinhopping', **kws)

    assert_allclose(out.residual, ret.fun)
    assert_allclose(out.params['x'].value, ret.x, rtol=1e-5)


def test_basinhopping_2d_lmfit_vs_scipy():
    """Test basinhopping in lmfit versus scipy."""
    # SciPy
    def func2d(x):
        return np.cos(14.5*x[0] - 0.3) + (x[1]+0.2) * x[1] + (x[0]+0.2) * x[0]

    minimizer_kwargs = {'method': 'L-BFGS-B'}
    x0 = [1.0, 1.0]

    ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs, seed=7)

    # lmfit
    def residual_2d(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.cos(14.5*x0 - 0.3) + (x1+0.2) * x1 + (x0+0.2) * x0

    pars = lmfit.Parameters()
    pars.add_many(('x0', 1.), ('x1', 1.))

    mini = lmfit.Minimizer(residual_2d, pars)
    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    out = mini.minimize(method='basinhopping', **kws)

    assert_allclose(out.residual, ret.fun)
    assert_allclose(out.params['x0'].value, ret.x[0], rtol=1e-5)
    assert_allclose(out.params['x1'].value, ret.x[1], rtol=1e-5)

    # FIXME: update when SciPy requirement is >= 1.8
    if int(scipy_version.split('.')[1]) >= 8:
        assert 'target_accept_rate' in out.call_kws
        assert 'stepwise_factor' in out.call_kws


def test_basinhopping_Alpine02(minimizer_Alpine02):
    """Test basinhopping on Alpine02 function."""
    global_optimum = [7.91705268, 4.81584232]
    fglob = -6.12950

    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    out = minimizer_Alpine02.minimize(method='basinhopping', **kws)
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert out.method == 'basinhopping'


def test_basinhopping_bounds(minimizer_Alpine02):
    """Test basinhopping algorithm with bounds."""
    # change boundaries of parameters
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    out = minimizer_Alpine02.minimize(params=pars_bounds,
                                      method='basinhopping', **kws)
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5


def test_basinhopping_solver_options(minimizer_Alpine02):
    """Test basinhopping algorithm, pass incorrect options to solver."""
    # use minimizer_kwargs to pass an invalid method for local solver to
    # scipy.basinhopping
    kws = {'minimizer_kwargs': {'method': 'unknown'}}
    with pytest.raises(ValueError, match=r'Unknown solver'):
        minimizer_Alpine02.minimize(method='basinhopping', **kws)

    # pass an incorrect value for niter to scipy.basinhopping
    kws = {'niter': 'string'}
    with pytest.raises(TypeError):
        minimizer_Alpine02.minimize(method='basinhopping', **kws)
