"""Tests for maximum number of function evaluations (max_nfev)."""

import numpy as np
import pytest

from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, LinearModel
from lmfit.minimizer import Minimizer


nvarys = 5
methods = ['leastsq', 'least_squares', 'nelder', 'brute', 'ampgo',
           'basinopping', 'differential_evolution', 'shgo', 'dual_annealing']


@pytest.fixture
def modelGaussian():
    """Return data, parameters and Model class for Gaussian + Linear model."""
    # generate data with random noise added
    np.random.seed(7)
    x = np.linspace(0, 20, 401)
    y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
    y -= 0.20*x + 3.333 + np.random.normal(scale=0.23, size=len(x))

    mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')

    # make parameters and set bounds
    pars = mod.make_params(peak_amplitude=21.0, peak_center=7.0,
                           peak_sigma=2.0, bkg_intercept=2, bkg_slope=0.0)

    pars['bkg_intercept'].set(min=0, max=10, brute_step=5.0)
    pars['bkg_slope'].set(min=-5, max=5, brute_step=5.0)
    pars['peak_amplitude'].set(min=20, max=25, brute_step=2.5)
    pars['peak_center'].set(min=5, max=10, brute_step=2.5)
    pars['peak_sigma'].set(min=0.5, max=2, brute_step=0.5)

    return x, y, mod, pars


@pytest.fixture
def minimizerGaussian(modelGaussian):
    """Return a Mininizer class for the Gaussian + Linear model."""
    x, y, _, pars = modelGaussian

    def residual(params, x, y):
        pars = params.valuesdict()
        model = (gaussian(x, pars['peak_amplitude'], pars['peak_center'],
                          pars['peak_sigma']) +
                 pars['bkg_intercept'] + x*pars['bkg_slope'])
        return y - model

    mini = Minimizer(residual, pars, fcn_args=(x, y))

    return mini


@pytest.mark.parametrize("method", methods)
def test_max_nfev_Minimizer(minimizerGaussian, method):
    """Test the max_nfev argument for all solvers using Minimizer interface."""
    if method in ('brute', 'basinhopping'):
        pytest.xfail('max_nfev not yet supported in {}'.format(method))  # FIXME

    result = minimizerGaussian.minimize(method=method, max_nfev=10)
    assert minimizerGaussian.max_nfev == 10
    assert result.nfev < 15
    assert result.aborted
    assert not result.errorbars
    assert not result.success


@pytest.mark.parametrize("method", methods)
def test_max_nfev_Model(modelGaussian, minimizerGaussian, method):
    """Test the max_nfev argument for all solvers using Model interfce."""
    x, y, mod, pars = modelGaussian
    out = mod.fit(y, pars, x=x, method=method, max_nfev=10)

    assert out.max_nfev == 10
    assert out.nfev < 15
    assert out.aborted
    assert not out.errorbars
    assert not out.success


@pytest.mark.parametrize("method, default_max_nfev",
                         [('leastsq', 2000*(nvarys+1)),
                          ('least_squares', 1000*(nvarys+1)),
                          ('nelder', 1000*(nvarys+1)),
                          ('brute', np.inf),
                          ('ampgo', 1000000*(nvarys+1)),
                          ('basinhopping', 2000*(nvarys+1)),
                          ('differential_evolution', 1000*(nvarys+1)),
                          ('shgo', 1000000*(nvarys+1)),
                          ('dual_annealing', 1.e7)])
def test_default_max_nfev(modelGaussian, minimizerGaussian, method,
                          default_max_nfev):
    """Test the default values when setting max_nfev=None."""
    if method in ('brute', 'basinhopping'):
        pytest.xfail('max_nfev not yet supported in {}'.format(method))  # FIXME

    x, y, mod, pars = modelGaussian
    result = mod.fit(y, pars, x=x, method=method, max_nfev=None)
    assert result.max_nfev == default_max_nfev

    _ = minimizerGaussian.minimize(method=method, max_nfev=None)
    assert minimizerGaussian.max_nfev == default_max_nfev
