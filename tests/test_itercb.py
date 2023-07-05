"""Tests for the Iteration Callback Function."""

import numpy as np
import pytest

from lmfit.lineshapes import gaussian
from lmfit.minimizer import Minimizer
from lmfit.models import GaussianModel, LinearModel

try:
    import numdifftools  # noqa: F401
    calc_covar_options = [False, True]
except ImportError:
    calc_covar_options = [False]


np.random.seed(7)
x = np.linspace(0, 20, 401)
y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
y -= 0.20*x + 3.333 + np.random.normal(scale=0.23, size=len(x))
mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')


def residual(pars, x, data):
    parvals = pars.valuesdict()
    gauss = gaussian(x, parvals['peak_amplitude'], parvals['peak_center'],
                     parvals['peak_sigma'])
    linear = parvals['bkg_slope']*x + parvals['bkg_intercept']
    return data - gauss - linear


pars = mod.make_params(peak_amplitude=21.0, peak_center=7.0,
                       peak_sigma=2.0, bkg_intercept=2, bkg_slope=0.0)

# set bounds for use with 'differential_evolution' and 'brute'
pars['bkg_intercept'].set(min=0, max=10)
pars['bkg_slope'].set(min=-5, max=5)
pars['peak_amplitude'].set(min=20, max=25)
pars['peak_center'].set(min=5, max=10)
pars['peak_sigma'].set(min=0.5, max=2)


def per_iteration(pars, iteration, resid, *args, **kws):
    """Iteration callback, will abort at iteration 23."""
    return iteration == 17


fitmethods = ['ampgo', 'brute', 'basinhopping', 'differential_evolution',
              'leastsq', 'least_squares', 'nelder', 'shgo', 'dual_annealing']


@pytest.mark.parametrize("calc_covar", calc_covar_options)
@pytest.mark.parametrize("method", fitmethods)
def test_itercb_model_class(method, calc_covar):
    """Test the iteration callback for all solvers."""
    out = mod.fit(y, pars, x=x, method=method, iter_cb=per_iteration,
                  calc_covar=calc_covar)

    assert out.nfev == 17
    assert out.aborted
    assert not out.errorbars
    assert not out.success


@pytest.mark.parametrize("calc_covar", calc_covar_options)
@pytest.mark.parametrize("method", fitmethods)
def test_itercb_minimizer_class(method, calc_covar):
    """Test the iteration callback for all solvers."""
    mini = Minimizer(residual, pars, fcn_args=(x, y), iter_cb=per_iteration,
                     calc_covar=calc_covar)
    out = mini.minimize(method=method)
    assert out.nfev == 17
    assert out.aborted
    assert not out.errorbars
    assert not out.success
    if method not in ('nelder', 'differential_evolution'):
        assert mini._abort


fitmethods = ['leastsq', 'least_squares']


@pytest.mark.parametrize("method", fitmethods)
def test_itercb_reset_abort(method):
    """Regression test for GH Issue #756.

    Make sure that ``self._abort`` is reset to ``False`` at the start of each
    fit.

    """
    if method in ('nelder', 'differential_evolution'):
        pytest.xfail("scalar_minimizers behave differently, but shouldn't!!")

    must_stop = True

    def callback(*args, **kwargs):
        return must_stop

    # perform minimization with ``iter_cb``
    out_model = mod.fit(y, pars, x=x, method=method, iter_cb=callback)

    mini = Minimizer(residual, pars, fcn_args=(x, y), iter_cb=callback)
    out_minimizer = mini.minimize(method=method)

    assert out_model.aborted is must_stop
    assert out_model.errorbars is not must_stop
    assert out_model.success is not must_stop
    assert out_minimizer.aborted is must_stop
    assert out_minimizer.errorbars is not must_stop
    assert out_minimizer.success is not must_stop
    assert mini._abort is must_stop

    # perform another minimization now without ``iter_cb``
    must_stop = False
    out_minimizer_no_callback = mini.minimize(method=method)
    assert out_minimizer_no_callback.aborted is must_stop
    assert out_minimizer_no_callback.errorbars is not must_stop
    assert out_minimizer_no_callback.success is not must_stop
    assert mini._abort is must_stop

    # reset to mini._abort to False and call the optimization method directly
    func = getattr(mini, method)
    out_no_callback = func()
    assert out_no_callback.aborted is must_stop
    assert out_no_callback.errorbars is not must_stop
    assert out_no_callback.success is not must_stop
    assert mini._abort is must_stop
