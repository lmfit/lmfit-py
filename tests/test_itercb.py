"""Tests for the Iteration Callback Function."""
import numpy as np
import pytest

from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, LinearModel

try:
    import numdifftools
    calc_covar_options = [False, True]
except ImportError:
    calc_covar_options = [False]


np.random.seed(7)
x = np.linspace(0, 20, 401)
y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
y -= 0.20*x + 3.333 + np.random.normal(scale=0.23, size=len(x))
mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')

pars = mod.make_params(peak_amplitude=21.0, peak_center=7.0,
                       peak_sigma=2.0, bkg_intercept=2, bkg_slope=0.0)

# set bounds for use with 'differential_evolution' and 'brute'
pars['bkg_intercept'].set(min=0, max=10)
pars['bkg_slope'].set(min=-5, max=5)
pars['peak_amplitude'].set(min=20, max=25)
pars['peak_center'].set(min=5, max=10)
pars['peak_sigma'].set(min=0.5, max=2)


def per_iteration(pars, iter, resid, *args, **kws):
    """Iteration callback, will abort at iteration 23."""
    return iter == 23


@pytest.mark.parametrize("calc_covar", calc_covar_options)
@pytest.mark.parametrize("method", ['ampgo', 'brute', 'basinhopping',
                                    'differential_evolution','leastsq',
                                    'least_squares', 'nelder'])
def test_itercb(method, calc_covar):
    """Test the iteration callback for all solvers."""
    out = mod.fit(y, pars, x=x, method=method, iter_cb=per_iteration,
                  calc_covar=calc_covar)

    assert out.nfev == 23
    assert out.aborted
    assert not out.errorbars
    assert not out.success
