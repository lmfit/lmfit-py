import numpy as np
import pytest
from scipy import __version__ as scipy_version

from lmfit import Minimizer, Parameters
from lmfit.models import GaussianModel, VoigtModel


def test_scalar_minimize_neg_value():
    x0 = 3.14
    fmin = -1.1
    xtol = 0.001
    ftol = 2.0 * xtol

    def objective(pars):
        return (pars['x'] - x0) ** 2.0 + fmin

    params = Parameters()
    params.add('x', value=2*x0)

    minr = Minimizer(objective, params)
    result = minr.scalar_minimize(method='Nelder-Mead',
                                  options={'xatol': xtol, 'fatol': ftol})
    assert abs(result.params['x'].value - x0) < xtol
    assert abs(result.fun - fmin) < ftol


@pytest.mark.parametrize('method', ('leastsq', 'least_squares', 'nelder',
                                    'lbfgsb', 'powell', 'cg', 'bfgs', 'brute',
                                    'dual_annealing', 'differential_evolution',
                                    'ampgo', 'shgo', 'cobyla', 'cobyqa',
                                    'basinhopping'))
def test_aborted_solvers(method):
    # github discussion #894
    x = np.array([18.025, 18.075, 18.125, 18.175, 18.225, 18.275, 18.325, 18.375,
                  18.425, 18.475, 18.525, 18.575, 18.625, 18.675, 18.725, 18.775,
                  18.825, 18.875, 18.925, 18.975, 19.025, 19.075, 19.125, 19.175,
                  19.225, 19.275, 19.325, 19.375, 19.425, 19.475, 19.525, 19.575,
                  19.625, 19.675, 19.725, 19.775, 19.825, 19.875, 19.925, 19.975,
                  20.025, 20.075, 20.125, 20.175, 20.225, 20.275, 20.325, 20.375,
                  20.425, 20.475, 20.525, 20.575, 20.625, 20.675, 20.725, 20.775,
                  20.825, 20.875, 20.925, 20.975])

    y = np.array([43, 27, 14, 16, 20, 10, 15, 8, 3, 9, 2, 4, 8, 3, 3, 1, 2, 4,
                  5, 0, 1, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 23, 202, 824,
                  344, 279, 276, 23, 2, 7, 8, 17, 22, 3, 25, 24, 17, 99, 288,
                  172, 252, 103, 24, 20, 4, 1, 3, 1.0])

    mod = GaussianModel(prefix='p1_') + GaussianModel(prefix='p2_')

    max_nfev = 40
    if method in ('cobyla', 'cobyqa'):
        max_nfev = 30
    pars = mod.make_params(p1_amplitude={'value': 70, 'min': 0, 'max': 200},
                           p1_center={'value': 19.5, 'min': 8, 'max': 20.5},
                           p1_sigma={'value': 0.05, 'min': 0, 'max': 1},
                           p2_amplitude={'value': 40, 'min': 0, 'max': 100},
                           p2_center={'value': 20.5, 'min': 19.5, 'max': 21.5},
                           p2_sigma={'value': 0.05, 'min': 0, 'max': 1})

    result = mod.fit(y, pars, x=x, max_nfev=max_nfev, method=method)
    assert not result.success
    assert not result.errorbars
    assert result.redchi > 1000
    assert result.redchi < 90000
    assert result.nfev > max_nfev - 5
    assert result.nfev < max_nfev + 5


# FIXME: remove when SciPy requirement is >= 1.16
@pytest.mark.parametrize('method', ('leastsq', 'least_squares', 'nelder',
                                    'lbfgsb', 'powell', 'cg', 'bfgs',
                                    'differential_evolution', 'cobyla',
                                    'cobyqa'))
def test_workers_keyword_solvers(peakdata, method):
    """New solver keyword in SciPy v1.16.0."""
    x = peakdata[0]
    y = peakdata[1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['amplitude'].min = 0
    params['amplitude'].max = 1000
    params['center'].min = 0
    params['center'].max = 1000
    params['sigma'].min = 0
    params['sigma'].max = 1000

    result = mod.fit(y, params, x=x, method=method)

    if (int(scipy_version.split('.')[1]) < 16 and method != 'differential_evolution'):
        assert 'workers' not in result.call_kws

    elif (int(scipy_version.split('.')[1]) > 16 and method in
          ('least_squares', 'lbfgsb', 'bfgs', 'differential_evolution')):
        assert 'workers' in result.call_kws
        if method != 'differential_evolution':
            assert result.call_kws['workers'] is None
        else:
            assert result.call_kws['workers'] == 1
