import os

import numpy as np
from numpy import pi
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

from lmfit import Parameters, minimize
from lmfit.lineshapes import exponential
from lmfit.models import ExponentialModel, LinearModel, VoigtModel


def check(para, real_val, sig=3):
    err = abs(para.value - real_val)
    assert err < sig * para.stderr


def test_bounded_parameters():
    # create data to be fitted
    np.random.seed(1)
    x = np.linspace(0, 15, 301)
    data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
            np.random.normal(size=len(x), scale=0.2))

    # define objective function: returns the array to be minimized
    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        amp = params['amp']
        shift = params['shift']
        omega = params['omega']
        decay = params['decay']

        model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
        return model - data

    # create a set of Parameters
    params = Parameters()
    params.add('amp', value=10, min=0, max=50)
    params.add('decay', value=0.1, min=0, max=10)
    params.add('shift', value=0.0, min=-pi/2., max=pi/2.)
    params.add('omega', value=3.0, min=0, max=np.inf)

    # do fit, here with leastsq model
    result = minimize(fcn2min, params, args=(x, data),
                      epsfcn=1.e-14)

    # assert that the real parameters are found
    for para, val in zip(result.params.values(), [5, 0.025, -.1, 2]):
        check(para, val)

    # assert that the covariance matrix is correct [cf. lmfit v0.9.10]
    cov_x = np.array([
        [1.42428250e-03, 9.45395985e-06, -4.33997922e-05, 1.07362106e-05],
        [9.45395985e-06, 1.84110424e-07, -2.90588963e-07, 7.19107184e-08],
        [-4.33997922e-05, -2.90588963e-07, 9.53427031e-05, -2.37750362e-05],
        [1.07362106e-05, 7.19107184e-08, -2.37750362e-05, 9.60952336e-06]])
    assert_allclose(result.covar, cov_x, rtol=1.e-3, atol=1.e-6)

    # assert that stderr and correlations are correct [cf. lmfit v0.9.10]
    assert_almost_equal(result.params['amp'].stderr, 0.03773967, decimal=4)
    assert_almost_equal(result.params['decay'].stderr, 4.2908e-04, decimal=4)
    assert_almost_equal(result.params['shift'].stderr, 0.00976436, decimal=4)
    assert_almost_equal(result.params['omega'].stderr, 0.00309992, decimal=4)

    assert_almost_equal(result.params['amp'].correl['decay'],
                        0.5838166760743324, decimal=4)
    assert_almost_equal(result.params['amp'].correl['shift'],
                        -0.11777303073961824, decimal=4)
    assert_almost_equal(result.params['amp'].correl['omega'],
                        0.09177027400788784, decimal=4)
    assert_almost_equal(result.params['decay'].correl['shift'],
                        -0.0693579417651835, decimal=4)
    assert_almost_equal(result.params['decay'].correl['omega'],
                        0.05406342001021014, decimal=4)
    assert_almost_equal(result.params['shift'].correl['omega'],
                        -0.7854644476455469, decimal=4)


def test_bounds_expression():
    # load data to be fitted
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                   'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['amplitude'].set(min=0, max=100)
    params['center'].set(min=5, max=10)

    # do fit, here with leastsq model
    result = mod.fit(y, params, x=x, fit_kws={'epsfcn': 1.e-14})

    # assert that stderr and correlations are correct [cf. lmfit v0.9.10]
    assert_almost_equal(result.params['sigma'].stderr, 0.00368468, decimal=4)
    assert_almost_equal(result.params['center'].stderr, 0.00505496, decimal=4)
    assert_almost_equal(result.params['amplitude'].stderr, 0.13861506,
                        decimal=4)
    assert_almost_equal(result.params['gamma'].stderr, 0.00368468, decimal=4)
    assert_almost_equal(result.params['fwhm'].stderr, 0.01326862028, decimal=4)
    assert_almost_equal(result.params['height'].stderr, 0.0395990, decimal=4)

    assert_almost_equal(result.params['sigma'].correl['center'],
                        -4.6623973788006615e-05, decimal=4)
    assert_almost_equal(result.params['sigma'].correl['amplitude'],
                        0.651304091954038, decimal=4)
    assert_almost_equal(result.params['center'].correl['amplitude'],
                        -4.390334984618851e-05, decimal=4)


@pytest.mark.parametrize("fit_method", ['nelder', 'lbfgs'])
def test_numdifftools_no_bounds(fit_method):
    pytest.importorskip("numdifftools")

    np.random.seed(7)
    x = np.linspace(0, 100, num=50)
    noise = np.random.normal(scale=0.25, size=x.size)
    y = exponential(x, amplitude=5, decay=15) + noise

    mod = ExponentialModel()
    params = mod.guess(y, x=x)

    # do fit, here with leastsq model
    result = mod.fit(y, params, x=x, method='leastsq')

    result_ndt = mod.fit(y, params, x=x, method=fit_method)

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_ndt = [result_ndt.params[p].value for p in result_ndt.params.valuesdict()]
    assert_allclose(vals_ndt, vals, rtol=0.1)
    assert_allclose(result_ndt.chisqr, result.chisqr, rtol=1e-5)

    # assert that parameter uncertainties from leastsq and calculated from
    # the covariance matrix using numdifftools are very similar
    stderr = [result.params[p].stderr for p in result.params.valuesdict()]
    stderr_ndt = [result_ndt.params[p].stderr for p in result_ndt.params.valuesdict()]

    perr = np.array(stderr) / np.array(vals)
    perr_ndt = np.array(stderr_ndt) / np.array(vals_ndt)
    assert_almost_equal(perr_ndt, perr, decimal=3)

    # assert that parameter correlatations from leastsq and calculated from
    # the covariance matrix using numdifftools are very similar
    for par1 in result.var_names:
        cor = [result.params[par1].correl[par2] for par2 in
               result.params[par1].correl.keys()]
        cor_ndt = [result_ndt.params[par1].correl[par2] for par2 in
                   result_ndt.params[par1].correl.keys()]
        assert_almost_equal(cor_ndt, cor, decimal=2)


@pytest.mark.parametrize("fit_method", ['nelder', 'basinhopping', 'ampgo',
                                        'shgo', 'dual_annealing'])
def test_numdifftools_with_bounds(fit_method):
    pytest.importorskip("numdifftools")

    # load data to be fitted
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                   'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['amplitude'].set(min=25, max=70)
    params['sigma'].set(max=1)
    params['center'].set(min=5, max=15)

    # do fit, here with leastsq model
    result = mod.fit(y, params, x=x, method='leastsq')

    result_ndt = mod.fit(y, params, x=x, method=fit_method)

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_ndt = [result_ndt.params[p].value for p in result_ndt.params.valuesdict()]
    assert_allclose(vals_ndt, vals, rtol=0.1)
    assert_allclose(result_ndt.chisqr, result.chisqr, rtol=1e-5)

    # assert that parameter uncertainties from leastsq and calculated from
    # the covariance matrix using numdifftools are very similar
    stderr = [result.params[p].stderr for p in result.params.valuesdict()]
    stderr_ndt = [result_ndt.params[p].stderr for p in result_ndt.params.valuesdict()]

    perr = np.array(stderr) / np.array(vals)
    perr_ndt = np.array(stderr_ndt) / np.array(vals_ndt)
    assert_almost_equal(perr_ndt, perr, decimal=3)

    # assert that parameter correlatations from leastsq and calculated from
    # the covariance matrix using numdifftools are very similar
    for par1 in result.var_names:
        cor = [result.params[par1].correl[par2] for par2 in
               result.params[par1].correl.keys()]
        cor_ndt = [result_ndt.params[par1].correl[par2] for par2 in
                   result_ndt.params[par1].correl.keys()]
        assert_almost_equal(cor_ndt, cor, decimal=2)


def test_numdifftools_calc_covar_false():
    pytest.importorskip("numdifftools")
    # load data to be fitted
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                   'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['sigma'].set(min=-np.inf)

    # do fit, with leastsq and nelder
    result = mod.fit(y, params, x=x, method='leastsq')
    result_ndt = mod.fit(y, params, x=x, method='nelder', calc_covar=False)

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_ndt = [result_ndt.params[p].value for p in result_ndt.params.valuesdict()]
    assert_allclose(vals_ndt, vals, rtol=5e-3)
    assert_allclose(result_ndt.chisqr, result.chisqr)

    assert result_ndt.covar is None
    assert result_ndt.errorbars is False


def test_final_parameter_values():
    model = LinearModel()
    params = model.make_params()
    params['intercept'].set(value=-1, min=-20, max=0)
    params['slope'].set(value=1, min=-100, max=400)

    np.random.seed(78281)
    x = np.linspace(0, 9, 10)
    y = x * 1.34 - 4.5 + np.random.normal(scale=0.05, size=x.size)

    result = model.fit(y, x=x, method='nelder', params=params)

    assert_almost_equal(result.chisqr, 0.014625543, decimal=6)
    assert_almost_equal(result.params['intercept'].value, -4.511087126, decimal=6)
    assert_almost_equal(result.params['slope'].value, 1.339685514, decimal=6)
