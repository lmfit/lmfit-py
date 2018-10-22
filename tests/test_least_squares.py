import os

import numpy as np
from numpy import exp, linspace, pi, random, sign, sin
from numpy.testing import assert_allclose, assert_almost_equal

from lmfit import Minimizer, Parameters
from lmfit.models import VoigtModel
from lmfit_testutils import assert_paramval


def test_bounds():
    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.4321)
    p_true.add('shift', value=0.12345)
    p_true.add('decay', value=0.01000)

    def residual(pars, x, data=None):
        amp = pars['amp']
        per = pars['period']
        shift = pars['shift']
        decay = pars['decay']

        if abs(shift) > pi/2:
            shift = shift - sign(shift)*pi

        model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
        if data is None:
            return model
        return (model - data)

    n = 1500
    xmin = 0.
    xmax = 250.0
    random.seed(0)
    noise = random.normal(scale=2.80, size=n)
    x = linspace(xmin, xmax, n)
    data = residual(p_true, x) + noise

    fit_params = Parameters()
    fit_params.add('amp', value=13.0, max=20, min=0.0)
    fit_params.add('period', value=2, max=10)
    fit_params.add('shift', value=0.0, max=pi/2., min=-pi/2.)
    fit_params.add('decay', value=0.02, max=0.10, min=0.00)

    min = Minimizer(residual, fit_params, (x, data))
    out = min.least_squares()

    assert(out.nfev > 10)
    assert(out.nfree > 50)
    assert(out.chisqr > 1.0)

    assert_paramval(out.params['decay'], 0.01, tol=1.e-2)
    assert_paramval(out.params['shift'], 0.123, tol=1.e-2)


def test_cov_x_no_bounds():
    # load data to be fitted
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                   'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['sigma'].set(min=-np.inf)

    # do fit, here with leastsq model
    result = mod.fit(y, params, x=x, method='least_squares')
    result_lsq = mod.fit(y, params, x=x, method='leastsq')

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_lsq = [result_lsq.params[p].value for p in result_lsq.params.valuesdict()]
    assert_allclose(vals_lsq, vals, rtol=1e-5)
    assert_allclose(result_lsq.chisqr, result.chisqr)

    # assert that parameter uncertaintes obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    stderr = [result.params[p].stderr for p in result.params.valuesdict()]
    stderr_lsq = [result_lsq.params[p].stderr for p in result_lsq.params.valuesdict()]
    assert_almost_equal(stderr_lsq, stderr, decimal=5)

    # assert that parameter correlations obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    for par1 in result.var_names:
        cor = [result.params[par1].correl[par2] for par2 in
               result.params[par1].correl.keys()]
        cor_lsq = [result_lsq.params[par1].correl[par2] for par2 in
                   result_lsq.params[par1].correl.keys()]
        assert_almost_equal(cor_lsq, cor, decimal=5)


def test_cov_x_with_bounds():
    # load data to be fitted
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'examples',
                                   'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    params['amplitude'].set(min=25, max=70)
    params['sigma'].set(min=0, max=1)
    params['center'].set(min=5, max=15)

    # do fit, here with leastsq model
    result = mod.fit(y, params, x=x, method='least_squares')
    result_lsq = mod.fit(y, params, x=x, method='leastsq')

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_lsq = [result_lsq.params[p].value for p in result_lsq.params.valuesdict()]
    assert_allclose(vals_lsq, vals, rtol=1e-5)
    assert_allclose(result_lsq.chisqr, result.chisqr)

    # assert that parameter uncertaintes obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    stderr = [result.params[p].stderr for p in result.params.valuesdict()]
    stderr_lsq = [result_lsq.params[p].stderr for p in result_lsq.params.valuesdict()]
    assert_almost_equal(stderr_lsq, stderr, decimal=6)

    # assert that parameter correlations obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    for par1 in result.var_names:
        cor = [result.params[par1].correl[par2] for par2 in
               result.params[par1].correl.keys()]
        cor_lsq = [result_lsq.params[par1].correl[par2] for par2 in
                   result_lsq.params[par1].correl.keys()]
        assert_almost_equal(cor_lsq, cor, decimal=6)
