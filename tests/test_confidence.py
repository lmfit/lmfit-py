import numpy as np
from numpy.testing import assert_allclose

import lmfit
from lmfit_testutils import assert_paramval


def residual(params, x, data):
    return data - 1.0/(params['a']*x) + params['b']


def residual2(params, x, data):
    return data - params['c']/(params['a']*x) + params['b']


def test_confidence1():
    x = np.linspace(0.3, 10, 100)
    np.random.seed(0)

    y = 1/(0.1*x) + 2 + 0.1*np.random.randn(x.size)

    pars = lmfit.Parameters()
    pars.add_many(('a', 0.1), ('b', 1))

    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(x, y))
    out = minimizer.leastsq()

    assert(out.nfev > 5)
    assert(out.nfev < 500)
    assert(out.chisqr < 3.0)
    assert(out.nvarys == 2)

    assert_paramval(out.params['a'], 0.1, tol=0.1)
    assert_paramval(out.params['b'], -2.0, tol=0.1)

    ci = lmfit.conf_interval(minimizer, out)
    assert_allclose(ci['b'][0][0], 0.997, rtol=0.01)
    assert_allclose(ci['b'][0][1], -2.022, rtol=0.01)
    assert_allclose(ci['b'][2][0], 0.683, rtol=0.01)
    assert_allclose(ci['b'][2][1], -1.997, rtol=0.01)
    assert_allclose(ci['b'][5][0], 0.95, rtol=0.01)
    assert_allclose(ci['b'][5][1], -1.96, rtol=0.01)


def test_confidence2():
    x = np.linspace(0.3, 10, 100)
    np.random.seed(0)

    y = 1/(0.1*x) + 2 + 0.1*np.random.randn(x.size)

    pars = lmfit.Parameters()
    pars.add_many(('a', 0.1), ('b', 1), ('c', 1.0))
    pars['a'].max = 0.25
    pars['a'].min = 0.00
    pars['a'].value = 0.2
    pars['c'].vary = False

    minimizer = lmfit.Minimizer(residual2, pars, fcn_args=(x, y))
    out = minimizer.minimize(method='nelder')
    out = minimizer.minimize(method='leastsq', params=out.params)

    assert(out.nfev > 3)
    assert(out.nfev < 500)
    assert(out.chisqr < 3.0)
    assert(out.nvarys == 2)

    assert_paramval(out.params['a'], 0.1, tol=0.1)
    assert_paramval(out.params['b'], -2.0, tol=0.1)

    ci = lmfit.conf_interval(minimizer, out)
    assert_allclose(ci['b'][0][0], 0.997, rtol=0.01)
    assert_allclose(ci['b'][0][1], -2.022, rtol=0.01)
    assert_allclose(ci['b'][2][0], 0.683, rtol=0.01)
    assert_allclose(ci['b'][2][1], -1.997, rtol=0.01)
    assert_allclose(ci['b'][5][0], 0.95, rtol=0.01)
    assert_allclose(ci['b'][5][1], -1.96, rtol=0.01)


def test_ci_with_trace():
    np.random.seed(1)
    p_true = lmfit.Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.33)
    p_true.add('shift', value=0.123)
    p_true.add('decay', value=0.010)

    def residual(pars, x, data=None):
        amp = pars['amp']
        per = pars['period']
        shift = pars['shift']
        decay = pars['decay']

        if abs(shift) > np.pi / 2:
            shift = shift - np.sign(shift) * np.pi
        model = amp * np.sin(shift + x / per) * np.exp(-x * x * decay * decay)
        if data is None:
            return model
        return model - data

    n = 2500
    xmin = 0.
    xmax = 250.0
    noise = np.random.normal(scale=0.7215, size=n)
    x = np.linspace(xmin, xmax, n)
    data = residual(p_true, x) + noise

    fit_params = lmfit.Parameters()
    fit_params.add('amp', value=13.0)
    fit_params.add('period', value=4)
    fit_params.add('shift', value=0.1)
    fit_params.add('decay', value=0.02)

    mini = lmfit.Minimizer(residual, fit_params, fcn_args=(x, data))
    out = mini.minimize()

    ci, tr = lmfit.conf_interval(mini, out, sigmas=[0.674], trace=True)
    for p in out.params:
        diff1 = ci[p][1][1] - ci[p][0][1]
        diff2 = ci[p][2][1] - ci[p][1][1]
        stderr = out.params[p].stderr
        assert(abs(diff1 - stderr) / stderr < 0.05)
        assert(abs(diff2 - stderr) / stderr < 0.05)
