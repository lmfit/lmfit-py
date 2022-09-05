"""Tests for the calculation of confidence intervals."""
import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import f

import lmfit


@pytest.fixture
def data():
    """Generate synthetic data."""
    x = np.linspace(0.3, 10, 100)
    np.random.seed(0)
    y = 1.0 / (0.1 * x) + 2.0 + 0.1 * np.random.randn(x.size)
    return (x, y)


@pytest.fixture
def pars():
    """Create and initialize parameter set."""
    parameters = lmfit.Parameters()
    parameters.add_many(('a', 0.1), ('b', 1))
    return parameters


def residual(params, x, data):
    """Define objective function for the minimization."""
    model = 1.0 / (params['a'] * x) + params['b']
    return data - model


def test_default_f_compare(data, pars):
    """Test the default f_compare function: F-test."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()

    # "fixing" a parameter, keeping the chisqr the same
    out2 = copy.deepcopy(out)
    out2.nvarys = 1
    prob = lmfit.confidence.f_compare(out, out2)
    assert_allclose(prob, 0.0)

    # "fixing" a parameter, increasing the chisqr
    out2.chisqr = 1.0015*out.chisqr
    prob = lmfit.confidence.f_compare(out, out2)
    assert_allclose(prob, 0.2977506)


def test_copy_and_restore_vals(data, pars):
    """Test functions to save and restore parameter values and stderrs."""
    # test copy_vals without/with stderr present
    copy_pars = lmfit.confidence.copy_vals(pars)

    assert isinstance(copy_pars, dict)
    for _, par in enumerate(pars):
        assert_allclose(pars[par].value, copy_pars[par][0])
        assert copy_pars[par][1] is None  # no stderr present

    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()
    copy_pars_out = lmfit.confidence.copy_vals(out.params)

    assert isinstance(copy_pars_out, dict)
    for _, par in enumerate(out.params):
        assert_allclose(out.params[par].value, copy_pars_out[par][0])
        assert_allclose(out.params[par].stderr, copy_pars_out[par][1])

    # test restore_vals to the original parameter set after changing them first
    pars['a'].set(value=1.0)
    pars['b'].set(value=10)
    lmfit.confidence.restore_vals(copy_pars, pars)

    assert isinstance(pars, lmfit.parameter.Parameters)
    assert_allclose(pars['a'].value, 0.1)
    assert_allclose(pars['b'].value, 1.0)
    assert pars['a'].stderr is None
    assert pars['b'].stderr is None

    lmfit.confidence.restore_vals(copy_pars_out, pars)
    for _, par in enumerate(pars):
        assert_allclose(pars[par].value, out.params[par].value)
        assert_allclose(pars[par].stderr, out.params[par].stderr)


@pytest.mark.parametrize("verbose", [False, True])
def test_confidence_leastsq(data, pars, verbose, capsys):
    """Calculate confidence interval after leastsq minimization."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()

    assert 5 < out.nfev < 500
    assert out.chisqr < 3.0
    assert out.nvarys == 2
    assert_allclose(out.params['a'], 0.1, rtol=0.01)
    assert_allclose(out.params['b'], 2.0, rtol=0.01)

    ci = lmfit.conf_interval(minimizer, out, verbose=verbose)
    assert_allclose(ci['b'][0][0], 0.997, rtol=0.01)
    assert_allclose(ci['b'][0][1], 1.947, rtol=0.01)
    assert_allclose(ci['b'][2][0], 0.683, rtol=0.01)
    assert_allclose(ci['b'][2][1], 1.972, rtol=0.01)
    assert_allclose(ci['b'][5][0], 0.95, rtol=0.01)
    assert_allclose(ci['b'][5][1], 2.01, rtol=0.01)

    if verbose:
        captured = capsys.readouterr()
        assert 'Calculating CI for' in captured.out


def test_confidence_pnames(data, pars):
    """Test if pnames works as expected."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()

    assert_allclose(out.params['a'], 0.1, rtol=0.01)
    assert_allclose(out.params['b'], 2.0, rtol=0.01)

    ci = lmfit.conf_interval(minimizer, out, p_names=['a'])
    assert 'a' in ci
    assert 'b' not in ci


def test_confidence_bounds_reached(data, pars):
    """Check if conf_interval handles bounds correctly"""

    # Should work
    pars['a'].max = 0.2
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()
    out.params['a'].stderr = 1
    lmfit.conf_interval(minimizer, out, verbose=True)

    # Should warn (i.e,. limit < para.min)
    pars['b'].max = 2.03
    pars['b'].min = 1.97
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()
    out.params['b'].stderr = 0.005
    out.params['a'].stderr = 0.01
    with pytest.warns(UserWarning, match="Bound reached"):
        lmfit.conf_interval(minimizer, out, verbose=True)

    # Should warn (i.e,. limit > para.max)
    out.params['b'].stderr = 0.1
    with pytest.warns(UserWarning, match="Bound reached"):
        lmfit.conf_interval(minimizer, out, verbose=True)


def test_confidence_sigma_vs_prob(data, pars):
    """Calculate confidence by specifying sigma or probability."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=(data))
    out = minimizer.leastsq()

    ci_sigma_None = lmfit.conf_interval(minimizer, out, sigmas=None)
    ci_sigmas = lmfit.conf_interval(minimizer, out, sigmas=[1, 2, 3])
    ci_1sigma = lmfit.conf_interval(minimizer, out, sigmas=[1])
    ci_probs = lmfit.conf_interval(minimizer, out,
                                   sigmas=[0.68269, 0.9545, 0.9973])

    assert ci_sigma_None == ci_sigmas
    assert_allclose(ci_sigmas['a'][0][1], ci_probs['a'][0][1], rtol=0.01)
    assert_allclose(ci_sigmas['b'][2][1], ci_probs['b'][2][1], rtol=0.01)
    assert len(ci_1sigma['a']) == 3
    assert len(ci_probs['a']) == 7


def test_confidence_exceptions(data, pars):
    """Make sure the proper exceptions are raised when needed."""
    minimizer = lmfit.Minimizer(residual, pars, calc_covar=False,
                                fcn_args=data)
    out = minimizer.minimize(method='nelder')
    out_lsq = minimizer.minimize(params=out.params, method='leastsq')

    # no uncertainty estimated
    msg = 'Cannot determine Confidence Intervals without sensible uncertainty'
    with pytest.raises(lmfit.MinimizerException, match=msg):
        lmfit.conf_interval(minimizer, out)

    # uncertainty is NaN
    out_lsq.params['a'].stderr = np.nan
    with pytest.raises(lmfit.MinimizerException, match=msg):
        lmfit.conf_interval(minimizer, out_lsq)

    # only one varying parameter
    out_lsq.params['a'].vary = False
    msg = r'Cannot determine Confidence Intervals with < 2 variables'
    with pytest.raises(lmfit.MinimizerException, match=msg):
        lmfit.conf_interval(minimizer, out_lsq)


def test_confidence_warnings(data, pars):
    """Make sure the proper warnings are emitted when needed."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=data)
    out = minimizer.minimize(method='leastsq')

    with pytest.warns(UserWarning) as record:
        lmfit.conf_interval(minimizer, out, maxiter=1)
        assert 'maxiter=1 reached and prob' in str(record[0].message)


def test_confidence_with_trace(data, pars):
    """Test calculation of confidence intervals with trace."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=data)
    out = minimizer.leastsq()

    ci, tr = lmfit.conf_interval(minimizer, out, sigmas=[0.6827], trace=True)
    for p in out.params:
        diff1 = ci[p][1][1] - ci[p][0][1]
        diff2 = ci[p][2][1] - ci[p][1][1]
        stderr = out.params[p].stderr
        assert abs(diff1 - stderr) / stderr < 0.05
        assert abs(diff2 - stderr) / stderr < 0.05

        assert p in tr.keys()
        assert 'prob' in tr[p].keys()


def test_confidence_2d(data, pars):
    """Test the 2D confidence interval calculation."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=data)
    out = minimizer.minimize(method='leastsq')

    cx, cy, grid = lmfit.conf_interval2d(minimizer, out, 'a', 'b', 30, 20)
    assert len(cx.ravel()) == 30
    assert len(cy.ravel()) == 20
    assert grid.shape == (20, 30)


def test_confidence_2d_limits(data, pars):
    """Test the 2D confidence interval calculation using limits."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=data)
    out = minimizer.minimize(method='leastsq')

    lim = ((1.0e-6, 0.02), (1.0e-6, -4.0))
    cx, cy, grid = lmfit.conf_interval2d(minimizer, out, 'a', 'b', limits=lim)
    assert grid.shape == (10, 10)
    assert_allclose(min(cx.ravel()), 1.0e-6)
    assert_allclose(max(cx.ravel()), 0.02)
    assert_allclose(min(cy.ravel()), -4.0)
    assert_allclose(max(cy.ravel()), 1.0e-6)


def test_confidence_prob_func(data, pars):
    """Test conf_interval with alternate prob_func."""
    minimizer = lmfit.Minimizer(residual, pars, fcn_args=data)
    out = minimizer.minimize(method='leastsq')
    called = 0

    def my_f_compare(best_fit, new_fit):
        nonlocal called
        called += 1
        nfree = best_fit.nfree
        nfix = best_fit.nvarys - new_fit.nvarys
        dchi = new_fit.chisqr / best_fit.chisqr - 1.0
        return f.cdf(dchi * nfree / nfix, nfix, nfree)

    lmfit.conf_interval(minimizer, out, sigmas=[1], prob_func=my_f_compare)
    assert called > 10
