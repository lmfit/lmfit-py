"""Tests for the least_squares minimization algorithm."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import aslinearoperator

import lmfit
from lmfit.models import VoigtModel


def test_least_squares_with_bounds():
    """Test least_squares algorihhm with bounds."""
    # define "true" parameters
    p_true = lmfit.Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.4321)
    p_true.add('shift', value=0.12345)
    p_true.add('decay', value=0.01000)

    def residual(pars, x, data=None):
        """Objective function of decaying sine wave."""
        amp = pars['amp']
        per = pars['period']
        shift = pars['shift']
        decay = pars['decay']

        if abs(shift) > np.pi/2:
            shift = shift - np.sign(shift)*np.pi

        model = amp*np.sin(shift + x/per) * np.exp(-x*x*decay*decay)
        if data is None:
            return model
        return model - data

    # generate synthetic data
    np.random.seed(0)
    x = np.linspace(0.0, 250.0, 1500)
    noise = np.random.normal(scale=2.80, size=x.size)
    data = residual(p_true, x) + noise

    # create Parameters and set initial values and bounds
    fit_params = lmfit.Parameters()
    fit_params.add('amp', value=13.0, min=0.0, max=20)
    fit_params.add('period', value=2, max=10)
    fit_params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2.)
    fit_params.add('decay', value=0.02, min=0.0, max=0.10)

    mini = lmfit.Minimizer(residual, fit_params, fcn_args=(x, data))
    out = mini.minimize(method='least_squares')

    assert out.method == 'least_squares'
    assert out.nfev > 10
    assert out.nfree > 50
    assert out.chisqr > 1.0
    assert out.errorbars
    assert out.success
    assert_allclose(out.params['decay'], p_true['decay'], rtol=1e-2)
    assert_allclose(out.params['shift'], p_true['shift'], rtol=1e-2)


@pytest.mark.parametrize("bounds", [False, True])
def test_least_squares_cov_x(peakdata, bounds):
    """Test calculation of cov. matrix from Jacobian, with/without bounds."""
    x = peakdata[0]
    y = peakdata[1]

    # define the model and initialize parameters
    mod = VoigtModel()
    params = mod.guess(y, x=x)

    if bounds:
        params['amplitude'].set(min=25, max=70)
        params['sigma'].set(min=0, max=1)
        params['center'].set(min=5, max=15)
    else:
        params['sigma'].set(min=-np.inf)

    # do fit with least_squares and leastsq algorithm
    result = mod.fit(y, params, x=x, method='least_squares')
    result_lsq = mod.fit(y, params, x=x, method='leastsq',
                         fit_kws={'epsfcn': 1.e-14})

    # assert that fit converged to the same result
    vals = [result.params[p].value for p in result.params.valuesdict()]
    vals_lsq = [result_lsq.params[p].value for p in
                result_lsq.params.valuesdict()]
    assert_allclose(vals, vals_lsq, rtol=1e-5)
    assert_allclose(result.chisqr, result_lsq.chisqr)

    # assert that parameter uncertainties obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    stderr = [result.params[p].stderr for p in result.params.valuesdict()]
    stderr_lsq = [result_lsq.params[p].stderr for p in
                  result_lsq.params.valuesdict()]
    assert_allclose(stderr, stderr_lsq, rtol=1e-4)

    # assert that parameter correlations obtained from the leastsq method and
    # those from the covariance matrix estimated from the Jacbian matrix in
    # least_squares are similar
    for par1 in result.var_names:
        cor = [result.params[par1].correl[par2] for par2 in
               result.params[par1].correl.keys()]
        cor_lsq = [result_lsq.params[par1].correl[par2] for par2 in
                   result_lsq.params[par1].correl.keys()]

        assert_allclose(cor, cor_lsq, rtol=0.01, atol=1.e-6)


def test_least_squares_solver_options(peakdata, capsys):
    """Test least_squares algorithm, pass options to solver."""
    x = peakdata[0]
    y = peakdata[1]
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    solver_kws = {'verbose': 2}
    mod.fit(y, params, x=x, method='least_squares', fit_kws=solver_kws)
    captured = capsys.readouterr()

    assert 'Iteration' in captured.out
    assert 'final cost' in captured.out


def test_least_squares_jacobian_types():
    """Test support for Jacobian of all types supported by least_squares."""
    # Build function
    # f(x, y) = (x - a)^2 + (y - b)^2
    np.random.seed(42)
    a = np.random.normal(0, 1, 50)
    np.random.seed(43)
    b = np.random.normal(0, 1, 50)

    def f(params):
        return (params['x'] - a)**2 + (params['y'] - b)**2

    # Build analytic Jacobian functions with the different possible return types
    # numpy.ndarray, scipy.sparse.spmatrix, scipy.sparse.linalg.LinearOperator
    # J = [ 2x - 2a , 2y - 2b ]
    def jac_array(params, *args, **kwargs):
        return np.column_stack((2 * params[0] - 2 * a, 2 * params[1] - 2 * b))

    def jac_sparse(params, *args, **kwargs):
        return bsr_matrix(jac_array(params, *args, **kwargs))

    def jac_operator(params, *args, **kwargs):
        return aslinearoperator(jac_array(params, *args, **kwargs))
    # Build parameters
    params = lmfit.Parameters()
    params.add('x', value=0)
    params.add('y', value=0)
    # Solve model for numerical Jacobian and each analytic Jacobian function
    result = lmfit.minimize(f, params, method='least_squares')
    result_array = lmfit.minimize(
        f, params, method='least_squares',
        jac=jac_array)
    result_sparse = lmfit.minimize(
        f, params, method='least_squares',
        jac=jac_sparse)
    result_operator = lmfit.minimize(
        f, params, method='least_squares',
        jac=jac_operator)
    # Check that all have uncertainties
    assert result.errorbars
    assert result_array.errorbars
    assert result_sparse.errorbars
    assert result_operator.errorbars
    # Check that all have ~equal covariance matrix
    assert_allclose(result.covar, result_array.covar)
    assert_allclose(result.covar, result_sparse.covar)
    assert_allclose(result.covar, result_operator.covar)
