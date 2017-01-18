import numpy as np
from numpy.testing import assert_allclose
from lmfit import Parameters, minimize, report_fit
from lmfit.lineshapes import gaussian
from lmfit.models import VoigtModel

def test_param_set():
    np.random.seed(2015)
    x = np.arange(0, 20, 0.05)
    y = gaussian(x, amplitude=15.43, center=4.5, sigma=2.13)
    y = y + 0.05 - 0.01*x + np.random.normal(scale=0.03, size=len(x))

    model  = VoigtModel()
    params = model.guess(y, x=x)

    # test #1:  gamma is constrained to equal sigma
    assert(params['gamma'].expr == 'sigma')
    params.update_constraints()
    sigval = params['gamma'].value
    assert_allclose(params['gamma'].value, sigval, 1e-4, 1e-4, '', True)

    # test #2: explicitly setting a param value should work, even when
    #          it had been an expression.  The value will be left as fixed
    gamval = 0.87543
    params['gamma'].set(value=gamval)
    assert(params['gamma'].expr is None)
    assert(not params['gamma'].vary)
    assert_allclose(params['gamma'].value, gamval, 1e-4, 1e-4, '', True)

    # test #3: explicitly setting an expression should work
    # Note, the only way to ensure that **ALL** constraints are up to date
    # is to call params.update_constraints(). This is because the constraint
    # may have multiple dependencies.
    params['gamma'].set(expr='sigma/2.0')
    assert(params['gamma'].expr is not None)
    assert(not params['gamma'].vary)
    params.update_constraints()
    assert_allclose(params['gamma'].value, sigval/2.0, 1e-4, 1e-4, '', True)

    # test #4: explicitly setting a param value WITH vary=True
    #          will set it to be variable
    gamval = 0.7777
    params['gamma'].set(value=gamval, vary=True)
    assert(params['gamma'].expr is None)
    assert(params['gamma'].vary)
    assert_allclose(params['gamma'].value, gamval, 1e-4, 1e-4, '', True)

    # tests to make sure issue #389 is fixed: set boundaries and make sure
    # they are kept when changing the value
    amplitude_vary = params['amplitude'].vary
    amplitude_expr = params['amplitude'].expr
    params['amplitude'].set(min=0.0, max=100.0)
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    params['amplitude'].set(value=40.0)
    assert_allclose(params['amplitude'].value, 40.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(params['amplitude'].vary == amplitude_vary)

    # test for possible regressions of this fix:
    # using the set function should only change the requested attribute and
    # not any others (in case no expression is set)
    params['amplitude'].set(value=35.0)
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)

    params['amplitude'].set(min=10.0)
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)

    params['amplitude'].set(max=110.0)
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 110.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)

    params['amplitude'].set(vary=False)
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 110.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == False)

test_param_set()
