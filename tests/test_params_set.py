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
    sigval = params['sigma'].value
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

    # test 5: make sure issue #389 is fixed: set boundaries and make sure
    #         they are kept when changing the value
    amplitude_vary = params['amplitude'].vary
    amplitude_expr = params['amplitude'].expr
    params['amplitude'].set(min=0.0, max=100.0)
    params.update_constraints()
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    params['amplitude'].set(value=40.0)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 40.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(not params['amplitude'].brute_step)

    # test for possible regressions of this fix (without 'expr'):
    # the set function should only change the requested attribute(s)
    params['amplitude'].set(value=35.0)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(not params['amplitude'].brute_step)

    # set minimum
    params['amplitude'].set(min=10.0)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 100.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(not params['amplitude'].brute_step)

    # set maximum
    params['amplitude'].set(max=110.0)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 110.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == amplitude_vary)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(not params['amplitude'].brute_step)

    # set vary
    params['amplitude'].set(vary=False)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 110.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == False)
    assert(params['amplitude'].expr == amplitude_expr)
    assert(not params['amplitude'].brute_step)

    # set brute_step
    params['amplitude'].set(brute_step=0.1)
    params.update_constraints()
    assert_allclose(params['amplitude'].value, 35.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].min, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['amplitude'].max, 110.0, 1e-4, 1e-4, '', True)
    assert(params['amplitude'].vary == False)
    assert(params['amplitude'].expr == amplitude_expr)
    assert_allclose(params['amplitude'].brute_step, 0.1, 1e-4, 1e-4, '', True)

    # test for possible regressions of this fix for variables WITH 'expr':
    height_value = params['height'].value
    height_min = params['height'].min
    height_max = params['height'].max
    height_vary = params['height'].vary
    height_expr = params['height'].expr
    height_brute_step = params['height'].brute_step

    # set vary=True should remove expression
    params['height'].set(vary=True)
    params.update_constraints()
    assert_allclose(params['height'].value, height_value, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, height_min, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == True)
    assert(params['height'].expr == None)
    assert(params['height'].brute_step == height_brute_step)

    # setting an expression should set vary=False
    params['height'].set(expr=height_expr)
    params.update_constraints()
    assert_allclose(params['height'].value, height_value, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, height_min, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == False)
    assert(params['height'].expr == height_expr)
    assert(params['height'].brute_step == height_brute_step)

    # changing min/max should not remove expression
    params['height'].set(min=0)
    params.update_constraints()
    assert_allclose(params['height'].value, height_value, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == height_vary)
    assert(params['height'].expr == height_expr)
    assert(params['height'].brute_step == height_brute_step)

    # changing brute_step should not remove expression
    params['height'].set(brute_step=0.1)
    params.update_constraints()
    assert_allclose(params['height'].value, height_value, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == height_vary)
    assert(params['height'].expr == height_expr)
    assert_allclose(params['amplitude'].brute_step, 0.1, 1e-4, 1e-4, '', True)

    # changing the value should remove expression and keep vary=False
    params['height'].set(brute_step=0)
    params['height'].set(value=10.0)
    params.update_constraints()
    assert_allclose(params['height'].value, 10.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == False)
    assert(params['height'].expr == None)
    assert(params['height'].brute_step == height_brute_step)

    # passing expr='' should only remove the expression
    params['height'].set(expr=height_expr) # first restore the original expr
    params.update_constraints()
    params['height'].set(expr='')
    params.update_constraints()
    assert_allclose(params['height'].value, height_value, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].min, 0.0, 1e-4, 1e-4, '', True)
    assert_allclose(params['height'].max, height_max, 1e-4, 1e-4, '', True)
    assert(params['height'].vary == False)
    assert(params['height'].expr == None)
    assert(params['height'].brute_step == height_brute_step)

test_param_set()
