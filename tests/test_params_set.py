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
    sigval = params['gamma'].value
    assert(params['gamma'].expr == 'sigma')
    assert_allclose(params['gamma'].value, sigval, 1e-4, 1e-4, '', True)

    # test #2: explicitly setting a param value should work, even when
    #          it had been an expression.  The value will be left as fixed
    gamval = 0.87543
    params['gamma'].set(value=gamval)
    assert(params['gamma'].expr is None)
    assert(not params['gamma'].vary)
    assert_allclose(params['gamma'].value, gamval, 1e-4, 1e-4, '', True)

    # test #3: explicitly setting an expression should work
    params['gamma'].set(expr='sigma/2.0')
    assert(params['gamma'].expr is not None)
    assert(not params['gamma'].vary)
    assert_allclose(params['gamma'].value, sigval/2.0, 1e-4, 1e-4, '', True)

    # test #4: explicitly setting a param value WITH vary=True
    #          will set it to be variable
    gamval = 0.7777
    params['gamma'].set(value=gamval, vary=True)
    assert(params['gamma'].expr is None)
    assert(params['gamma'].vary)
    assert_allclose(params['gamma'].value, gamval, 1e-4, 1e-4, '', True)
