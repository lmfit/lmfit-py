import numpy as np
from lmfit import Parameters, minimize, report_fit

def get_data():
    x = np.arange(0, 1, 0.01)
    y1 = 1.5*np.exp(0.9*x) + np.random.normal(scale=0.001, size=len(x))
    y2 = 2.0 + x + 1/2.*x**2 +1/3.*x**3
    y2 = y2 + np.random.normal(scale=0.001, size=len(x))
    return x, y1, y2

def residual(params, x, data):
    a = params['a'].value
    b = params['b'].value

    model = a*np.exp(b*x)
    return (data-model)

def test_copy_params():
    x, y1, y2 = get_data()

    params = Parameters()
    params.add('a', value = 2.0)
    params.add('b', value = 2.0)

    # fit to first data set
    out1 = minimize(residual, params, args=(x, y1))

    # fit to second data set
    out2 = minimize(residual, params, args=(x, y2))

    adiff = out1.params['a'].value - out2.params['a'].value
    bdiff = out1.params['b'].value - out2.params['b'].value

    assert(abs(adiff) > 1.e-2)
    assert(abs(bdiff) > 1.e-2)

