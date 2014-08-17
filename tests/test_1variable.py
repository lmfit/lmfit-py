# test of fitting one variable
# From Nick Schurch

import lmfit, numpy
from numpy.testing import assert_allclose

def linear_chisq(params, x, data, errs=None):

    ''' Calcs chi-squared residuals linear model (weighted by errors if given)
    '''

    if type(params) is not lmfit.parameter.Parameters:
        msg = "Params argument is not a lmfit parameter set"
        raise TypeError(msg)

    if "m" not in params.keys():
        msg = "No slope parameter (m) defined in the model"
        raise KeyError(msg)

    if "c" not in params.keys():
        msg = "No intercept parameter (c) defined in the model"
        raise KeyError(msg)

    m = params["m"].value
    c = params["c"].value

    model = m*x+c

    residuals = (data-model)
    if errs is not None:
        residuals = residuals/errs

    return(residuals)

def test_1var():
    rands = [-0.21698284, 0.41900591, 0.02349374, -0.218552, -0.3513699,
             0.33418304, 0.04226855, 0.213303, 0.45948731, 0.33587736]

    x = numpy.arange(10)+1
    y = numpy.arange(10)+1+rands
    y_errs = numpy.sqrt(y)/2

    params = lmfit.Parameters()
    params.add(name="m", value=1.0, vary=True)
    params.add(name="c", value=0.0, vary=False)

    out = lmfit.minimize(linear_chisq, params, args=(x, y))

    lmfit.report_fit(out)
    assert_allclose(params['m'].value, 1.025, rtol=0.02, atol=0.02)
    assert(len(params)==2)
    assert(out.nvarys == 1)
    assert(out.chisqr > 0.01)
    assert(out.chisqr < 5.00)

if __name__ == '__main__':
    test_1var()
