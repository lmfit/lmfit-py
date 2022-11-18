# test of fitting one variable
# From Nick Schurch

import numpy
from numpy.testing import assert_allclose

import lmfit


def linear_chisq(params, x, data, errs=None):
    """Calculates chi-squared residuals for linear model."""
    if not isinstance(params, lmfit.parameter.Parameters):
        msg = "Params argument is not a lmfit parameter set"
        raise TypeError(msg)

    if "m" not in params.keys():
        msg = "No slope parameter (m) defined in the model"
        raise KeyError(msg)

    if "c" not in params.keys():
        msg = "No intercept parameter (c) defined in the model"
        raise KeyError(msg)

    model = params["m"]*x + params["c"]
    residuals = (data-model)
    if errs is not None:
        residuals = residuals/errs

    return residuals


def test_1var():
    rands = [-0.21698284, 0.41900591, 0.02349374, -0.218552, -0.3513699,
             0.33418304, 0.04226855, 0.213303, 0.45948731, 0.33587736]

    x = numpy.arange(10)+1
    y = numpy.arange(10)+1+rands

    params = lmfit.Parameters()
    params.add(name="m", value=1.0, vary=True)
    params.add(name="c", value=0.0, vary=False)

    out = lmfit.minimize(linear_chisq, params, args=(x, y))

    assert_allclose(params['m'].value, 1.025, rtol=0.02, atol=0.02)
    assert len(params) == 2
    assert out.nvarys == 1
    assert out.chisqr > 0.01
    assert out.chisqr < 5.00
