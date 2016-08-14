"""
tests of ModelResult.eval_uncertainty()

"""
import numpy as np
from numpy.testing import assert_allclose

from lmfit.models import LinearModel

def setup(slope=0.8, intercept=0.5, noise=1.5):
    np.random.seed(1)

    # create data to be fitted
    x = np.linspace(0, 10, 101)
    y = intercept + x*slope
    y = y + np.random.normal(size=len(x), scale=noise)

    model = LinearModel()
    params = model.make_params(intercept=intercept, slope=slope)

    return x, y, model, params

def test_linear_constant_intercept():
    x, y, model, params = setup(slope=4, intercept=-10)

    params['intercept'].vary = False

    ret = model.fit(y, params, x=x)

    dely = ret.eval_uncertainty(sigma=1)
    slope_stderr = ret.params['slope'].stderr

    assert_allclose(dely.min(), 0, rtol=1.e-2)
    assert_allclose(dely.max(), slope_stderr*x.max(), rtol=1.e-2)
    assert_allclose(dely.mean(),slope_stderr*x.mean(), rtol=1.e-2)

def test_linear_constant_slope():
    x, y, model, params = setup(slope=-4, intercept=12.5)

    params['slope'].vary = False

    ret = model.fit(y, params, x=x)

    dely = ret.eval_uncertainty(sigma=1)

    intercept_stderr = ret.params['intercept'].stderr

    assert_allclose(dely.min(), intercept_stderr, rtol=1.e-2)
    assert_allclose(dely.max(), intercept_stderr, rtol=1.e-2)




if __name__ == '__main__':
    test_linear_constant_intercept()
    test_linear_constant_slope()
