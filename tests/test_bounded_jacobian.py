import numpy as np
from numpy.testing import assert_allclose

from lmfit import Parameters, minimize


def test_bounded_jacobian():
    pars = Parameters()
    pars.add('x0', value=2.0)
    pars.add('x1', value=2.0, min=1.5)

    global jac_count

    jac_count = 0

    def resid(params):
        x0 = params['x0']
        x1 = params['x1']
        return np.array([10 * (x1 - x0*x0), 1-x0])

    def jac(params):
        global jac_count
        jac_count += 1
        x0 = params['x0']
        return np.array([[-20*x0, 10], [-1, 0]])

    out0 = minimize(resid, pars, Dfun=None)

    assert_allclose(out0.params['x0'], 1.2243, rtol=1.0e-4)
    assert_allclose(out0.params['x1'], 1.5000, rtol=1.0e-4)
    assert jac_count == 0

    out1 = minimize(resid, pars, Dfun=jac)

    assert_allclose(out1.params['x0'], 1.2243, rtol=1.0e-4)
    assert_allclose(out1.params['x1'], 1.5000, rtol=1.0e-4)
    assert jac_count > 5


def test_bounded_jacobian_CG():
    pars = Parameters()
    pars.add('x0', value=2.0)
    pars.add('x1', value=2.0, min=1.5)

    global jac_count

    jac_count = 0

    def resid(params):
        x0 = params['x0']
        x1 = params['x1']
        return np.array([10 * (x1 - x0*x0), 1-x0])

    def jac(params):
        global jac_count
        jac_count += 1
        x0 = params['x0']
        # Jacobian of the *error*, i.e. the summed squared residuals
        return 2 * np.sum(np.array([[-20 * x0, 10],
                                    [-1, 0]]).T * resid(params), axis=1)

    out0 = minimize(resid, pars, method='CG')
    assert_allclose(out0.params['x0'], 1.2243, rtol=1.0e-4)
    assert_allclose(out0.params['x1'], 1.5000, rtol=1.0e-4)
    assert jac_count == 0

    out1 = minimize(resid, pars, method='CG', Dfun=jac)
    assert_allclose(out1.params['x0'], 1.2243, rtol=1.0e-4)
    assert_allclose(out1.params['x1'], 1.5000, rtol=1.0e-4)
    assert jac_count > 5
