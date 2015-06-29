from lmfit import Parameters, minimize, fit_report
from lmfit_testutils import assert_paramval, assert_paramattr

import numpy as np


def test_bounded_jacobian():
    pars = Parameters()
    pars.add('x0', value=2.0)
    pars.add('x1', value=2.0, min=1.5)

    global jac_count

    jac_count = 0

    def resid(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.array([10 * (x1 - x0*x0), 1-x0])

    def jac(params):
        global jac_count
        jac_count += 1
        x0 = params['x0'].value
        return np.array([[-20*x0, 10], [-1, 0]])
    
    out0 = minimize(resid, pars, Dfun=None)

    assert_paramval(out0.params['x0'], 1.2243, tol=0.02)
    assert_paramval(out0.params['x1'], 1.5000, tol=0.02)
    assert(jac_count == 0)

    out1 = minimize(resid, pars, Dfun=jac)

    assert_paramval(out1.params['x0'], 1.2243, tol=0.02)
    assert_paramval(out1.params['x1'], 1.5000, tol=0.02)
    assert(jac_count > 5)

    print(fit_report(out1, show_correl=True))


if __name__ == '__main__':
    test_bounded_jacobian()
