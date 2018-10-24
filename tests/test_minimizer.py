from lmfit import Minimizer, Parameters


def test_scalar_minimize_neg_value():
    x0 = 3.14
    fmin = -1.1
    xtol = 0.001
    ftol = 2.0 * xtol

    def objective(pars):
        return (pars['x'] - x0) ** 2.0 + fmin

    params = Parameters()
    params.add('x', value=2*x0)

    minr = Minimizer(objective, params)
    result = minr.scalar_minimize(method='Nelder-Mead',
                                  options={'xatol': xtol, 'fatol': ftol})
    assert abs(result.params['x'].value - x0) < xtol
    assert abs(result.fun - fmin) < ftol
