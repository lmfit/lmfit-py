from lmfit import Parameters, minimize, fit_report
from lmfit_testutils import assert_paramval, assert_paramattr

from numpy import linspace, zeros, sin, exp, random, pi, sign

def test_bounds():
    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.4321)
    p_true.add('shift', value=0.12345)
    p_true.add('decay', value=0.01000)

    def residual(pars, x, data=None):
        amp = pars['amp'].value
        per = pars['period'].value
        shift = pars['shift'].value
        decay = pars['decay'].value

        if abs(shift) > pi/2:
            shift = shift - sign(shift)*pi

        model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
        if data is None:
            return model
        return (model - data)

    n = 1500
    xmin = 0.
    xmax = 250.0
    random.seed(0)
    noise = random.normal(scale=2.80, size=n)
    x     = linspace(xmin, xmax, n)
    data  = residual(p_true, x) + noise

    fit_params = Parameters()
    fit_params.add('amp', value=13.0, max=20, min=0.0)
    fit_params.add('period', value=2, max=10)
    fit_params.add('shift', value=0.0, max=pi/2., min=-pi/2.)
    fit_params.add('decay', value=0.02, max=0.10, min=0.00)

    out = minimize(residual, fit_params, args=(x,), kws={'data':data})

    fit = residual(out.params, x)

    assert(out.nfev  > 10)
    assert(out.nfree > 50)
    assert(out.chisqr > 1.0)

    print(fit_report(out, show_correl=True, modelpars=p_true))
    assert_paramval(out.params['decay'], 0.01, tol=1.e-2)
    assert_paramval(out.params['shift'], 0.123, tol=1.e-2)

if __name__ == '__main__':
    test_bounds()
