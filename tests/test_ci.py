from __future__ import print_function
from lmfit import minimize, Parameters, conf_interval, report_ci, report_errors
import numpy as np
pi = np.pi
import nose

def test_ci():
    np.random.seed(1)
    p_true = Parameters()
    p_true.add('amp', value=14.0)
    p_true.add('period', value=5.33)
    p_true.add('shift', value=0.123)
    p_true.add('decay', value=0.010)

    def residual(pars, x, data=None):
        amp = pars['amp'].value
        per = pars['period'].value
        shift = pars['shift'].value
        decay = pars['decay'].value

        if abs(shift) > pi / 2:
            shift = shift - np.sign(shift) * pi
        model = amp * np.sin(shift + x / per) * np.exp(-x * x * decay * decay)
        if data is None:
            return model
        return model - data


    n = 2500
    xmin = 0.
    xmax = 250.0
    noise = np.random.normal(scale=0.7215, size=n)
    x = np.linspace(xmin, xmax, n)
    data = residual(p_true, x) + noise

    fit_params = Parameters()
    fit_params.add('amp', value=13.0)
    fit_params.add('period', value=4)
    fit_params.add('shift', value=0.1)
    fit_params.add('decay', value=0.02)

    out = minimize(residual, fit_params, args=(x,), kws={'data': data})

    fit = residual(fit_params, x)

    print( ' N fev = ', out.nfev)
    print( out.chisqr, out.redchi, out.nfree)

    report_errors(fit_params)
    ci, tr = conf_interval(out, sigmas=[0.674], trace=True)
    report_ci(ci)
    for p in out.params:
        diff1 = ci[p][1][1] - ci[p][0][1]
        diff2 = ci[p][2][1] - ci[p][1][1]
        stderr = out.params[p].stderr
        assert(abs(diff1 - stderr) / stderr < 0.05)
        assert(abs(diff2 - stderr) / stderr < 0.05)

