from lmfit import Parameter, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

from testutils import report_errors

import pylab

def residual(pars, x, data=None):
    # print 'RESID ', pars['amp_g'].value, pars['amp_g'].init_value
    yg = gauss(x, pars['amp_g'].value,
               pars['cen_g'].value, pars['wid_g'].value)
    yl = loren(x, pars['amp_l'].value,
               pars['cen_l'].value, pars['wid_l'].value)

    frac = pars['frac'].value
    slope = pars['line_slope'].value
    offset = pars['line_off'].value
    model = (1-frac) * yg + frac * yl + offset + x * slope
    if data is None:
        return model
    return (model - data)

n = 601
xmin = 0.
xmax = 20.0
x = linspace(xmin, xmax, n)

noise = random.normal(scale=2.5, size=n) + x*0.62 + -1.023

true_params = {'amp': Parameter(value=21.0),
            'cen': Parameter(value=8.3),
            'wid': Parameter(value=1.6),
            'frac': Parameter(value=0.37),
            }

data = pvoigt(x, true_params['amp'].value,
              true_params['cen'].value,
              true_params['wid'].value,
              true_params['frac'].value) + noise

pylab.plot(x, data, 'r+')

fit_params = {'amp_g': Parameter(value=10.0),
              'cen_g': Parameter(value=8.5),
              'wid_g': Parameter(value=1.6),
              'frac': Parameter(value=0.50, max=1.3),
              'amp_l': Parameter(expr='amp_g'),
              'cen_l': Parameter(expr='cen_g'),
              'wid_l': Parameter(expr='wid_g'),
              'line_slope': Parameter(value=0.0),
              'line_off': Parameter(value=0.0),
              }

myfit = Minimizer(residual, fit_params,
                  fcn_args=(x,), fcn_kws={'data':data})

myfit.prepare_fit()
init = residual(fit_params, x)

pylab.plot(x, init, 'b--')
myfit.fit()


print ' Nfev = ', myfit.nfev
print myfit.chisqr, myfit.redchi, myfit.nfree

report_errors(fit_params)

fit = residual(fit_params, x)


pylab.plot(x, fit, 'k-')
pylab.show()





