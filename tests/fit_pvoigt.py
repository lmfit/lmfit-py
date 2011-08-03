from lmfit import Parameter, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

from testutils import report_errors

import pylab

def residual(pars, x, data=None):
    # print 'RESID ', pars['w1'].value, pars['w2'].value
    yg = gauss(x, pars['ag'].value, pars['cg'].value, pars['wg'].value)
    yl = loren(x, pars['al'].value, pars['cl'].value, pars['wl'].value)
    frac = pars['frac'].value
    slope = pars['slope'].value
    offset = pars['offset'].value
    
    model = (1-frac) * yg + frac * yl + offset + x * slope
    if data is None:
        return model
    return (model - data)

n = 601
xmin = 0.
xmax = 20.0
x     = linspace(xmin, xmax, n)

noise = random.normal(scale=.715, size=n) + x*0.41 + 0.023


t_params = {'amp': Parameter(value=12.0),
            'cen': Parameter(value=8.3),
            'wid': Parameter(value=1.6),
            'frac': Parameter(value=0.357),
            }

data = pvoigt(x,
              t_params['amp'].value,
              t_params['cen'].value,
              t_params['wid'].value,
              t_params['frac'].value) + noise

pylab.plot(x, data, 'r+')

fit_params = {'ag': Parameter(value=16.0),
              'cg': Parameter(value=8.3),
              'wg': Parameter(value=1),
              'frac': Parameter(value=0.50, max=1), 
              'al': Parameter(expr='ag'),
              'cl': Parameter(expr='cg'),
              'wl': Parameter(expr='wg'), 
              'slope': Parameter(value=0.1),
              'offset': Parameter(value=0.0),
              
              }

myfit = Minimizer(residual, fit_params,
                  fcn_args=(x,), fcn_kws={'data':data})

myfit.prepare_fit()

init = residual(fit_params, x)

pylab.plot(x, init, 'b--')

myfit.fit()

myfit.prepare_fit()
print ' Nfev = ', myfit.nfev
print myfit.chisqr, myfit.redchi, myfit.nfree

report_errors(fit_params)

fit = residual(fit_params, x)

    
pylab.plot(x, fit, 'k-')
pylab.show()





