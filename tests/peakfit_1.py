from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq
import pylab

from lmfit import Parameter, Minimizer
from lmfit.utilfuncs import gauss, loren

from testutils import report_errors

def residual(pars, x, data=None):
    g1 = gauss(x, pars['a1'].value, pars['c1'].value, pars['w1'].value)
    g2 = gauss(x, pars['a2'].value, pars['c2'].value, pars['w2'].value)
    model = g1 + g2
    if data is None:
        return model
    return (model - data)

n    = 601
xmin = 0.
xmax = 15.0
noise = random.normal(scale=.65, size=n)
x = linspace(xmin, xmax, n)

fit_params = {'a1': Parameter(value=12.0),
              'c1': Parameter(value=5.3),
              'w1': Parameter(value=0.6),
              'a2': Parameter(value=13.0),
              'c2': Parameter(value=8.25),
              'w2': Parameter(value=2.5)}

data  = residual(fit_params, x) + noise

pylab.plot(x, data, 'r+')

fit_params = {'a1': Parameter(value=3.0, max=13.0),
              'c1': Parameter(value=5.0),
              'w1': Parameter(value=0.1),
              'a2': Parameter(value=4.0),
              'c2': Parameter(value=8.8),
              'w2': Parameter(expr='4*w1')}

myfit = Minimizer(residual, fit_params,
                  fcn_args=(x,), fcn_kws={'data':data})

myfit.prepare_fit()

init = residual(fit_params, x)

pylab.plot(x, init, 'b--')

myfit.fit()

myfit.prepare_fit()
print ' N fev = ', myfit.nfev
print myfit.chisqr, myfit.redchi, myfit.nfree

report_errors(fit_params)

fit = residual(fit_params, x)

pylab.plot(x, fit, 'k-')
pylab.show()





