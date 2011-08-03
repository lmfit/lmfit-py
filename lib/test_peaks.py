
from parameter import Parameter
from minimizer import Minimizer

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

from utilfuncs import gauss, loren

import pylab



def residual(pars, x, data=None):
    # print 'RESID ', pars['w1'].value, pars['w2'].value
    g1 = gauss(x, pars['a1'].value, pars['c1'].value, pars['w1'].value)
    g2 = gauss(x, pars['a2'].value, pars['c2'].value, pars['w2'].value)

    model = g1 + g2
    if data is None:
        return model
    return (model - data)

n = 601
xmin = 0.
xmax = 15.0
noise = random.normal(scale=.315, size=n)
x     = linspace(xmin, xmax, n)

fit_params = {'a1': Parameter(value=12.0),
              'c1': Parameter(value=5.3),
              'w1': Parameter(value=0.6),
              'a2': Parameter(value=13.0),
              'c2': Parameter(value=8.25),
              'w2': Parameter(value=2.5)}

data  = residual(fit_params, x) + noise

pylab.plot(x, data, 'r+')

fit_params = {'a1': Parameter(value=5.50),
              'c1': Parameter(value=4.4),
              'w1': Parameter(value=0.2),
              'a2': Parameter(value=3.0),
              'c2': Parameter(value=12.0),
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

parnames = sorted(fit_params)
for name, par in fit_params.items():
    print "%s: %.4g +/- %.4g" % (name, par.value, par.stderr)
print 'Correlations:'
for i, name in enumerate(parnames):
    par = fit_params[name]
    for name2 in parnames[i+1:]:
        if name != name2 and name2 in par.correl:
            print '  C(%s, %s) = %.3f ' % (name, name2,
                                           par.correl[name2])
fit = residual(fit_params, x)

pylab.plot(x, fit, 'k-')
pylab.show()





