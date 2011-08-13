from lmfit import Parameters, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

from testutils import report_errors

import sys
if sys.version_info[0] == 2:
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

p_true = Parameters()
p_true.add('amp_g', value=21.0)
p_true.add('cen_g', value=8.1)
p_true.add('wid_g', value=1.6)
p_true.add('frac', value=0.37)
p_true.add('line_off', value=-1.023)
p_true.add('line_slope', value=0.62)

data = (pvoigt(x, p_true['amp_g'].value, p_true['cen_g'].value,
              p_true['wid_g'].value, p_true['frac'].value) +
        random.normal(scale=0.23,  size=n) +
        x*p_true['line_slope'].value + p_true['line_off'].value )

if sys.version_info[0] == 2:
    pylab.plot(x, data, 'r+')

p_fit = Parameters()
p_fit.add('amp_g', value=10.0)
p_fit.add('cen_g', value=9)
p_fit.add('wid_g', value=1)
p_fit.add('frac', value=0.50)
p_fit.add('amp_l', expr='amp_g')
p_fit.add('cen_l', expr='cen_g')
p_fit.add('wid_l', expr='wid_g')
p_fit.add('line_slope', value=0.0)
p_fit.add('line_off', value=0.0)

myfit = Minimizer(residual, p_fit,
                  fcn_args=(x,), fcn_kws={'data':data})

myfit.prepare_fit()
init = residual(p_fit, x)

if sys.version_info[0] == 2:
    pylab.plot(x, init, 'b--')

myfit.leastsq()

print(' Nfev = ', myfit.nfev)
print( myfit.chisqr, myfit.redchi, myfit.nfree)

report_errors(p_fit, modelpars=p_true)

fit = residual(p_fit, x)

if sys.version_info[0] == 2:
    pylab.plot(x, fit, 'k-')
    pylab.show()





