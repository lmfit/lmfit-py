from lmfit import Parameters, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign, where
from scipy.optimize import leastsq

from testutils import report_errors

import sys

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def per_iteration(pars, i, resid, x, *args, **kws):
    if i < 10 or i % 10 == 0:
        print '====== Iteration ', i
        for p in pars.values():
            print p.name , p.value

def residual(pars, x, sigma=None, data=None):
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
    if sigma is  None:
        return (model - data)
    return (model - data)/sigma


n = 601
xmin = 0.
xmax = 20.0
x = linspace(xmin, xmax, n)
noise = random.normal(scale=0.2, size=n)

p_true = Parameters()
frac = 0.37
p_true.add('amp_g', value=21.0)
p_true.add('cen_g', value=8.1)
p_true.add('wid_g', value=1.6)
p_true.add('cen_l', value=13.1)
p_true.add('frac', value=frac)
p_true.add('line_off', value=-1.023)
p_true.add('line_slope', value=0.62)

data = ((1-frac)*gauss(x,     p_true['amp_g'].value,  p_true['cen_g'].value,     p_true['wid_g'].value) +
           frac*loren(x, 0.5*p_true['amp_g'].value,  p_true['cen_l'].value, 2.5*p_true['wid_g'].value) +
        x*p_true['line_slope'].value + p_true['line_off'].value ) + noise

if HASPYLAB:
    pylab.plot(x, data, 'r+')

p_fit = Parameters()
max_x = x[where(data == max(data))][0]
print 'MAX X = ', max_x
p_fit.add('amp_g', value=15.0)
p_fit.add('cen_g', value=max_x)
p_fit.add('wid_g', value=2.0)
p_fit.add('frac',  value=0.50)
p_fit.add('amp_l', expr='0.5*amp_g')
p_fit.add('cen_l', value=12.5)
p_fit.add('wid_l', expr='2.5*wid_g')
p_fit.add('line_slope', value=0.0)
p_fit.add('line_off', value=0.0)

sigma = 0.041  # estimate of data error (for all data points)

myfit = Minimizer(residual, None,            # iter_cb=per_iteration,
                  fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                  scale_covar=True)

myfit.prepare_fit(params=p_fit)
init = residual(p_fit, x)

if HASPYLAB:
    pylab.plot(x, init, 'b--')

myfit.leastsq()

print(' Nfev = ', myfit.nfev)
print( myfit.chisqr, myfit.redchi, myfit.nfree)

report_errors(myfit.params, modelpars=p_true)

fit = residual(myfit.params, x)

if HASPYLAB:
    pylab.plot(x, fit, 'k-')
    pylab.show()





