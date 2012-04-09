import sys

from lmfit import Parameters, Parameter, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from testutils import report_errors

try:
    import matplotlib
    matplotlib.use('WXAGG')
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

def per_iteration(pars, i, resid, x, *args, **kws):
    if i < 10 or i % 10 == 0:
        print( '====== Iteration ', i)
        for p in pars.values():
            print( p.name , p.value)

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

if HASPYLAB:
    pylab.plot(x, data, 'r+')

pfit = [Parameter(name='amp_g', value=10),
        Parameter(name='amp_g', value=10.0),
        Parameter(name='cen_g', value=9),
        Parameter(name='wid_g', value=1),
        Parameter(name='frac', value=0.50),
        Parameter(name='amp_l', expr='amp_g'),
        Parameter(name='cen_l', expr='cen_g'),
        Parameter(name='wid_l', expr='wid_g'),
        Parameter(name='line_slope', value=0.0),
        Parameter(name='line_off', value=0.0)]

sigma = 0.021  # estimate of data error (for all data points)

myfit = Minimizer(residual, pfit, iter_cb=per_iteration,
                  fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                  scale_covar=True)

myfit.prepare_fit()
init = residual(myfit.params, x)

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





