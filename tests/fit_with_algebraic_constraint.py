import sys

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign

from lmfit import Parameters, Parameter, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt
from lmfit.printfuncs import report_errors

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def residual(pars, x, sigma=None, data=None):
    yg = gauss(x, pars['amp_g'].value,
               pars['cen_g'].value, pars['wid_g'].value)
    yl = loren(x, pars['amp_l'].value,
               pars['cen_l'].value, pars['wid_l'].value)

    slope = pars['line_slope'].value
    offset = pars['line_off'].value
    model =  yg +  yl + offset + x * slope
    if data is None:
        return model
    if sigma is  None:
        return (model - data)
    return (model - data)/sigma


n = 601
xmin = 0.
xmax = 20.0
x = linspace(xmin, xmax, n)

data = (gauss(x, 21, 8.1, 1.2) + 
        loren(x, 10, 9.6, 2.4) +
        random.normal(scale=0.23,  size=n) +
        x*0.5)


if HASPYLAB:
    pylab.plot(x, data, 'r+')

pfit = [Parameter(name='amp_g',  value=10),
        Parameter(name='cen_g',  value=9),
        Parameter(name='wid_g',  value=1),

        Parameter(name='amp_tot',  value=20),
        Parameter(name='amp_l',  expr='amp_tot - amp_g'),
        Parameter(name='cen_l',  expr='1.5+cen_g'),
        Parameter(name='wid_l',  expr='2*wid_g'),
        
        Parameter(name='line_slope', value=0.0),
        Parameter(name='line_off', value=0.0)]

sigma = 0.021  # estimate of data error (for all data points)

myfit = Minimizer(residual, pfit, 
                  fcn_args=(x,), fcn_kws={'sigma':sigma, 'data':data},
                  scale_covar=True)

myfit.prepare_fit()
init = residual(myfit.params, x)

if HASPYLAB:
    pylab.plot(x, init, 'b--')

myfit.leastsq()

print(' Nfev = ', myfit.nfev)
print( myfit.chisqr, myfit.redchi, myfit.nfree)

report_errors(myfit.params)

fit = residual(myfit.params, x)

if HASPYLAB:
    pylab.plot(x, fit, 'k-')
    pylab.show()




