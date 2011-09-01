from lmfit import Parameters, Minimizer
from lmfit.utilfuncs import gauss, loren, pvoigt

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

from testutils import report_errors

import sys

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

HASPYLAB = False

def residual(pars, x, sigma=None, data=None):
    yg = gauss(x, pars['amp_g'].value,
                  pars['cen_g'].value, pars['wid_g'].value)
    
    slope = pars['line_slope'].value
    offset = pars['line_off'].value
    model = yg + offset + x * slope
    if data is None:
        return model
    if sigma is  None:
        return (model - data)

    return (model - data)/sigma
   

n = 201
xmin = 0.
xmax = 20.0
x = linspace(xmin, xmax, n)

p_true = Parameters()
p_true.add('amp_g', value=21.0)
p_true.add('cen_g', value=8.1)
p_true.add('wid_g', value=1.6)
p_true.add('line_off', value=-1.023)
p_true.add('line_slope', value=0.62)

data = (gauss(x, p_true['amp_g'].value, p_true['cen_g'].value,
              p_true['wid_g'].value) +
        random.normal(scale=0.23,  size=n) +
        x*p_true['line_slope'].value + p_true['line_off'].value )

if HASPYLAB:
    pylab.plot(x, data, 'r+')

p_fit = Parameters()
p_fit.add('amp_g', value=10.0)
p_fit.add('cen_g', value=9)
p_fit.add('wid_g', value=1)
p_fit.add('line_slope', value=0.0)
p_fit.add('line_off', value=0.0)

myfit = Minimizer(residual, p_fit,
                  fcn_args=(x,), 
                  fcn_kws={'sigma':0.2, 'data':data})

myfit.prepare_fit()
# 
for scale_covar in (True, False):
    myfit.scale_covar = scale_covar
    print '  ====  scale_covar = ', myfit.scale_covar, ' ==='
    for sigma in (0.1, 0.2, 0.23, 0.5):
        myfit.userkws['sigma'] = sigma

        p_fit['amp_g'].value  = 10
        p_fit['cen_g'].value  =  9
        p_fit['wid_g'].value  =  1
        p_fit['line_slope'].value =0.0
        p_fit['line_off'].value   =0.0

        myfit.leastsq()
        print '  sigma          = ', sigma
        print '  chisqr         = ', myfit.chisqr
        print '  reduced_chisqr = ', myfit.redchi

        report_errors(p_fit, modelpars=p_true, show_correl=False)
        print '  =============================='

        
# if HASPYLAB:
#     fit = residual(p_fit, x)
#     pylab.plot(x, fit, 'k-')
#     pylab.show()
#




