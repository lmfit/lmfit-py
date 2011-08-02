
from parameter import Parameter

from minimizer import minimize

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq
import pylab

fit_params = {'amp': Parameter(value=14.0),
             'period': Parameter(value=5.33),
             'shift': Parameter(value=0.123),
             'decay': Parameter(value=0.010)}

def residual(pars, x, data=None):
    amp = pars['amp'].value
    per = pars['period'].value
    shift = pars['shift'].value
    decay = pars['decay'].value

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
    if data is None:
        return model
    return (model - data)

n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.5, size=n)
x     = linspace(xmin, xmax, n)
data  = residual(fit_params, x) + noise

fit_params = {'amp': Parameter(value=13.0, vary=True),
             'period': Parameter(value=6.0, vary=True),
             'shift': Parameter(value=0.0),
             'decay': Parameter(value=0.02, vary=True)}

output = minimize(residual, fit_params, args=(x, data))

fit = residual(fit_params, x)

print ' N fev = ', output.nfev

for name, par in fit_params.items():
    print "%s: %.4g +/ %.4g" % (name, par.value, par.stderr)
    for name2, par2 in fit_params.items():
        if name != name2 and par.correl is not None and name2 in par.correl:
            print '  Correl(%s, %s) = %.3f ' % (name, name2, par.correl[name2])
# print output.params

pylab.plot(x, data, 'ro')
pylab.plot(x, fit, 'b')
pylab.show()





