
from parameter import Parameter

from minimizer import minimize

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq
import pylab

FitParams = {'amp': Parameter(value=14.0),
             'period': Parameter(value=5.33),
             'shift': Parameter(value=0.123),
             'decay': Parameter(value=0.010)}

params = {}
for name, par in FitParams.items():
    params[name] = par.value

FitParams = {'amp': Parameter(value=13.0, vary=False),
             'period': Parameter(value=8.0, vary=True),
             'shift': Parameter(value=0.07),
             'decay': Parameter(value=0.010)}

def residual(params, x, data):
    amp = params['amp']
    per = params['period']
    shift = params['shift']
    decay = params['decay']

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)

    return model - data

n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.5, size=n)
x     = linspace(xmin, xmax, n)
data  = residual(params, x, zeros(n)) + noise

output = minimize(residual, FitParams, args=(x, data))

for name, par in FitParams.items():
    params[name] = par.value
    
fit = residual(params, x, zeros(n)) 

pylab.plot(x, data, 'ro')
pylab.plot(x, fit, 'b')
pylab.show()



          
          
