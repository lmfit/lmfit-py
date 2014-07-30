
from lmfit import minimize, Parameters, Parameter, report_fit
import numpy as np
import sys

# Turn off plotting if run by nosetests.
WITHPLOT = True
for arg in sys.argv:
    if 'nose' in arg:
        WITHPLOT = False

if WITHPLOT:
    try:
        import matplotlib
        import pylab
    except ImportError:
        WITHPLOT = False


# create data to be fitted
x = np.linspace(0, 15, 301)
amp, decay, shift, omega = 8,  0.1229, 0.2,  2.75
data = (amp * np.sin(x*omega + shift) * np.exp(-x*x*decay) +
        np.random.normal(size=len(x), scale=0.16) )

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value

    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

# create a set of Parameters
params = Parameters()
params.add('amp',   value= 10,  min=0)
params.add('decay', value= 0.3, min=0.123)
params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
params.add('omega', value= 3.0)

# do fit, here with leastsq model
result = minimize(fcn2min, params, args=(x, data))

# calculate final result
final = data + result.residual

# write error report
report_fit(params, min_correl=0)

print( "Correl: ", params['amp'].correl)

# try to plot results
if WITHPLOT:
    pylab.plot(x, data, 'k+')
    pylab.plot(x, final, 'r')
    pylab.show()

