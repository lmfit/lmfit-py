
from lmfit import minimize, Parameters, Parameter, report_errors
import numpy as np

def residual(params, x, data):
    """ objective function: returns model-data,
    the array to be minimized.
    """
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value
    
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

params = Parameters()
params.add('amp',   value= 10,  min=0)
params.add('decay', value= 0.1) 
params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
params.add('omega', value= 3.0)

x = np.linspace(0, 15, 301)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=len(x), scale=0.2) )

result = minimize(residual, params, args=(x, data))

final = residual(params, x, data)  + data

report_errors(params)

try:
    import pylab
    pylab.plot(x, data, 'k+')
    pylab.plot(x, final, 'r')
    pylab.show()
except:
    pass

