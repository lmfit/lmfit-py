import numpy as np
import time
import cProfile, pstats

from lmfit import minimize, Parameters,  __version__

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

def run_fit(nruns=100):
    # create data to be fitted
    x = np.linspace(0, 15, 601)
    np.random.seed(201)
    for i in range(nruns):
        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.3) )
        params = Parameters()
        params.add('amp',   value= 1,  min=0, max=100)
        params.add('decay', value= 0.0, min=0, max=10)
        params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value= 1.0, min=0, max=10)
        out = minimize(fcn2min, params, args=(x, data))

def show_profile(command, filename='_fit_stats.dat'):
    prof = cProfile.run(command, filename=filename)
    stats = pstats.Stats(filename)
    stats.strip_dirs().sort_stats('tottime').print_stats(20)

print(__version__)
show_profile('run_fit()')
