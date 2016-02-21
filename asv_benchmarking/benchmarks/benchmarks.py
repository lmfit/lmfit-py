# Benchmarking scripts for lmfit

import numpy as np
import time
import cProfile, pstats
from subprocess import Popen, PIPE
from lmfit import minimize, Parameters,  __version__


def obj_func(params, x, data):
    """ decaying sine wave, subtract data"""
    amp = params['amp'].value
    shift = params['shift'].value
    omega = params['omega'].value
    decay = params['decay'].value
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

class MinimizeSuite:
    """
    Benchmarks using minimize() and least-squares
    """
    def setup(self):
        pass



    def time_minimize(self):
        x = np.linspace(0, 15, 601)
        np.random.seed(201)

        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.3) )
        params = Parameters()
        params.add('amp',   value= 1,  min=0, max=100)
        params.add('decay', value= 0.0, min=0, max=10)
        params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value= 1.0, min=0, max=10)

        out = minimize(obj_func, params, args=(x, data))
