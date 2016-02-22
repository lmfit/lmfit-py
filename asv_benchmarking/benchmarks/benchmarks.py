# Benchmarking scripts for lmfit

import numpy as np
import time
import cProfile, pstats
from subprocess import Popen, PIPE
from lmfit import (minimize, Parameters, Minimizer,  __version__,
                   conf_interval)


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
        np.random.seed(201)
        x = np.linspace(0, 15, 601)

        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.3) )
        params = Parameters()
        params.add('amp',   value= 1,  min=0, max=100)
        params.add('decay', value= 0.0, min=0, max=10)
        params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value= 1.0, min=0, max=10)

        out = minimize(obj_func, params, args=(x, data))

    def time_confinterval(self):
        np.random.seed(0)
        x = np.linspace(0.3,10,100)
        y = 1/(0.1*x)+2+0.1*np.random.randn(x.size)

        p = Parameters()
        p.add_many(('a', 0.1), ('b', 1))

        def residual(p):
            a = p['a'].value
            b = p['b'].value

            return 1/(a*x)+b-y

        minimizer = Minimizer(residual, p)
        out = minimizer.leastsq()
        ci = conf_interval(minimizer, out)
