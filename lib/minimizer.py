"""
Simple minimizer  is a wrapper around scipy.leastsq, allowing a
user to build a fitting model as a function of general purpose
Fit Parameters that can be fixed or floated, bounded, and written
as a simple expression of other Fit Parameters.

The user sets up a model in terms of a list of Parameters, writes
a function-to-be-minimized (residual function) in terms of the
Parameter list....

"""

import numpy
from scipy.optimize import leastsq

class Minimizer(object):
    """general minimizer"""
    def __init__(self, userfcn=None, userargs=None,
                 params=None):
        self.userfcn = userfcn
        self.userargs = userargs
        self.params = params

    def func_wrapper(self, vars):
        """
        """
        # unwrap parameters...
        # evaluate parameters
        # set bounds
        # call user-function

    def run_fit(self):
        lsargs = {'full_output': 1, 'maxfev': 10000000,
                  'xtol': 1.e-7, 'ftol': 1.e-7}
        # unpack parameters,
        output = leastsq(self.func_wrapper)

def minimize(fcn, params, args=None, **kws):
    m = Minimizer(userfunc=fcn, params=params)
    m.runfit()

# lsout = leastsq(misfit, vinit, args=(x, data), full_output=1,
#                 maxfev=1000000, xtol=1.e-4, ftol=1.e-4)
