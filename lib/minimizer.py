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
                 params=None, minimizer='leastsq'):
        self.userfcn = userfcn
        self.userargs = userargs
        if self.userargs is None:
            self.userargs = []
        self.params = params
        self.var_map = []
        self.output = None

    def func_wrapper(self, vars):
        """
        wrapper function for least-squares fit
        """
        # unwrap parameters...
        print 'F Wrapper: ', vars

        for varname, value in zip(self.var_map, vars):
            self.params[varname].value = value

        current_params = {}
        for name, par in self.params.items():
            val = par.value
            if par.expr is not None:
                print '%s / Needs Expr Eval / %s' % (name, par.expr)
            if par.min is not None:
                val = max(val, par.min)
            if par.max is not None:
                val = min(val, par.max)

            current_params[name] = val

            # print "Param: %s: %s" % (name, repr(val))
        # call user-function
        return self.userfcn(current_params, *self.userargs)
    
    def runfit(self):
        """run the actual fit."""
        lsargs = {'full_output': 1, 'maxfev': 10000000,
                  'xtol': 1.e-7, 'ftol': 1.e-7}

        # determine which parameters are actually variables
        self.var_map = []
        vars = []
        for pname, param in self.params.items():
            if param.vary:
                self.var_map.append(pname)
                vars.append(param.value)
        
        self.output = leastsq(self.func_wrapper, vars, **lsargs)


def minimize(fcn, params, args=None, **kws):
    print 'Minimize! '
    for nam, par in params.items():
        print nam, par
    print '-------------'
    m = Minimizer(userfcn=fcn, params=params, userargs=args)
    m.runfit()

# lsout = leastsq(misfit, vinit, args=(x, data), full_output=1,
#                 maxfev=1000000, xtol=1.e-4, ftol=1.e-4)
