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
        # set parameter values
        for varname, val in zip(self.var_map, vars):
            self.params[varname].value = val

        for name, par in self.params.items():
            val = par.value
            if par.expr is not None:
                print '%s / Needs Expr Eval / %s' % (name, par.expr)
            if par.min is not None:
                val = max(val, par.min)
            if par.max is not None:
                val = min(val, par.max)

            self.params[name].value = val

        return self.userfcn(self.params, *self.userargs)

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
        
        lsout = leastsq(self.func_wrapper, vars, **lsargs)
        vbest, cov, infodict, errmsg, ier = lsout
            
        if cov is not None:
            resid = self.func_wrapper(vbest)
            cov = cov * (resid**2).sum()/(len(resid)-len(vbest))

        self.nfev =  infodict['nfev']
        self.errmsg = errmsg
        
        for par in self.params.values():
            par.stderr = 0
            par.correl = None

        sqrt = numpy.sqrt
        
        for ivar, varname in enumerate(self.var_map):
            par = self.params[varname]
            par.stderr = sqrt(cov[ivar, ivar])
            par.correl = {}
            for jvar, varn2 in enumerate(self.var_map):
                if jvar != ivar:
                    par.correl[varn2] = cov[ivar, jvar]/( sqrt(cov[ivar, ivar]) *
                                                          sqrt(cov[jvar, jvar]))


def minimize(fcn, params, args=None, **kws):
    m = Minimizer(userfcn=fcn, params=params, userargs=args)
    m.runfit()
    return m

# lsout = leastsq(misfit, vinit, args=(x, data), full_output=1,
#                 maxfev=1000000, xtol=1.e-4, ftol=1.e-4)
