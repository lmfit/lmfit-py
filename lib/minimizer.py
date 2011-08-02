"""
Simple minimizer  is a wrapper around scipy.leastsq, allowing a
user to build a fitting model as a function of general purpose
Fit Parameters that can be fixed or floated, bounded, and written
as a simple expression of other Fit Parameters.

The user sets up a model in terms of a list of Parameters, writes
a function-to-be-minimized (residual function) in terms of the
Parameter list....

"""

from numpy import sqrt
from asteval import Interpreter
from scipy.optimize import leastsq

class ExpressionException(Exception):
    """Expression Syntax Exception"""
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        return "\n%s" % (self.msg)

def check_ast_errors(error):
    if len(error) > 0:
        msg = []
        for err in error:
            msg = '\n'.join(err.get_error())
        raise ExpressionException(msg)


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
        self.asteval = Interpreter()

    def func_wrapper(self, vars):
        """
        wrapper function for least-squares fit
        """
        # set parameter values
        for varname, val in zip(self.var_map, vars):
            self.params[varname].value = val

        asteval = self.asteval
        for name, par in self.params.items():
            val = par.value
            if par.expr is not None:
                val = asteval.interp(par.ast)
                check_ast_errors(asteval.error)
            if par.min is not None:
                val = max(val, par.min)
            if par.max is not None:
                val = min(val, par.max)

            self.params[name].value = val
            asteval.symtable[name] = val

        return self.userfcn(self.params, *self.userargs)

    def runfit(self):
        """run the actual fit."""
        lsargs = {'full_output': 1, 'maxfev': 10000000,
                  'xtol': 1.e-7, 'ftol': 1.e-7}

        # determine which parameters are actually variables
        self.var_map = []
        vars = []
        asteval = self.asteval
        for pname, param in self.params.items():
            if param.expr is not None:
                param.ast = asteval.compile(param.expr)
                check_ast_errors(asteval.error)
                param.vary = False
            if param.vary:
                self.var_map.append(pname)
                vars.append(param.value)
            asteval.symtable[pname] = param.value

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
            if hasattr(par, 'ast'):
                delattr(par, 'ast')

        for ivar, varname in enumerate(self.var_map):
            par = self.params[varname]
            par.stderr = sqrt(cov[ivar, ivar])
            par.correl = {}
            for jvar, varn2 in enumerate(self.var_map):
                if jvar != ivar:
                    par.correl[varn2] = cov[ivar, jvar]/(sqrt(cov[ivar, ivar])*
                                                         sqrt(cov[jvar, jvar]))


def minimize(fcn, params, args=None, **kws):
    m = Minimizer(userfcn=fcn, params=params, userargs=args)
    m.runfit()
    return m
