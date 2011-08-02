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
    def __init__(self, userfcn=None, userargs=None, userkws=None,
                 params=None, engine='leastsq', **kws):
        self.userfcn = userfcn
        self.userargs = userargs
        if self.userargs is None:
            self.userargs = []

        self.userkws = userkws
        if self.userkws is None:
            self.userkws = {}

        self.params = params
        self.var_map = []
        self.out = None
        self.engine = engine
        self.asteval = Interpreter()

    def func_wrapper(self, fvars):
        """
        wrapper function for least-squares fit
        """
        # set parameter values
        for varname, val in zip(self.var_map, fvars):
            self.params[varname].value = val

        for name, par in self.params.items():
            val = par.value
            if par.expr is not None:
                val = self.asteval.interp(par.ast)
                check_ast_errors(self.asteval.error)
            if par.min is not None:
                val = max(val, par.min)
            if par.max is not None:
                val = min(val, par.max)

            self.params[name].value = val
            self.asteval.symtable[name] = val

        return self.userfcn(self.params, *self.userargs, **self.userkws)

    def _prep_fit(self):
        """common pre-fit code"""

        # determine which parameters are actually variables
        # and which are defined expressions.
        
        self.var_map = []
        self.vars = []
        for name, param in self.params.items():
            if param.expr is not None:
                param.ast = self.asteval.compile(param.expr)
                check_ast_errors(self.asteval.error)
                param.vary = False
            if param.vary:
                self.var_map.append(name)
                self.vars.append(param.value)
            self.asteval.symtable[name] = param.value
        self.nvarys = len(self.vars)

        
    def fit_wrapper(self, engine=None):
        """
        eventually, we may want to support multiple fitting engines.
        """
        if engine is not None:
            self.engine = engine

        if self.engine == 'leastsq':
            self.fit_leastsq()
        else:
            print('Unknown fit engine')
            
    def fit(self):
        ""
        "perform fit with scipy optimize leastsq (Levenberg-Marquardt)
        """
        
        self._prep_fit()
        lsargs = {'full_output': 1, 'xtol': 1.e-7, 'ftol': 1.e-7,
                  'maxfev': 600 * (self.nvarys + 1)}
        
        lsout = leastsq(self.func_wrapper, self.vars, **lsargs)
        vbest, cov, infodict, errmsg, ier = lsout

        self.ier = ier
        self.errmsg = errmsg
        self.nfev =  infodict['nfev']
        self.residual = resid = infodict['fvec']
        nobs = len(resid)
        
        sum_sqr = (resid**2).sum()
        self.chisqr = sum_sqr
        self.nfree = (nobs - self.nvarys)
        self.redchi = sum_sqr / self.nfree

        if cov is not None:
            cov = cov * sum_sqr / self.nfree
        
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


def minimize(fcn, params, args=None, kws=None, **extrakws):
    fitter = Minimizer(userfcn=fcn, params=params,
                  userargs=args, userkws=kws, **extrakws)
    fitter.fit()
    return fitter
