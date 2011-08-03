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
from asteval import Interpreter, NameFinder

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
    def __init__(self, userfcn, params, fcn_args=None, fcn_kws=None,
                 engine='leastsq', **kws):
        self.userfcn = userfcn
        self.params = params

        self.userargs = fcn_args
        if self.userargs is None:
            self.userargs = []

        self.userkws = fcn_kws
        if self.userkws is None:
            self.userkws = {}

        self.var_map = []
        self.out = None
        self.engine = engine
        self.asteval = Interpreter()
        self.namefinder = NameFinder()

    def __update_paramval(self, name):
        """
        update parameter value, including setting bounds.
        For a constrained parameter (one with an expr defined),
        this first updates (recursively) all parameters on which
        the parameter depends (using the 'deps' field).

       """
        # Has this param already been updated?
        # if this is called as an expression dependency,
        # it may have been!
        if self.updated[name]:
            return

        par = self.params[name]
        val = par.value
        if par.expr is not None:
            for dep in par.deps:
                self.__update_paramval(dep)
            val = self.asteval.interp(par.ast)
            check_ast_errors(self.asteval.error)
        # apply min/max
        if par.min is not None:
            val = max(val, par.min)
        if par.max is not None:
            val = min(val, par.max)

        self.asteval.symtable[name] = par.value = val
        self.updated[name] = True

    def calc_residual(self, fvars):
        """
        residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables),
        this evaluates all parameters, including setting bounds and
        evaluating constraints, and then passes those to the
        user-supplied function to calculate the residual.
        """
        # set parameter values
        for varname, val in zip(self.var_map, fvars):
            self.params[varname].value = val

        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

        return self.userfcn(self.params, *self.userargs, **self.userkws)

    def prepare_fit(self):
        """prepare parameters for fit"""

        # determine which parameters are actually variables
        # and which are defined expressions.
        self.var_map = []
        self.vars = []
        for name, par in self.params.items():
            if par.expr is not None:
                par.ast = self.asteval.compile(par.expr)
                check_ast_errors(self.asteval.error)
                par.vary = False
                par.deps = []
                self.namefinder.names = []
                self.namefinder.generic_visit(par.ast)
                for symname in self.namefinder.names:
                    if (symname in self.params and
                        symname not in par.deps):
                        par.deps.append(symname)

            elif par.vary:
                self.var_map.append(name)
                self.vars.append(par.value)

            self.asteval.symtable[name] = par.value
        self.nvarys = len(self.vars)

        # now evaluate make sure initial values
        # are used to set values of the defined expressions.
        # this also acts as a check of expression syntax.
        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

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
        """
        perform fit with scipy optimize leastsq (Levenberg-Marquardt),
        calculate estimated uncertainties and variable correlations.
        """
        self.prepare_fit()
        lsargs = {'full_output': 1, 'xtol': 1.e-7, 'ftol': 1.e-7,
                  'maxfev': 600 * (self.nvarys + 1)}

        lsout = leastsq(self.calc_residual, self.vars, **lsargs)
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
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws, **extrakws)
    fitter.fit()
    return fitter
