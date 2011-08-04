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
from astutils import valid_symbol_name

from scipy.optimize import leastsq

class Parameter(object):
    """A Parameter is the basic Parameter going
into Fit Model.  The Parameter holds many attributes:
  value, vary, max_value, min_value, constraint expression.
    """
    def __init__(self, value=None, vary=True, name=None,
                 min=None, max=None, expr=None, **leastsq_kws):
        self.value = value
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr
        self.name = None
        self.stderr = None
        self.correl = None
        self.leastsq_kws = leastsq_kws


    def __repr__(self):
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        val = repr(self.value)
        if self.vary and self.stderr is not None:
            val = "value=%s +/- %.3g" % (repr(self.value), self.stderr)
        elif not self.vary:
            val = "value=%s (fixed)" % (repr(self.value))
        s.append(val)
        s.append("bounds=[%s:%s]" % (repr(self.min),repr(self.max)))
        if self.expr is not None:
            s.append("expr='%s'" % (self.expr))

        return "<Parameter %s>" % ', '.join(s)

class MinimizerException(Exception):
    """General Purpose Exception"""
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
        raise MinimizernException(msg)


class Minimizer(object):
    """general minimizer"""
    err_nondict  = "Parameter argument must be a dictionary"
    err_nonparam = "Param entry for '%%s' must be a lmfit.Parameter"
    err_symname  = "'%%s' is not a valid parameter name"
    err_maxfev   = """Too many function calls (max set to  %%i)!  Use:
    minimize(func, params, ...., maxfev=NNN)
or set  leastsq_kws['maxfev']  to increase this maximum."""
        
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
        try:
            for name, par in self.params.items():
                if not valid_symbol_name(name):
                    raise MinimizerException(self.err_symname % name)
                if not isinstance(par, Parameter):
                    raise MinimizerException(self.err_nonparam % name)
        except TypeError:
            raise MinimizerException(self.err_nondict)
            
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
            if par.name is None:
                par.name = name

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
        lskws = {'full_output': 1, 'xtol': 1.e-7, 'ftol': 1.e-7,
                  'maxfev': 1000 * (self.nvarys + 1)}
        lskws.update(self.leastsq_kws)
        
        lsout = leastsq(self.calc_residual, self.vars, **lskws)
        vbest, cov, infodict, errmsg, ier = lsout

        self.residual = resid = infodict['fvec']

        self.ier = ier
        self.lmdif_message = errmsg
        self.message = 'Fit succeeded'
        self.success = ier in [1, 2, 3, 4]
        
        if ier == 0:
            self.message = 'Invalid Input Parameters'
        elif ier == 5:
            self.message = self.err_maxfev % lskws['maxfev']
        else:
            self.message = 'Tolerance seems to be too small.'
            
        self.nfev =  infodict['nfev']
        self.ndata = len(resid)

        sum_sqr = (resid**2).sum()
        self.chisqr = sum_sqr
        self.nfree = (self.ndata - self.nvarys)
        self.redchi = sum_sqr / self.nfree

        for par in self.params.values():
            par.stderr = 0
            par.correl = None
            if hasattr(par, 'ast'):
                delattr(par, 'ast')

        if cov is None:
            print 'Warning: cannot estimate uncertainties!'
            self.errorbars = False
        else:
            self.errorbars = True
            cov = cov * sum_sqr / self.nfree
            for ivar, varname in enumerate(self.var_map):
                par = self.params[varname]
                par.stderr = sqrt(cov[ivar, ivar])
                par.correl = {}
                for jvar, varn2 in enumerate(self.var_map):
                    if jvar != ivar:
                        par.correl[varn2] = (cov[ivar, jvar]/
                                        (par.stderr * sqrt(cov[jvar, jvar])))

def minimize(fcn, params, args=None, kws=None, **leastsq_kws):
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws, **leastsq_kws)
    fitter.fit()
    return fitter
