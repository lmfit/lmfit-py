"""
Simple minimizer  is a wrapper around scipy.leastsq, allowing a
user to build a fitting model as a function of general purpose
Fit Parameters that can be fixed or floated, bounded, and written
as a simple expression of other Fit Parameters.

The user sets up a model in terms of instance of Parameters, writes a
function-to-be-minimized (residual function) in terms of these Parameters.

   Copyright (c) 2011 Matthew Newville, The University of Chicago
   <newville@cars.uchicago.edu>
"""

from numpy import sqrt

from scipy.optimize import leastsq as scipy_leastsq
from scipy.optimize import anneal as scipy_anneal
from scipy.optimize import fmin_l_bfgs_b as scipy_lbfgsb

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from .asteval import Interpreter
from .astutils import NameFinder, valid_symbol_name
    
class Parameters(OrderedDict):
    """a custom dictionary of Parameters.  All keys must be
    strings, and valid Python symbol names, and all values
    must be Parameters.

    Custom methods:
    ---------------

    add()
    add_many()
    """
    def __init__(self, *args, **kwds):
        OrderedDict.__init__(self)
        self.update(*args, **kwds)

    def __setitem__(self, key, value):
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if value is not None and not isinstance(value, Parameter):
            raise ValueError("'%s' is not a Parameter" % value)
        OrderedDict.__setitem__(self, key, value)
        value.name = key

    def add(self, name, value=None, vary=True, expr=None,
            min=None, max=None):
        """convenience function for adding a Parameter:
        with   p = Parameters()
        p.add(name, value=XX, ....)

        is equivalent to
        p[name] = Parameter(name=name, value=XX, ....
        """
        self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                         expr=expr, min=min, max=max))

    def add_many(self, *parlist):
        """convenience function for adding a list of Parameters:
        Here, you must provide a sequence of tuples, each containing
        at least the name. The order in each tuple is the following:
            name, value, vary, min, max, expr
        with   p = Parameters()
        p.add_many( (name1, val1, True, None, None, None),
                    (name2, val2, True,  0.0, None, None),
                    (name3, val3, False, None, None, None),
                    (name4, val4))

        """
        for para in parlist:            
            self.add(*para)

class Parameter(object):
    """A Parameter is the basic Parameter going
    into Fit Model.  The Parameter holds many attributes:
    value, vary, max_value, min_value, constraint expression.
    The value and min/max values will be be set to floats.
    """
    def __init__(self, name=None, value=None, vary=True,
                 min=None, max=None, expr=None, **kws):
        self.name = name
        self.value = value
        self.init_value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.stderr = None
        self.correl = None

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
        s.append("bounds=[%s:%s]" % (repr(self.min), repr(self.max)))
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
    """check for errors derived from asteval, raise MinimizerException"""
    if len(error) > 0:
        msg = []
        for err in error:
            msg = '\n'.join(err.get_error())
        raise MinimizerException(msg)


class Minimizer(object):
    """general minimizer"""
    err_nonparam = \
     "params must be a minimizer.Parameters() instance or list of Parameters()"
    err_maxfev   = """Too many function calls (max set to  %i)!  Use:
    minimize(func, params, ...., maxfev=NNN)
or set  leastsq_kws['maxfev']  to increase this maximum."""

    def __init__(self, userfcn, params, fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **kws):
        self.userfcn = userfcn
        self.__set_params(params)
        self.userargs = fcn_args
        if self.userargs is None:
            self.userargs = []

        self.userkws = fcn_kws
        if self.userkws is None:
            self.userkws = {}
        self.kws = kws
        self.iter_cb = iter_cb
        self.scale_covar = scale_covar
        self.nfev = 0
        self.nfree = 0
        self.message = None
        self.var_map = []
        self.jacfcn = None
        self.asteval = Interpreter()
        self.namefinder = NameFinder()
        self.__prepared = False

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
            val = self.asteval.run(par.ast)
            check_ast_errors(self.asteval.error)
        # apply min/max
        if par.min is not None:
            val = max(val, par.min)
        if par.max is not None:
            val = min(val, par.max)

        self.asteval.symtable[name] = par.value = float(val)
        self.updated[name] = True

    def __residual(self, fvars):
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
        self.nfev = self.nfev + 1

        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

        out = self.userfcn(self.params, *self.userargs, **self.userkws)
        if hasattr(self.iter_cb, '__call__'):
            self.iter_cb(self.params, self.nfev, out,
                         *self.userargs, **self.userkws)
        return out

    def __jacobian(self, fvars):
        """
        analytical jacobian to be used with the Levenberg-Marquardt

        modified 02-01-2012 by Glenn Jones, Aberystwyth University
        """
        for varname, val in zip(self.var_map, fvars):
            self.params[varname].value = val
        self.nfev = self.nfev + 1

        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

        # computing the jacobian
        return self.jacfcn(self.params, *self.userargs, **self.userkws)

    def __set_params(self, params):
        """ set internal self.params from a Parameters object or
        a list/tuple of Parameters"""
        if params is None or isinstance(params, Parameters):
            self.params = params
        elif isinstance(params, (list, tuple)):
            _params = Parameters()
            for _par in params:
                if not isinstance(_par, Parameter):
                    raise MinimizerException(self.err_nonparam)
                else:
                    _params[_par.name] = _par
            self.params = _params
        else:
            raise MinimizerException(self.err_nonparam)

    def prepare_fit(self, params=None):
        """prepare parameters for fit"""
        # determine which parameters are actually variables
        # and which are defined expressions.
        if params is None and self.params is not None and self.__prepared:
            return
        if params is not None and self.params is None:
            self.__set_params(params)
        self.nfev = 0
        self.var_map = []
        self.vars = []
        self.vmin = []
        self.vmax = []
        for name, par in self.params.items():
            if par.expr is not None:
                par.ast = self.asteval.parse(par.expr)
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
                self.vmin.append(par.min)
                self.vmax.append(par.max)

            self.asteval.symtable[name] = par.value
            par.init_value = par.value
            if par.name is None:
                par.name = name

        self.nvarys = len(self.vars)

        # now evaluate make sure initial values
        # are used to set values of the defined expressions.
        # this also acts as a check of expression syntax.
        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)
        self.__prepared = True

    def anneal(self, schedule='cauchy', **kws):
        """
        use simulated annealing
        """
        sched = 'fast'
        if schedule in ('cauchy', 'boltzmann'):
            sched = schedule

        self.prepare_fit()
        sakws = dict(full_output=1, schedule=sched,
                     maxiter = 2000 * (self.nvarys + 1))

        sakws.update(self.kws)
        sakws.update(kws)

        def penalty(params):
            "local penalty function -- anneal wants sum-squares residual"
            r = self.__residual(params)
            return (r*r).sum()

        saout = scipy_anneal(penalty, self.vars, **sakws)
        self.sa_out = saout
        return

    def lbfgsb(self, **kws):
        """
        use l-bfgs-b minimization
        """
        self.prepare_fit()
        lb_kws = dict(factr=1000.0, approx_grad=True, m=20,
                      maxfun = 2000 * (self.nvarys + 1),
                      bounds = zip(self.vmin, self.vmax))
        lb_kws.update(self.kws)
        lb_kws.update(kws)
        def penalty(params):
            "local penalty function -- lbgfsb wants sum-squares residual"
            r = self.__residual(params)
            return (r*r).sum()

        xout, fout, info = scipy_lbfgsb(penalty, self.vars, **lb_kws)

        self.nfev =  info['funcalls']
        self.message = info['task']


    def leastsq(self, scale_covar=True, **kws):
        """
        use Levenberg-Marquardt minimization to perform fit.
        This assumes that ModelParameters have been stored,
        and a function to minimize has been properly set up.

        This wraps scipy.optimize.leastsq, and keyward arguments are passed
        directly as options to scipy.optimize.leastsq

        When possible, this calculates the estimated uncertainties and
        variable correlations from the covariance matrix.

        writes outputs to many internal attributes, and
        returns True if fit was successful, False if not.
        """
        self.prepare_fit()
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7,
                     gtol=1.e-7, maxfev=1000*(self.nvarys+1), Dfun=None)

        lskws.update(self.kws)
        lskws.update(kws)

        if lskws['Dfun'] is not None:
            self.jacfcn = lskws['Dfun']
            lskws['Dfun'] = self.__jacobian

        lsout = scipy_leastsq(self.__residual, self.vars, **lskws)
        vbest, cov, infodict, errmsg, ier = lsout

        self.residual = resid = infodict['fvec']

        self.ier = ier
        self.lmdif_message = errmsg
        self.message = 'Fit succeeded.'
        self.success = ier in [1, 2, 3, 4]

        if ier == 0:
            self.message = 'Invalid Input Parameters.'
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
            self.errorbars = False
            self.message = '%s. Could not estimate error-bars'
        else:
            self.errorbars = True
            self.covar = cov
            if self.scale_covar:
                cov = cov * sum_sqr / self.nfree
            for ivar, varname in enumerate(self.var_map):
                par = self.params[varname]
                par.stderr = sqrt(cov[ivar, ivar])
                par.correl = {}
                for jvar, varn2 in enumerate(self.var_map):
                    if jvar != ivar:
                        par.correl[varn2] = (cov[ivar, jvar]/
                                        (par.stderr * sqrt(cov[jvar, jvar])))

        return self.success

def minimize(fcn, params, engine='leastsq', args=None, kws=None,
             scale_covar=True,
             iter_cb=None, **fit_kws):
    """simple minimization function,
    finding the values for the params which give the
    minimal sum-of-squares of the array return by fcn
    """
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                       iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
    if engine == 'anneal':
        fitter.anneal()
    elif engine == 'lbfgsb':
        fitter.lbfgsb()
    else:
        fitter.leastsq()
    return fitter
