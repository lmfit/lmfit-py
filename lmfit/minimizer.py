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

from numpy import (dot, eye, ndarray, ones_like,
                   sqrt, take, transpose, triu)
from numpy.dual import inv
from numpy.linalg import LinAlgError

from scipy.optimize import leastsq as scipy_leastsq
from scipy.optimize import fmin as scipy_fmin
from scipy.optimize import anneal as scipy_anneal
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as scipy_lbfgsb

# check for scipy.optimize.minimize
HAS_SCALAR_MIN = False
try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCALAR_MIN = True
except ImportError:
    pass

from .asteval import Interpreter
from .astutils import NameFinder
from .parameter import Parameter, Parameters

# use locally modified version of uncertainties package
from . import uncertainties

def asteval_with_uncertainties(*vals,  **kwargs):
    """
    given values for variables, calculate object value.
    This is used by the uncertainties package to calculate
    the uncertainty in an object even with a complicated
    expression.
    """
    _obj   = kwargs.get('_obj', None)
    _pars  = kwargs.get('_pars', None)
    _names = kwargs.get('_names', None)
    _asteval = kwargs.get('_asteval', None)
    if (_obj is None or
        _pars is None or
        _names is None or
        _asteval is None or
        _obj.ast is None):
        return 0
    for val, name in zip(vals, _names):
        _asteval.symtable[name] = val
    return _asteval.eval(_obj.ast)

wrap_ueval = uncertainties.wrap(asteval_with_uncertainties)

def eval_stderr(obj, uvars, _names, _pars, _asteval):
    """evaluate uncertainty and set .stderr for a parameter `obj`
    given the uncertain values `uvars` (a list of uncertainties.ufloats),
    a list of parameter names that matches uvars, and a dict of param
    objects, keyed by name.

    This uses the uncertainties package wrapped function to evaluate
    the uncertainty for an arbitrary expression (in obj.ast) of parameters.
    """
    if not isinstance(obj, Parameter) or not hasattr(obj, 'ast'):
        return
    uval = wrap_ueval(*uvars, _obj=obj, _names=_names,
                      _pars=_pars, _asteval=_asteval)
    try:
        obj.stderr = uval.std_dev()
    except:
        obj.stderr = 0

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
        for err in error:
            msg = '%s: %s' % (err.get_error())
            return msg
    return None


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
        self.__set_params(params)
        self.prepare_fit()

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
        if par.expr is not None:
            for dep in par.deps:
                self.__update_paramval(dep)
            par.value = self.asteval.run(par.ast)
            out = check_ast_errors(self.asteval.error)
            if out is not None:
                self.asteval.raise_exception(None)
        self.asteval.symtable[name] = par.value
        self.updated[name] = True

    def update_constraints(self):
        """update all constrained parameters, checking that
        dependencies are evaluated as needed."""
        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

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
            # self.params[varname].value = val
            par = self.params[varname]
            par.value = par.from_internal(val)
        self.nfev = self.nfev + 1

        self.update_constraints()
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
            # self.params[varname].value = val
            self.params[varname].from_internal(val)

        self.nfev = self.nfev + 1
        self.update_constraints()
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

    def penalty(self, params):
        """penalty function for scalar minimizers:
        evaluates user-supplied objective function,
        if result is an array, return array sum-of-squares.
        """
        r = self.__residual(params)
        if isinstance(r, ndarray):
            r = (r*r).sum()
        return r

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
        self.vmin, self.vmax = [], []
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
                self.vars.append(par.setup_bounds())
                # self.vars.append(par.set_internal_value())
                #self.vmin.append(par.min)
                #self.vmax.append(par.max)

            self.asteval.symtable[name] = par.value
            par.init_value = par.value
            if par.name is None:
                par.name = name

        self.nvarys = len(self.vars)

        # now evaluate make sure initial values
        # are used to set values of the defined expressions.
        # this also acts as a check of expression syntax.
        self.update_constraints()
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
        print("WARNING:  scipy anneal appears unusable!")
        saout = scipy_anneal(self.penalty, self.vars, **sakws)
        self.sa_out = saout
        return

    def lbfgsb(self, **kws):
        """
        use l-bfgs-b minimization
        """
        self.prepare_fit()
        lb_kws = dict(factr=1000.0, approx_grad=True, m=20,
                      maxfun = 2000 * (self.nvarys + 1),
                      # bounds = zip(self.vmin, self.vmax),
                      )
        lb_kws.update(self.kws)
        lb_kws.update(kws)

        xout, fout, info = scipy_lbfgsb(self.penalty, self.vars, **lb_kws)
        self.nfev =  info['funcalls']
        self.message = info['task']
        self.chisqr = (self.penalty(xout)**2).sum()

    def fmin(self, **kws):
        """
        use nelder-mead (simplex) minimization
        """
        self.prepare_fit()
        fmin_kws = dict(full_output=True, disp=False, retall=True,
                        ftol=1.e-4, xtol=1.e-4,
                        maxfun = 5000 * (self.nvarys + 1))

        fmin_kws.update(kws)
        ret = scipy_fmin(self.penalty, self.vars, **fmin_kws)
        xout, fout, iter, funccalls, warnflag, allvecs = ret
        self.nfev =  funccalls
        self.chisqr = (self.penalty(xout)**2).sum()

    def scalar_minimize(self, method='Nelder-Mead', hess=None, tol=None, **kws):
        """use one of the scaler minimization methods from scipy.
        Available methods include:
          Nelder-Mead
          Powell
          CG  (conjugate gradient)
          BFGS
          Newton-CG
          Anneal
          L-BFGS-B
          TNC
          COBYLA
          SLSQP

        If the objective function returns a numpy array instead
        of the expected scalar, the sum of squares of the array
        will be used.

        Note that bounds and constraints can be set on Parameters
        for any of these methods, so are not supported separately
        for those designed to use bounds.

        """
        if not HAS_SCALAR_MIN :
            raise NotImplementedError

        self.prepare_fit()

        maxfev = 1000*(self.nvarys + 1)
        opts = {'maxiter': maxfev}
        if method not in ('L-BFGS-B', 'TNC', 'SLSQP'):
            opts['maxfev'] = maxfev

        fmin_kws = dict(method=method, tol=tol, hess=hess, options=opts)
        fmin_kws.update(self.kws)
        fmin_kws.update(kws)

        ret = scipy_minimize(self.penalty, self.vars, **fmin_kws)
        xout = ret.x
        self.message = ret.message
        self.nfev = ret.nfev
        self.chisqr = (self.penalty(xout)**2).sum()

    def leastsq(self, **kws):
        """
        use Levenberg-Marquardt minimization to perform fit.
        This assumes that ModelParameters have been stored,
        and a function to minimize has been properly set up.

        This wraps scipy.optimize.leastsq, and keyword arguments are passed
        directly as options to scipy.optimize.leastsq

        When possible, this calculates the estimated uncertainties and
        variable correlations from the covariance matrix.

        writes outputs to many internal attributes, and
        returns True if fit was successful, False if not.
        """
        self.prepare_fit()
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7,
                     gtol=1.e-7, maxfev=2000*(self.nvarys+1), Dfun=None)

        lskws.update(self.kws)
        lskws.update(kws)

        if lskws['Dfun'] is not None:
            self.jacfcn = lskws['Dfun']
            lskws['Dfun'] = self.__jacobian

        lsout = scipy_leastsq(self.__residual, self.vars, **lskws)
        _best, _cov, infodict, errmsg, ier = lsout

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

        # need to map _best values to params, then calculate the
        # grad for the variable parameters
        grad = ones_like(_best)
        vbest = ones_like(_best)
        for ivar, varname in enumerate(self.var_map):
            par = self.params[varname]
            grad[ivar] = par.scale_gradient(_best[ivar])
            vbest[ivar] = par.value

        # modified from JJ Helmus' leastsqbound.py

        infodict['fjac'] = transpose(transpose(infodict['fjac']) /
                                     take(grad, infodict['ipvt'] - 1))
        rvec = dot(triu(transpose(infodict['fjac'])[:self.nvarys, :]),
                   take(eye(self.nvarys), infodict['ipvt'] - 1, 0))
        try:
            cov = inv(dot(transpose(rvec), rvec))
        except (LinAlgError, ValueError):
            cov = None

        for par in self.params.values():
            par.stderr, par.correl = 0, None

        self.covar = cov
        if cov is None:
            self.errorbars = False
            self.message = '%s. Could not estimate error-bars'
        else:
            self.errorbars = True
            if self.scale_covar:
                self.covar = cov = cov * sum_sqr / self.nfree

            # uncertainties on constrained parameters:
            #   get values with uncertainties (including correlations),
            #   temporarily set Parameter values to these,
            #   re-evaluate contrained parameters to extract stderr
            #   and then set Parameters back to best-fit value
            uvars = uncertainties.correlated_values(vbest, self.covar)

            for ivar, varname in enumerate(self.var_map):
                par = self.params[varname]
                par.stderr = sqrt(cov[ivar, ivar])
                par.correl = {}
                for jvar, varn2 in enumerate(self.var_map):
                    if jvar != ivar:
                        par.correl[varn2] = (cov[ivar, jvar]/
                                        (par.stderr * sqrt(cov[jvar, jvar])))

            for pname, par in self.params.items():
                eval_stderr(par, uvars, self.var_map,
                            self.params, self.asteval)

            # restore nominal values
            for v, nam in zip(uvars, self.var_map):
                self.asteval.symtable[nam] = v.nominal_value

        for par in self.params.values():
            if hasattr(par, 'ast'):
                delattr(par, 'ast')
        return self.success

def minimize(fcn, params, method='leastsq', args=None, kws=None,
             scale_covar=True, engine=None, iter_cb=None, **fit_kws):
    """simple minimization function,
    finding the values for the params which give the
    minimal sum-of-squares of the array return by fcn
    """
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                       iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)

    _scalar_methods = {'nelder': 'Nelder-Mead',     'powell': 'Powell',
                       'cg': 'CG ',                 'bfgs': 'BFGS',
                       'newton': 'Newton-CG',       'anneal': 'Anneal',
                       'lbfgs': 'L-BFGS-B',         'l-bfgs': 'L-BFGS-B',
                       'tnc': 'TNC',                'cobyla': 'COBYLA',
                       'slsqp': 'SLSQP'}

    _fitmethods = {'anneal': 'anneal',               'nelder': 'fmin',
                   'lbfgsb': 'lbfgsb',               'leastsq': 'leastsq'}

    if engine is not None:
        method = engine
    meth = method.lower()

    fitfunction = None
    kwargs = {}
    # default and most common option: use leastsq method.
    if meth == 'leastsq':
        fitfunction = fitter.leastsq
    else:
        # if scalar_minimize() is supported and method is in list, use it.
        if HAS_SCALAR_MIN:
            for name, method in _scalar_methods.items():
                if meth.startswith(name):
                    fitfunction = fitter.scalar_minimize
                    kwargs = dict(method=method)
        # look for other built-in methods
        if fitfunction is None:
            for name, method in _fitmethods.items():
                if meth.startswith(name):
                    fitfunction = getattr(fitter, method)
    if fitfunction is not None:
        fitfunction(**kwargs)
    return fitter

def make_paras_and_func(fcn, x0, used_kwargs=None):
    """nach 
    A function which takes a function a makes a parameters-dict
    for it.

    Takes the function fcn. A starting guess x0 for the
    non kwargs paramter must be also given. If kwargs
    are used, used_kwargs is dict were the keys are the
    used kwarg and the values are the starting values.
    """
    import inspect
    args = inspect.getargspec(fcn)
    defaults = args[-1]
    len_def = len(defaults) if defaults is not None else 0
    # have_defaults = args[-len(defaults):]
    
    args_without_defaults = len(args[0])- len_def

    if len(x0) < args_without_defaults:
        raise ValueError( 'x0 to short')

    p = Parameters()
    for i, val in enumerate(x0):
        p.add(args[0][i], val)

    if used_kwargs:
        for arg, val in used_kwargs.items():
            p.add(arg, val)
    else:
        used_kwargs = {}

    def func(para):
        "wrapped func"
        kwdict = {}
        
        for arg in used_kwargs.keys():
            kwdict[arg] = para[arg].value

        vals = [para[i].value for i in p]
        return fcn(*vals[:len(x0)], **kwdict)
        
    return p, func



