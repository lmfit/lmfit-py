"""
Simple minimizer is a wrapper around scipy.leastsq, allowing a
user to build a fitting model as a function of general purpose
Fit Parameters that can be fixed or floated, bounded, and written
as a simple expression of other Fit Parameters.

The user sets up a model in terms of instance of Parameters, writes a
function-to-be-minimized (residual function) in terms of these Parameters.

   Copyright (c) 2011 Matthew Newville, The University of Chicago
   <newville@cars.uchicago.edu>
"""

from copy import deepcopy
import numpy as np
from numpy import (dot, eye, ndarray, ones_like,
                   sqrt, take, transpose, triu, deprecate)
from numpy.dual import inv
from numpy.linalg import LinAlgError

from scipy.optimize import leastsq as scipy_leastsq
from scipy.optimize import fmin as scipy_fmin
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as scipy_lbfgsb

# differential_evolution is only present in scipy >= 0.15
try:
    from scipy.optimize import differential_evolution as scipy_diffev
except ImportError:
    from ._differentialevolution import differential_evolution as scipy_diffev

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


def asteval_with_uncertainties(*vals, **kwargs):
    """
    given values for variables, calculate object value.
    This is used by the uncertainties package to calculate
    the uncertainty in an object even with a complicated
    expression.
    """
    _obj = kwargs.get('_obj', None)
    _pars = kwargs.get('_pars', None)
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

def _differential_evolution(func, x0, **kwds):
    """
    A wrapper for differential_evolution that can be used with scipy.minimize
    """
    kwargs = dict(args=(), strategy='best1bin', maxiter=None, popsize=15,
                  tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                  callback=None, disp=False, polish=True,
                  init='latinhypercube')

    for k, v in kwds.items():
        if k in kwargs:
            kwargs[k] = v

    return scipy_diffev(func, kwds['bounds'], **kwargs)

SCALAR_METHODS = {'nelder': 'Nelder-Mead',
                  'powell': 'Powell',
                  'cg': 'CG',
                  'bfgs': 'BFGS',
                  'newton': 'Newton-CG',
                  'lbfgs': 'L-BFGS-B',
                  'l-bfgs':'L-BFGS-B',
                  'tnc': 'TNC',
                  'cobyla': 'COBYLA',
                  'slsqp': 'SLSQP',
                  'dogleg': 'dogleg',
                  'trust-ncg': 'trust-ncg',
                  'differential_evolution': 'differential_evolution'}


class Minimizer(object):
    """A general minimizer for curve fitting"""
    err_nonparam = ("params must be a minimizer.Parameters() instance or list "
                    "of Parameters()")
    err_maxfev = ("Too many function calls (max set to %i)!  Use:"
                  " minimize(func, params, ..., maxfev=NNN)"
                  "or set leastsq_kws['maxfev']  to increase this maximum.")

    def __init__(self, userfcn, params, fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **kws):
        """
        Initialization of the Minimzer class

        Parameters
        ----------
        userfcn : callable
            objective function that returns the residual (difference between
            model and data) to be minimized in a least squares sense.  The
            function must have the signature:
            `userfcn(params, *fcn_args, **fcn_kws)`
        params : lmfit.parameter.Parameters object.
            contains the Parameters for the model.
        fcn_args : tuple, optional
            positional arguments to pass to userfcn.
        fcn_kws : dict, optional
            keyword arguments to pass to userfcn.
        iter_cb : callable, optional
            Function to be called at each fit iteration. This function should
            have the signature:
            `iter_cb(params, iter, resid, *fcn_args, **fcn_kws)`,
            where where `params` will have the current parameter values, `iter`
            the iteration, `resid` the current residual array, and `*fcn_args`
            and `**fcn_kws` as passed to the objective function.
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix (leastsq
            only).
        kws : dict, optional
            Options to pass to the minimizer being used.

        Notes
        -----
        The objective function should return the value to be minimized. For the
        Levenberg-Marquardt algorithm from leastsq(), this returned value must
        be an array, with a length greater than or equal to the number of
        fitting variables in the model. For the other methods, the return value
        can either be a scalar or an array. If an array is returned, the sum of
        squares of the array will be sent to the underlying fitting method,
        effectively doing a least-squares optimization of the return values.

        A common use for the fcn_args and fcn_kwds would be to pass in other
        data needed to calculate the residual, including such things as the
        data array, dependent variable, uncertainties in the data, and other
        data structures for the model calculation.
        """
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
        self.ndata = 0
        self.nvarys = 0
        self.ier = 0
        self.success = True
        self.errorbars = False
        self.message = None
        self.lmdif_message = None
        self.chisqr = None
        self.redchi = None
        self.covar = None
        self.residual = None
        self.var_map = []
        self.vars = []
        self.params = []
        self.updated = []
        self.jacfcn = None
        self.asteval = Interpreter()
        self.namefinder = NameFinder()
        self.__prepared = False
        self.__set_params(params)
        # self.prepare_fit()

    @property
    def values(self):
        """
        Returns
        -------
        param_values : dict
            Parameter values in a simple dictionary.
        """

        return dict([(name, p.value) for name, p in self.params.items()])

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
        if getattr(par, 'expr', None) is not None:
            if getattr(par, 'ast', None) is None:
                par.ast = self.asteval.parse(par.expr)
            if par.deps is not None:
                for dep in par.deps:
                    self.__update_paramval(dep)
            par.value = self.asteval.run(par.ast)
            out = check_ast_errors(self.asteval.error)
            if out is not None:
                self.asteval.raise_exception(None)
        self.asteval.symtable[name] = par.value
        self.updated[name] = True

    def update_constraints(self):
        """
        Update all constrained parameters, checking that dependencies are
        evaluated as needed.
        """
        self.updated = dict([(name, False) for name in self.params])
        for name in self.params:
            self.__update_paramval(name)

    def __residual(self, fvars):
        """
        Residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables), this
        evaluates all parameters, including setting bounds and evaluating
        constraints, and then passes those to the user-supplied function to
        calculate the residual.
        """
        # set parameter values
        for varname, val in zip(self.var_map, fvars):
            # self.params[varname].value = val
            par = self.params[varname]
            par.value = par.from_internal(val)
        self.nfev = self.nfev + 1

        self.update_constraints()
        out = self.userfcn(self.params, *self.userargs, **self.userkws)
        if callable(self.iter_cb):
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
            self.params[varname].value = self.params[varname].from_internal(val)

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
        """
        Penalty function for scalar minimizers:

        Parameters
        ----------
        params : lmfit.parameter.Parameters object
            The model parameters

        Returns
        -------
        r - float
            The user evaluated user-supplied objective function. If the
            objective function is an array, return the array sum-of-squares
        """
        r = self.__residual(params)
        if isinstance(r, ndarray):
            r = (r*r).sum()
        return r

    def prepare_fit(self, params=None):
        """
        Prepares parameters for fitting
        """
        # determine which parameters are actually variables
        # and which are defined expressions.
        if params is None and self.params is not None and self.__prepared:
            return
        if params is not None and self.params is None:
            self.__set_params(params)
        self.nfev = 0
        self.var_map = []
        self.vars = []
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

    def unprepare_fit(self):
        """
        Unprepares the fit, so that subsequent fits will be
        forced to run prepare_fit.

        removes ast compilations of constraint expressions
        """
        self.__prepared = False
        self.params = deepcopy(self.params)
        for par in self.params.values():
            if hasattr(par, 'ast'):
                delattr(par, 'ast')

    @deprecate(message='    Deprecated in lmfit 0.8.2, use scalar_minimize '
                       'and method=\'L-BFGS-B\' instead')
    def lbfgsb(self, **kws):
        """
        Use l-bfgs-b minimization

        Parameters
        ----------
        kws : dict
            Minimizer options to pass to the
            scipy.optimize.lbfgsb.fmin_l_bfgs_b function.

        """

        self.prepare_fit()
        lb_kws = dict(factr=1000.0, approx_grad=True, m=20,
                      maxfun=2000 * (self.nvarys + 1),
                      )
        lb_kws.update(self.kws)
        lb_kws.update(kws)

        xout, fout, info = scipy_lbfgsb(self.penalty, self.vars, **lb_kws)
        self.nfev = info['funcalls']
        self.message = info['task']
        self.chisqr = self.residual = self.__residual(xout)
        self.ndata = 1
        self.nfree = 1
        if isinstance(self.residual, ndarray):
            self.chisqr = (self.chisqr**2).sum()
            self.ndata = len(self.residual)
            self.nfree = self.ndata - self.nvarys
        self.redchi = self.chisqr/self.nfree
        self.unprepare_fit()
        return

    @deprecate(message='    Deprecated in lmfit 0.8.2, use scalar_minimize '
                       'and method=\'Nelder-Mead\' instead')
    def fmin(self, **kws):
        """
        Use Nelder-Mead (simplex) minimization

        Parameters
        ----------
        kws : dict
            Minimizer options to pass to the scipy.optimize.fmin minimizer.
        """

        self.prepare_fit()
        fmin_kws = dict(full_output=True, disp=False, retall=True,
                        ftol=1.e-4, xtol=1.e-4,
                        maxfun=5000 * (self.nvarys + 1))

        fmin_kws.update(kws)

        ret = scipy_fmin(self.penalty, self.vars, **fmin_kws)
        xout, fout, niter, funccalls, warnflag, allvecs = ret
        self.nfev = funccalls
        self.chisqr = self.residual = self.__residual(xout)
        self.ndata = 1
        self.nfree = 1
        if isinstance(self.residual, ndarray):
            self.chisqr = (self.chisqr**2).sum()
            self.ndata = len(self.residual)
            self.nfree = self.ndata - self.nvarys
        self.redchi = self.chisqr/self.nfree
        self.unprepare_fit()
        return

    def scalar_minimize(self, method='Nelder-Mead', **kws):
        """
        Use one of the scalar minimization methods from
        scipy.optimize.minimize.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use.
            One of:
                'Nelder-Mead' (default)
                'L-BFGS-B'
                'Powell'
                'CG'
                'Newton-CG'
                'COBYLA'
                'TNC'
                'trust-ncg'
                'dogleg'
                'SLSQP'
                'differential_evolution'

        kws : dict, optional
            Minimizer options pass to scipy.optimize.minimize.

        If the objective function returns a numpy array instead
        of the expected scalar, the sum of squares of the array
        will be used.

        Note that bounds and constraints can be set on Parameters
        for any of these methods, so are not supported separately
        for those designed to use bounds. However, if you use the
        differential_evolution option you must specify finite
        (min, max) for each Parameter.

        Returns
        -------
        success : bool
            Whether the fit was successful.

        """
        if not HAS_SCALAR_MIN:
            raise NotImplementedError
        self.prepare_fit()

        fmin_kws = dict(method=method,
                        options={'maxiter': 1000 * (self.nvarys + 1)})
        fmin_kws.update(self.kws)
        fmin_kws.update(kws)

        # hess supported only in some methods
        if 'hess' in fmin_kws and method not in ('Newton-CG',
                                                 'dogleg', 'trust-ncg'):
            fmin_kws.pop('hess')

        # jac supported only in some methods (and Dfun could be used...)
        if 'jac' not in fmin_kws and fmin_kws.get('Dfun', None) is not None:
            self.jacfcn = fmin_kws.pop('jac')
            fmin_kws['jac'] = self.__jacobian

        if 'jac' in fmin_kws and method not in ('CG', 'BFGS', 'Newton-CG',
                                                'dogleg', 'trust-ncg'):
            self.jacfcn = None
            fmin_kws.pop('jac')

        if method == 'differential_evolution':
            fmin_kws['method'] = _differential_evolution
            pars = self.params
            bounds = [(pars[par].min, pars[par].max) for par in pars]
            if not np.all(np.isfinite(bounds)):
                raise ValueError('With differential evolution finite bounds '
                                 'are required for each parameter')
            bounds = [(-np.pi / 2., np.pi / 2.)] * len(self.vars)
            fmin_kws['bounds'] = bounds

        ret = scipy_minimize(self.penalty, self.vars, **fmin_kws)
        xout = ret.x
        self.message = ret.message

        self.nfev = ret.nfev
        self.chisqr = self.residual = self.__residual(xout)
        self.ndata = 1
        self.nfree = 1
        if isinstance(self.residual, ndarray):
            self.chisqr = (self.chisqr**2).sum()
            self.ndata = len(self.residual)
            self.nfree = self.ndata - self.nvarys
        self.redchi = self.chisqr / self.nfree
        self.unprepare_fit()
        return ret.success

    def leastsq(self, **kws):
        """
        Use Levenberg-Marquardt minimization to perform a fit.
        This assumes that ModelParameters have been stored, and a function to
        minimize has been properly set up.

        This wraps scipy.optimize.leastsq.

        When possible, this calculates the estimated uncertainties and
        variable correlations from the covariance matrix.

        Writes outputs to many internal attributes.

        Parameters
        ----------
        kws : dict, optional
            Minimizer options to pass to scipy.optimize.leastsq.

        Returns
        -------
        success : bool
            True if fit was successful, False if not.
        """
        self.prepare_fit()
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7,
                     gtol=1.e-7, maxfev=2000*(self.nvarys+1), Dfun=None)

        lskws.update(self.kws)
        lskws.update(kws)

        if lskws['Dfun'] is not None:
            self.jacfcn = lskws['Dfun']
            lskws['Dfun'] = self.__jacobian

        # suppress runtime warnings during fit and error analysis
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')
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

        self.nfev = infodict['nfev']
        self.ndata = len(resid)

        sum_sqr = (resid**2).sum()
        self.chisqr = sum_sqr
        self.nfree = (self.ndata - self.nvarys)
        self.redchi = sum_sqr / self.nfree

        # need to map _best values to params, then calculate the
        # grad for the variable parameters
        grad = ones_like(_best)
        vbest = ones_like(_best)

        # ensure that _best, vbest, and grad are not
        # broken 1-element ndarrays.
        if len(np.shape(_best)) == 0:
            _best = np.array([_best])
        if len(np.shape(vbest)) == 0:
            vbest = np.array([vbest])
        if len(np.shape(grad)) == 0:
            grad = np.array([grad])

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
            self.covar = inv(dot(transpose(rvec), rvec))
        except (LinAlgError, ValueError):
            self.covar = None

        has_expr = False
        for par in self.params.values():
            par.stderr, par.correl = 0, None
            has_expr = has_expr or par.expr is not None

        # self.errorbars = error bars were successfully estimated
        self.errorbars = self.covar is not None

        if self.covar is not None:
            if self.scale_covar:
                self.covar = self.covar * sum_sqr / self.nfree

            for ivar, varname in enumerate(self.var_map):
                par = self.params[varname]
                par.stderr = sqrt(self.covar[ivar, ivar])
                par.correl = {}
                try:
                    self.errorbars = self.errorbars and (par.stderr > 0.0)
                    for jvar, varn2 in enumerate(self.var_map):
                        if jvar != ivar:
                            par.correl[varn2] = (self.covar[ivar, jvar] /
                                 (par.stderr * sqrt(self.covar[jvar, jvar])))
                except:
                    self.errorbars = False

            uvars = None
            if has_expr:
                # uncertainties on constrained parameters:
                #   get values with uncertainties (including correlations),
                #   temporarily set Parameter values to these,
                #   re-evaluate contrained parameters to extract stderr
                #   and then set Parameters back to best-fit value
                try:
                    uvars = uncertainties.correlated_values(vbest, self.covar)
                except (LinAlgError, ValueError):
                    uvars = None

                if uvars is not None:
                    for par in self.params.values():
                        eval_stderr(par, uvars, self.var_map,
                                    self.params, self.asteval)
                    # restore nominal values
                    for v, nam in zip(uvars, self.var_map):
                        self.asteval.symtable[nam] = v.nominal_value

        if not self.errorbars:
            self.message = '%s. Could not estimate error-bars'  % self.message

        np.seterr(**orig_warn_settings)
        self.unprepare_fit()
        return self.success

    def minimize(self, method='leastsq'):
        """
        Perform the minimization.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use.
            One of:
            'leastsq'                -    Levenberg-Marquardt (default)
            'nelder'                 -    Nelder-Mead
            'lbfgsb'                 -    L-BFGS-B
            'powell'                 -    Powell
            'cg'                     -    Conjugate-Gradient
            'newton'                 -    Newton-CG
            'cobyla'                 -    Cobyla
            'tnc'                    -    Truncate Newton
            'trust-ncg'              -    Trust Newton-CGn
            'dogleg'                 -    Dogleg
            'slsqp'                  -    Sequential Linear Squares Programming
            'differential_evolution' -    differential evolution

        Returns
        -------
        success : bool
            Whether the fit was successful.
        """

        function = self.leastsq
        kwargs = {}

        user_method = method.lower()
        if user_method.startswith('least'):
            function = self.leastsq
        elif HAS_SCALAR_MIN:
            function = self.scalar_minimize
            for key, val in SCALAR_METHODS.items():
                if (key.lower().startswith(user_method) or
                    val.lower().startswith(user_method)):
                    kwargs['method'] = val
        elif (user_method.startswith('nelder') or
              user_method.startswith('fmin')):
            function = self.fmin
        elif user_method.startswith('lbfgsb'):
            function = self.lbfgsb

        return function(**kwargs)


def minimize(fcn, params, method='leastsq', args=None, kws=None,
             scale_covar=True, iter_cb=None, **fit_kws):
    """
    A general purpose curvefitting function
    The minimize function takes a objective function to be minimized, a
    dictionary (lmfit.parameter.Parameters) containing the model parameters,
    and several optional arguments.

    Parameters
    ----------
    fcn : callable
        objective function that returns the residual (difference between
        model and data) to be minimized in a least squares sense.  The
        function must have the signature:
        `fcn(params, *args, **kws)`
    params : lmfit.parameter.Parameters object.
        contains the Parameters for the model.
    method : str, optional
        Name of the fitting method to use.
        One of:
            'leastsq'                -    Levenberg-Marquardt (default)
            'nelder'                 -    Nelder-Mead
            'lbfgsb'                 -    L-BFGS-B
            'powell'                 -    Powell
            'cg'                     -    Conjugate-Gradient
            'newton'                 -    Newton-CG
            'cobyla'                 -    Cobyla
            'tnc'                    -    Truncate Newton
            'trust-ncg'              -    Trust Newton-CGn
            'dogleg'                 -    Dogleg
            'slsqp'                  -    Sequential Linear Squares Programming
            'differential_evolution' -    differential evolution

    args : tuple, optional
        Positional arguments to pass to fcn.
    kws : dict, optional
        keyword arguments to pass to fcn.
    iter_cb : callable, optional
        Function to be called at each fit iteration. This function should
        have the signature `iter_cb(params, iter, resid, *args, **kws)`,
        where where `params` will have the current parameter values, `iter`
        the iteration, `resid` the current residual array, and `*args`
        and `**kws` as passed to the objective function.
    scale_covar : bool, optional
        Whether to automatically scale the covariance matrix (leastsq
        only).
    fit_kws : dict, optional
        Options to pass to the minimizer being used.

    Notes
    -----
    The objective function should return the value to be minimized. For the
    Levenberg-Marquardt algorithm from leastsq(), this returned value must
    be an array, with a length greater than or equal to the number of
    fitting variables in the model. For the other methods, the return value
    can either be a scalar or an array. If an array is returned, the sum of
    squares of the array will be sent to the underlying fitting method,
    effectively doing a least-squares optimization of the return values.

    A common use for `args` and `kwds` would be to pass in other
    data needed to calculate the residual, including such things as the
    data array, dependent variable, uncertainties in the data, and other
    data structures for the model calculation.
    """
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                       iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
    fitter.minimize(method=method)
    return fitter
