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
    _asteval = _pars._asteval
    if (_obj is None or  _pars is None or _names is None or
        _asteval is None or _obj._expr_ast is None):
        return 0
    for val, name in zip(vals, _names):
        _asteval.symtable[name] = val
    return _asteval.eval(_obj._expr_ast)

wrap_ueval = uncertainties.wrap(asteval_with_uncertainties)

def eval_stderr(obj, uvars, _names, _pars):
    """evaluate uncertainty and set .stderr for a parameter `obj`
    given the uncertain values `uvars` (a list of uncertainties.ufloats),
    a list of parameter names that matches uvars, and a dict of param
    objects, keyed by name.

    This uses the uncertainties package wrapped function to evaluate the
    uncertainty for an arbitrary expression (in obj._expr_ast) of parameters.
    """
    if not isinstance(obj, Parameter) or getattr(obj, '_expr_ast', None) is None:
        return
    uval = wrap_ueval(*uvars, _obj=obj, _names=_names, _pars=_pars)
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
                  'lbfgsb': 'L-BFGS-B',
                  'l-bfgsb':'L-BFGS-B',
                  'tnc': 'TNC',
                  'cobyla': 'COBYLA',
                  'slsqp': 'SLSQP',
                  'dogleg': 'dogleg',
                  'trust-ncg': 'trust-ncg',
                  'differential_evolution': 'differential_evolution'}


class MinimizerResult(object):
    """ The result of a minimization.

    Attributes
    ----------
    params : Parameters
        The best-fit parameters
    success : bool
        Whether the minimization was successful
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.

    Notes
    -----
    additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)

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
        self.ier = 0
        self._abort = False
        self.success = True
        self.errorbars = False
        self.message = None
        self.lmdif_message = None
        self.chisqr = None
        self.redchi = None
        self.covar = None
        self.residual = None

        self.params = params
        self.jacfcn = None

    @property
    def values(self):
        """
        Returns
        -------
        param_values : dict
            Parameter values in a simple dictionary.
        """

        return dict([(name, p.value) for name, p in self.result.params.items()])

    def __residual(self, fvars):
        """
        Residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables), this
        evaluates all parameters, including setting bounds and evaluating
        constraints, and then passes those to the user-supplied function to
        calculate the residual.
        """
        # set parameter values
        if self._abort:
            return None
        params = self.result.params
        for name, val in zip(self.result.var_names, fvars):
            params[name].value = params[name].from_internal(val)
        self.result.nfev = self.result.nfev + 1

        out = self.userfcn(params, *self.userargs, **self.userkws)
        if callable(self.iter_cb):
            abort = self.iter_cb(params, self.result.nfev, out,
                                 *self.userargs, **self.userkws)
            self._abort = self._abort or abort
        if not self._abort:
            return np.asarray(out).ravel()

    def __jacobian(self, fvars):
        """
        analytical jacobian to be used with the Levenberg-Marquardt

        modified 02-01-2012 by Glenn Jones, Aberystwyth University
        modified 06-29-2015 M Newville to apply gradient scaling
               for bounded variables (thanks to JJ Helmus, N Mayorov)
        """
        pars  = self.result.params
        grad_scale = ones_like(fvars)
        for ivar, name in enumerate(self.result.var_names):
            val = fvars[ivar]
            pars[name].value = pars[name].from_internal(val)
            grad_scale[ivar] = pars[name].scale_gradient(val)

        self.result.nfev = self.result.nfev + 1
        pars.update_constraints()
        # compute the jacobian for "internal" unbounded variables,
        # the rescale for bounded "external" variables.
        jac = self.jacfcn(pars, *self.userargs, **self.userkws)
        if self.col_deriv:
            jac = (jac.transpose()*grad_scale).transpose()
        else:
            jac = jac*grad_scale
        return jac

    def penalty(self, fvars):
        """
        Penalty function for scalar minimizers:

        Parameters
        ----------
        fvars : array of values for the variable parameters

        Returns
        -------
        r - float
            The user evaluated user-supplied objective function. If the
            objective function is an array, return the array sum-of-squares
        """
        r = self.__residual(fvars)
        if isinstance(r, ndarray):
            r = (r*r).sum()
        return r

    def prepare_fit(self, params=None):
        """
        Prepares parameters for fitting,
        return array of initial values
        """
        # determine which parameters are actually variables
        # and which are defined expressions.
        result = self.result = MinimizerResult()
        if params is not None:
            self.params = params
        if isinstance(self.params, Parameters):
            result.params = deepcopy(self.params)
        elif isinstance(self.params, (list, tuple)):
            result.params = Parameters()
            for par in self.params:
                if not isinstance(par, Parameter):
                    raise MinimizerException(self.err_nonparam)
                else:
                    result.params[par.name] = par
        elif self.params is None:
            raise MinimizerException(self.err_nonparam)

        # determine which parameters are actually variables
        # and which are defined expressions.

        result.var_names = [] # note that this *does* belong to self...
        result.init_vals = []
        result.params.update_constraints()
        result.nfev = 0
        result.errorbars = False
        result.aborted = False
        for name, par in self.result.params.items():
            par.stderr = None
            par.correl = None
            if par.expr is not None:
                par.vary = False
            if par.vary:
                result.var_names.append(name)
                result.init_vals.append(par.setup_bounds())

            par.init_value = par.value
            if par.name is None:
                par.name = name
        result.nvarys = len(result.var_names)
        return result

    def unprepare_fit(self):
        """
        Unprepares the fit, so that subsequent fits will be
        forced to run prepare_fit.

        removes ast compilations of constraint expressions
        """
        pass

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
        raise NotImplementedError("use scalar_minimize(method='L-BFGS-B')")


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
        raise NotImplementedError("use scalar_minimize(method='Nelder-Mead')")

    def scalar_minimize(self, method='Nelder-Mead', params=None, **kws):
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

        params : Parameters, optional
           Parameters to use as starting points.
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

        result = self.prepare_fit(params=params)
        vars   = result.init_vals
        params = result.params

        fmin_kws = dict(method=method,
                        options={'maxiter': 1000 * (len(vars) + 1)})
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
            bounds = [(par.min, par.max) for par in params.values()]
            if not np.all(np.isfinite(bounds)):
                raise ValueError('With differential evolution finite bounds '
                                 'are required for each parameter')
            bounds = [(-np.pi / 2., np.pi / 2.)] * len(vars)
            fmin_kws['bounds'] = bounds

            # in scipy 0.14 this can be called directly from scipy_minimize
            # When minimum scipy is 0.14 the following line and the else
            # can be removed.
            ret = _differential_evolution(self.penalty, vars, **fmin_kws)
        else:
            ret = scipy_minimize(self.penalty, vars, **fmin_kws)

        result.aborted = self._abort
        self._abort = False

        for attr in dir(ret):
            if not attr.startswith('_'):
                setattr(result, attr, getattr(ret, attr))

        result.chisqr = result.residual = self.__residual(ret.x)
        result.nvarys = len(vars)
        result.ndata = 1
        result.nfree = 1
        if isinstance(result.residual, ndarray):
            result.chisqr = (result.chisqr**2).sum()
            result.ndata = len(result.residual)
            result.nfree = result.ndata - result.nvarys
        result.redchi = result.chisqr / result.nfree
        _log_likelihood = result.ndata * np.log(result.redchi)
        result.aic = _log_likelihood + 2 * result.nvarys
        result.bic = _log_likelihood + np.log(result.ndata) * result.nvarys

        return result

    def leastsq(self, params=None, **kws):
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
        params : Parameters, optional
           Parameters to use as starting points.
        kws : dict, optional
            Minimizer options to pass to scipy.optimize.leastsq.

        Returns
        -------
        success : bool
            True if fit was successful, False if not.
        """
        result = self.prepare_fit(params=params)
        vars   = result.init_vals
        nvars = len(vars)
        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7, col_deriv=False,
                     gtol=1.e-7, maxfev=2000*(nvars+1), Dfun=None)

        lskws.update(self.kws)
        lskws.update(kws)

        self.col_deriv = False
        if lskws['Dfun'] is not None:
            self.jacfcn = lskws['Dfun']
            self.col_deriv = lskws['col_deriv']
            lskws['Dfun'] = self.__jacobian

        # suppress runtime warnings during fit and error analysis
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')

        lsout = scipy_leastsq(self.__residual, vars, **lskws)
        _best, _cov, infodict, errmsg, ier = lsout
        result.aborted = self._abort
        self._abort = False

        result.residual = resid = infodict['fvec']
        result.ier = ier
        result.lmdif_message = errmsg
        result.message = 'Fit succeeded.'
        result.success = ier in [1, 2, 3, 4]
        if result.aborted:
            result.message = 'Fit aborted by user callback.'
            result.success = False
        elif ier == 0:
            result.message = 'Invalid Input Parameters.'
        elif ier == 5:
            result.message = self.err_maxfev % lskws['maxfev']
        else:
            result.message = 'Tolerance seems to be too small.'

        result.ndata = len(resid)

        result.chisqr = (resid**2).sum()
        result.nfree = (result.ndata - nvars)
        result.redchi = result.chisqr / result.nfree
        _log_likelihood = result.ndata * np.log(result.redchi)
        result.aic = _log_likelihood + 2 * nvars
        result.bic = _log_likelihood + np.log(result.ndata) * nvars

        params = result.params

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

        for ivar, name in enumerate(result.var_names):
            grad[ivar] = params[name].scale_gradient(_best[ivar])
            vbest[ivar] = params[name].value

        # modified from JJ Helmus' leastsqbound.py
        infodict['fjac'] = transpose(transpose(infodict['fjac']) /
                                     take(grad, infodict['ipvt'] - 1))
        rvec = dot(triu(transpose(infodict['fjac'])[:nvars, :]),
                   take(eye(nvars), infodict['ipvt'] - 1, 0))
        try:
            result.covar = inv(dot(transpose(rvec), rvec))
        except (LinAlgError, ValueError):
            result.covar = None

        has_expr = False
        for par in params.values():
            par.stderr, par.correl = 0, None
            has_expr = has_expr or par.expr is not None

        # self.errorbars = error bars were successfully estimated
        result.errorbars = (result.covar is not None)
        if result.aborted:
            result.errorbars = False
        if result.errorbars:
            if self.scale_covar:
                result.covar *= result.redchi
            for ivar, name in enumerate(result.var_names):
                par = params[name]
                par.stderr = sqrt(result.covar[ivar, ivar])
                par.correl = {}
                try:
                    result.errorbars = result.errorbars and (par.stderr > 0.0)
                    for jvar, varn2 in enumerate(result.var_names):
                        if jvar != ivar:
                            par.correl[varn2] = (result.covar[ivar, jvar] /
                                 (par.stderr * sqrt(result.covar[jvar, jvar])))
                except:
                    result.errorbars = False

            uvars = None
            if has_expr:
                # uncertainties on constrained parameters:
                #   get values with uncertainties (including correlations),
                #   temporarily set Parameter values to these,
                #   re-evaluate contrained parameters to extract stderr
                #   and then set Parameters back to best-fit value
                try:
                    uvars = uncertainties.correlated_values(vbest, result.covar)
                except (LinAlgError, ValueError):
                    uvars = None
                if uvars is not None:
                    for par in params.values():
                        eval_stderr(par, uvars, result.var_names, params)
                    # restore nominal values
                    for v, nam in zip(uvars, result.var_names):
                        params[nam].value = v.nominal_value

        if not result.errorbars:
            result.message = '%s. Could not estimate error-bars'% result.message

        np.seterr(**orig_warn_settings)
        return result

    def minimize(self, method='leastsq', params=None, **kws):
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

        params : Parameters, optional
            parameters to use as starting values

        Returns
        -------
        result : MinimizerResult

            MinimizerResult object contains updated params, fit statistics, etc.

        """

        function = self.leastsq
        kwargs = {'params': params}
        kwargs.update(kws)

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
    return fitter.minimize(method=method)
