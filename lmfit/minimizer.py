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
import multiprocessing
import numbers

from scipy.optimize import leastsq as scipy_leastsq

# differential_evolution is only present in scipy >= 0.15
try:
    from scipy.optimize import differential_evolution as scipy_diffev
except ImportError:
    from ._differentialevolution import differential_evolution as scipy_diffev

# check for EMCEE
HAS_EMCEE = False
try:
    import emcee as emcee
    HAS_EMCEE = True
except ImportError:
    pass

# check for pandas
HAS_PANDAS = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass

# check for scipy.optimize.minimize
HAS_SCALAR_MIN = False
try:
    from scipy.optimize import minimize as scipy_minimize
    HAS_SCALAR_MIN = True
except ImportError:
    pass

# check for scipy.opitimize.least_squares
HAS_LEAST_SQUARES = False
try:
    from scipy.optimize import least_squares
    HAS_LEAST_SQUARES = True
except ImportError:
    pass

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
    if (_obj is None or _pars is None or _names is None or
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
        return "\n%s" % self.msg


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
                  'l-bfgsb': 'L-BFGS-B',
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
    Additional attributes not listed above may be present, depending on the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)

    @property
    def flatchain(self):
        """
        A flatchain view of the sampling chain from the `emcee` method.
        """
        if hasattr(self, 'chain'):
            if HAS_PANDAS:
                return pd.DataFrame(self.chain.reshape((-1, self.nvarys)),
                                    columns=self.var_names)
            else:
                raise NotImplementedError('Please install Pandas to see the '
                                          'flattened chain')
        else:
            return None


class Minimizer(object):
    """A general minimizer for curve fitting"""
    err_nonparam = ("params must be a minimizer.Parameters() instance or list "
                    "of Parameters()")
    err_maxfev = ("Too many function calls (max set to %i)!  Use:"
                  " minimize(func, params, ..., maxfev=NNN)"
                  "or set leastsq_kws['maxfev']  to increase this maximum.")

    def __init__(self, userfcn, params, fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, nan_policy='raise',
                 **kws):
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
        nan_policy : str, optional
            Specifies action if `userfcn` (or a Jacobian) returns nan
            values. One of:

                'raise' - a `ValueError` is raised
                'propagate' - the values returned from `userfcn` are un-altered
                'omit' - the non-finite values are filtered.

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
        effectively doing a least-squares optimization of the return values. If
        the objective function returns non-finite values then a `ValueError`
        will be raised because the underlying solvers cannot deal with them.

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
        self.nan_policy = nan_policy

    @property
    def values(self):
        """
        Returns
        -------
        param_values : dict
            Parameter values in a simple dictionary.
        """

        return dict([(name, p.value) for name, p in self.result.params.items()])

    def __residual(self, fvars, apply_bounds_transformation=True):
        """
        Residual function used for least-squares fit.
        With the new, candidate values of fvars (the fitting variables), this
        evaluates all parameters, including setting bounds and evaluating
        constraints, and then passes those to the user-supplied function to
        calculate the residual.

        Parameters
        ----------------
        fvars : np.ndarray
            Array of new parameter values suggested by the minimizer.
        apply_bounds_transformation : bool, optional
            If true, apply lmfits parameter transformation to constrain
            parameters. This is needed for solvers without inbuilt support for
            bounds.

        Returns
        -----------
        residuals : np.ndarray
             The evaluated function values for given fvars.
        """
        # set parameter values
        if self._abort:
            return None
        params = self.result.params

        if apply_bounds_transformation:
            for name, val in zip(self.result.var_names, fvars):
                params[name].value = params[name].from_internal(val)
        else:
            for name, val in zip(self.result.var_names, fvars):
                params[name].value = val
        params.update_constraints()

        self.result.nfev += 1

        out = self.userfcn(params, *self.userargs, **self.userkws)
        out = _nan_policy(out, nan_policy=self.nan_policy)

        if callable(self.iter_cb):
            abort = self.iter_cb(params, self.result.nfev, out,
                                 *self.userargs, **self.userkws)
            self._abort = self._abort or abort
        self._abort = self._abort and self.result.nfev > len(fvars)
        if not self._abort:
            return np.asarray(out).ravel()

    def __jacobian(self, fvars):
        """
        analytical jacobian to be used with the Levenberg-Marquardt

        modified 02-01-2012 by Glenn Jones, Aberystwyth University
        modified 06-29-2015 M Newville to apply gradient scaling
               for bounded variables (thanks to JJ Helmus, N Mayorov)
        """
        pars = self.result.params
        grad_scale = ones_like(fvars)
        for ivar, name in enumerate(self.result.var_names):
            val = fvars[ivar]
            pars[name].value = pars[name].from_internal(val)
            grad_scale[ivar] = pars[name].scale_gradient(val)

        self.result.nfev += 1
        pars.update_constraints()

        # compute the jacobian for "internal" unbounded variables,
        # then rescale for bounded "external" variables.
        jac = self.jacfcn(pars, *self.userargs, **self.userkws)
        jac = _nan_policy(jac, nan_policy=self.nan_policy)

        if self.col_deriv:
            jac = (jac.transpose()*grad_scale).transpose()
        else:
            jac *= grad_scale
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
        self.result = MinimizerResult()
        result = self.result
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

        result.var_names = []  # note that this *does* belong to self...
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
        vars = result.init_vals
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
            bounds = np.asarray([(par.min, par.max)
                                 for par in params.values()])
            varying = np.asarray([par.vary for par in params.values()])

            if not np.all(np.isfinite(bounds[varying])):
                raise ValueError('With differential evolution finite bounds '
                                 'are required for each varying parameter')
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
        if isinstance(ret, dict):
            for attr, value in ret.items():
                setattr(result, attr, value)
        else:
            for attr in dir(ret):
                if not attr.startswith('_'):
                    setattr(result, attr, getattr(ret, attr))

        result.x = np.atleast_1d(result.x)
        result.chisqr = result.residual = self.__residual(result.x)
        result.nvarys = len(vars)
        result.ndata = 1
        result.nfree = 1
        if isinstance(result.residual, ndarray):
            result.chisqr = (result.chisqr**2).sum()
            result.ndata = len(result.residual)
            result.nfree = result.ndata - result.nvarys
        result.redchi = result.chisqr / result.nfree
        # this is -2*loglikelihood
        _neg2_log_likel = result.ndata * np.log(result.chisqr / result.ndata)
        result.aic = _neg2_log_likel + 2 * result.nvarys
        result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys

        return result

    def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
              ntemps=1, pos=None, reuse_sampler=False, workers=1,
              float_behavior='posterior', is_weighted=True, seed=None):
        """
        Bayesian sampling of the posterior distribution for the parameters
        using the `emcee` Markov Chain Monte Carlo package. The method assumes
        that the prior is Uniform. You need to have `emcee` installed to use
        this method.

        Parameters
        ----------
        params : lmfit.Parameters, optional
            Parameters to use as starting point. If this is not specified
            then the Parameters used to initialise the Minimizer object are
            used.
        steps : int, optional
            How many samples you would like to draw from the posterior
            distribution for each of the walkers?
        nwalkers : int, optional
            Should be set so :math:`nwalkers >> nvarys`, where `nvarys` are
            the number of parameters being varied during the fit.
            "Walkers are the members of the ensemble. They are almost like
            separate Metropolis-Hastings chains but, of course, the proposal
            distribution for a given walker depends on the positions of all
            the other walkers in the ensemble." - from the `emcee` webpage.
        burn : int, optional
            Discard this many samples from the start of the sampling regime.
        thin : int, optional
            Only accept 1 in every `thin` samples.
        ntemps : int, optional
            If `ntemps > 1` perform a Parallel Tempering.
        pos : np.ndarray, optional
            Specify the initial positions for the sampler.  If `ntemps == 1`
            then `pos.shape` should be `(nwalkers, nvarys)`. Otherwise,
            `(ntemps, nwalkers, nvarys)`. You can also initialise using a
            previous chain that had the same `ntemps`, `nwalkers` and
            `nvarys`. Note that `nvarys` may be one larger than you expect it
            to be if your `userfcn` returns an array and `is_weighted is
            False`.
        reuse_sampler : bool, optional
            If you have already run `emcee` on a given `Minimizer` object then
            it possesses an internal ``sampler`` attribute. You can continue to
            draw from the same sampler (retaining the chain history) if you set
            this option to `True`. Otherwise a new sampler is created. The
            `nwalkers`, `ntemps`, `pos`, and `params` keywords are ignored with
            this option.
            **Important**: the Parameters used to create the sampler must not
            change in-between calls to `emcee`. Alteration of Parameters
            would include changed ``min``, ``max``, ``vary`` and ``expr``
            attributes. This may happen, for example, if you use an altered
            Parameters object and call the `minimize` method in-between calls
            to `emcee`.
        workers : Pool-like or int, optional
            For parallelization of sampling.  It can be any Pool-like object
            with a map method that follows the same calling sequence as the
            built-in `map` function. If int is given as the argument, then a
            multiprocessing-based pool is spawned internally with the
            corresponding number of parallel processes. 'mpi4py'-based
            parallelization and 'joblib'-based parallelization pools can also
            be used here. **Note**: because of multiprocessing overhead it may
            only be worth parallelising if the objective function is expensive
            to calculate, or if there are a large number of objective
            evaluations per step (`ntemps * nwalkers * nvarys`).
        float_behavior : str, optional
            Specifies meaning of the objective function output if it returns a
            float. One of:

                'posterior' - objective function returns a log-posterior
                               probability
                'chi2' - objective function returns :math:`\chi^2`.

            See Notes for further details.
        is_weighted : bool, optional
            Has your objective function been weighted by measurement
            uncertainties? If `is_weighted is True` then your objective
            function is assumed to return residuals that have been divided by
            the true measurement uncertainty `(data - model) / sigma`. If
            `is_weighted is False` then the objective function is assumed to
            return unweighted residuals, `data - model`. In this case `emcee`
            will employ a positive measurement uncertainty during the sampling.
            This measurement uncertainty will be present in the output params
            and output chain with the name `__lnsigma`. A side effect of this
            is that you cannot use this parameter name yourself.
            **Important** this parameter only has any effect if your objective
            function returns an array. If your objective function returns a
            float, then this parameter is ignored. See Notes for more details.
        seed : int or `np.random.RandomState`, optional
            If `seed` is an int, a new `np.random.RandomState` instance is used,
            seeded with `seed`.
            If `seed` is already a `np.random.RandomState` instance, then that
            `np.random.RandomState` instance is used.
            Specify `seed` for repeatable minimizations.

        Returns
        -------
        result : MinimizerResult
            MinimizerResult object containing updated params, statistics,
            etc. The `MinimizerResult` also contains the ``chain``,
            ``flatchain`` and ``lnprob`` attributes. The ``chain``
            and ``flatchain`` attributes contain the samples and have the shape
            `(nwalkers, (steps - burn) // thin, nvarys)` or
            `(ntemps, nwalkers, (steps - burn) // thin, nvarys)`,
            depending on whether Parallel tempering was used or not.
            `nvarys` is the number of parameters that are allowed to vary.
            The ``flatchain`` attribute is a `pandas.DataFrame` of the
            flattened chain, `chain.reshape(-1, nvarys)`. To access flattened
            chain values for a particular parameter use
            `result.flatchain[parname]`. The ``lnprob`` attribute contains the
            log probability for each sample in ``chain``. The sample with the
            highest probability corresponds to the maximum likelihood estimate.

        Notes
        -----
        This method samples the posterior distribution of the parameters using
        Markov Chain Monte Carlo.  To do so it needs to calculate the
        log-posterior probability of the model parameters, `F`, given the data,
        `D`, :math:`\ln p(F_{true} | D)`. This 'posterior probability' is
        calculated as:

        ..math::

        \ln p(F_{true} | D) \propto \ln p(D | F_{true}) + \ln p(F_{true})

        where :math:`\ln p(D | F_{true})` is the 'log-likelihood' and
        :math:`\ln p(F_{true})` is the 'log-prior'. The default log-prior
        encodes prior information already known about the model. This method
        assumes that the log-prior probability is `-np.inf` (impossible) if the
        one of the parameters is outside its limits. The log-prior probability
        term is zero if all the parameters are inside their bounds (known as a
        uniform prior). The log-likelihood function is given by [1]_:

        ..math::

        \ln p(D|F_{true}) = -\frac{1}{2}\sum_n \left[\frac{\left(g_n(F_{true}) - D_n \right)^2}{s_n^2}+\ln (2\pi s_n^2)\right]

        The first summand in the square brackets represents the residual for a
        given datapoint (:math:`g` being the generative model) . This term
        represents :math:`\chi^2` when summed over all datapoints.
        Ideally the objective function used to create `lmfit.Minimizer` should
        return the log-posterior probability, :math:`\ln p(F_{true} | D)`.
        However, since the in-built log-prior term is zero, the objective
        function can also just return the log-likelihood, unless you wish to
        create a non-uniform prior.

        If a float value is returned by the objective function then this value
        is assumed by default to be the log-posterior probability, i.e.
        `float_behavior is 'posterior'`. If your objective function returns
        :math:`\chi^2`, then you should use a value of `'chi2'` for
        `float_behavior`. `emcee` will then multiply your :math:`\chi^2` value
        by -0.5 to obtain the posterior probability.

        However, the default behaviour of many objective functions is to return
        a vector of (possibly weighted) residuals. Therefore, if your objective
        function returns a vector, `res`, then the vector is assumed to contain
        the residuals. If `is_weighted is True` then your residuals are assumed
        to be correctly weighted by the standard deviation of the data points
        (`res = (data - model) / sigma`) and the log-likelihood (and
        log-posterior probability) is calculated as: `-0.5 * np.sum(res **2)`.
        This ignores the second summand in the square brackets. Consequently,
        in order to calculate a fully correct log-posterior probability value
        your objective function should return a single value. If
        `is_weighted is False` then the data uncertainty, `s_n`, will be
        treated as a nuisance parameter and will be marginalised out. This is
        achieved by employing a strictly positive uncertainty
        (homoscedasticity) for each data point, :math:`s_n = exp(__lnsigma)`.
        `__lnsigma` will be present in `MinimizerResult.params`, as well as
        `Minimizer.chain`, `nvarys` will also be increased by one.

        References
        ----------
        .. [1] http://dan.iel.fm/emcee/current/user/line/
        """
        if not HAS_EMCEE:
            raise NotImplementedError('You must have emcee to use'
                                      ' the emcee method')
        tparams = params
        # if you're reusing the sampler then ntemps, nwalkers have to be
        # determined from the previous sampling
        if reuse_sampler:
            if not hasattr(self, 'sampler') or not hasattr(self, '_lastpos'):
                raise ValueError("You wanted to use an existing sampler, but"
                                 "it hasn't been created yet")
            if len(self._lastpos.shape) == 2:
                ntemps = 1
                nwalkers = self._lastpos.shape[0]
            elif len(self._lastpos.shape) == 3:
                ntemps = self._lastpos.shape[0]
                nwalkers = self._lastpos.shape[1]
            tparams = None

        result = self.prepare_fit(params=tparams)
        params = result.params

        # check if the userfcn returns a vector of residuals
        out = self.userfcn(params, *self.userargs, **self.userkws)
        out = np.asarray(out).ravel()
        if out.size > 1 and is_weighted is False:
            # we need to marginalise over a constant data uncertainty
            if '__lnsigma' not in params:
                # __lnsigma should already be in params if is_weighted was
                # previously set to True.
                params.add('__lnsigma', value=0.01, min=-np.inf, max=np.inf, vary=True)
                # have to re-prepare the fit
                result = self.prepare_fit(params)
                params = result.params

        # Removing internal parameter scaling. We could possibly keep it,
        # but I don't know how this affects the emcee sampling.
        bounds = []
        var_arr = np.zeros(len(result.var_names))
        i = 0
        for par in params:
            param = params[par]
            if param.expr is not None:
                param.vary = False
            if param.vary:
                var_arr[i] = param.value
                i += 1
            else:
                # don't want to append bounds if they're not being varied.
                continue

            param.from_internal = lambda val: val
            lb, ub = param.min, param.max
            if lb is None or lb is np.nan:
                lb = -np.inf
            if ub is None or ub is np.nan:
                ub = np.inf
            bounds.append((lb, ub))
        bounds = np.array(bounds)

        self.nvarys = len(result.var_names)

        # set up multiprocessing options for the samplers
        auto_pool = None
        sampler_kwargs = {}
        if type(workers) is int and workers > 1:
            auto_pool = multiprocessing.Pool(workers)
            sampler_kwargs['pool'] = auto_pool
        elif hasattr(workers, 'map'):
            sampler_kwargs['pool'] = workers

        # function arguments for the log-probability functions
        # these values are sent to the log-probability functions by the sampler.
        lnprob_args = (self.userfcn, params, result.var_names, bounds)
        lnprob_kwargs = {'is_weighted': is_weighted,
                         'float_behavior': float_behavior,
                         'userargs': self.userargs,
                         'userkws': self.userkws,
                         'nan_policy': self.nan_policy}

        if ntemps > 1:
            # the prior and likelihood function args and kwargs are the same
            sampler_kwargs['loglargs'] = lnprob_args
            sampler_kwargs['loglkwargs'] = lnprob_kwargs
            sampler_kwargs['logpargs'] = (bounds,)
        else:
            sampler_kwargs['args'] = lnprob_args
            sampler_kwargs['kwargs'] = lnprob_kwargs

        # set up the random number generator
        rng = _make_random_gen(seed)

        # now initialise the samplers
        if reuse_sampler:
            if auto_pool is not None:
                self.sampler.pool = auto_pool

            p0 = self._lastpos
            if p0.shape[-1] != self.nvarys:
                raise ValueError("You cannot reuse the sampler if the number"
                                 "of varying parameters has changed")
        elif ntemps > 1:
            # Parallel Tempering
            # jitter the starting position by scaled Gaussian noise
            p0 = 1 + rng.randn(ntemps, nwalkers, self.nvarys) * 1.e-4
            p0 *= var_arr
            self.sampler = emcee.PTSampler(ntemps, nwalkers, self.nvarys,
                                           _lnpost, _lnprior, **sampler_kwargs)
        else:
            p0 = 1 + rng.randn(nwalkers, self.nvarys) * 1.e-4
            p0 *= var_arr
            self.sampler = emcee.EnsembleSampler(nwalkers, self.nvarys,
                                                 _lnpost, **sampler_kwargs)

        # user supplies an initialisation position for the chain
        # If you try to run the sampler with p0 of a wrong size then you'll get
        # a ValueError. Note, you can't initialise with a position if you are
        # reusing the sampler.
        if pos is not None and not reuse_sampler:
            tpos = np.asfarray(pos)
            if p0.shape == tpos.shape:
                pass
            # trying to initialise with a previous chain
            elif (tpos.shape[0::2] == (nwalkers, self.nvarys)):
                tpos = tpos[:, -1, :]
            # initialising with a PTsampler chain.
            elif ntemps > 1 and tpos.ndim == 4:
                tpos_shape = list(tpos.shape)
                tpos_shape.pop(2)
                if tpos_shape == (ntemps, nwalkers, self.nvarys):
                    tpos = tpos[..., -1, :]
            else:
                raise ValueError('pos should have shape (nwalkers, nvarys)'
                                 'or (ntemps, nwalkers, nvarys) if ntemps > 1')
            p0 = tpos

        # if you specified a seed then you also need to seed the sampler
        if seed is not None:
            self.sampler.random_state = rng.get_state()

        # now do a production run, sampling all the time
        output = self.sampler.run_mcmc(p0, steps)
        self._lastpos = output[0]

        # discard the burn samples and thin
        chain = self.sampler.chain[..., burn::thin, :]
        lnprobability = self.sampler.lnprobability[:, burn::thin]

        flatchain = chain.reshape((-1, self.nvarys))

        quantiles = np.percentile(flatchain, [15.87, 50, 84.13], axis=0)

        for i, var_name in enumerate(result.var_names):
            std_l, median, std_u = quantiles[:, i]
            params[var_name].value = median
            params[var_name].stderr = 0.5 * (std_u - std_l)
            params[var_name].correl = {}

        params.update_constraints()

        # work out correlation coefficients
        corrcoefs = np.corrcoef(flatchain.T)

        for i, var_name in enumerate(result.var_names):
            for j, var_name2 in enumerate(result.var_names):
                if i != j:
                    result.params[var_name].correl[var_name2] = corrcoefs[i, j]

        result.chain = np.copy(chain)
        result.lnprob = np.copy(lnprobability)
        result.errorbars = True
        result.nvarys = len(result.var_names)

        if auto_pool is not None:
            auto_pool.terminate()

        return result

    def least_squares(self, params=None, **kws):
        """
        Use the least_squares (new in scipy 0.17) function to perform a fit.
        This assumes that Parameters have been stored, and a function to
        minimize has been properly set up.

        This wraps scipy.optimize.least_squares, which has inbuilt support
        for bounds and robust loss functions.

        Parameters
        ----------
        params : Parameters, optional
           Parameters to use as starting points.
        kws : dict, optional
            Minimizer options to pass to scipy.optimize.least_squares.
        """

        if not HAS_LEAST_SQUARES:
            raise NotImplementedError("Scipy with a version higher than 0.17 "
                                      "is needed for this method.")

        result = self.prepare_fit(params)

        replace_none = lambda x, sign: sign*np.inf if x is None else x
        upper_bounds = [replace_none(i.max, 1) for i in self.params.values()]
        lower_bounds = [replace_none(i.min, -1) for i in self.params.values()]
        start_vals = [i.value for i in self.params.values()]

        ret = least_squares(self.__residual,
                            start_vals,
                            bounds=(lower_bounds, upper_bounds),
                            kwargs=dict(apply_bounds_transformation=False),
                            **kws
                            )

        for attr in ret:
            setattr(result, attr, ret[attr])

        result.x = np.atleast_1d(result.x)
        result.chisqr = result.residual = self.__residual(result.x, False)
        result.nvarys = len(start_vals)
        result.ndata = 1
        result.nfree = 1
        if isinstance(result.residual, ndarray):
            result.chisqr = (result.chisqr**2).sum()
            result.ndata = len(result.residual)
            result.nfree = result.ndata - result.nvarys
        result.redchi = result.chisqr / result.nfree
        # this is -2*loglikelihood
        _neg2_log_likel = result.ndata * np.log(result.chisqr / result.ndata)
        result.aic = _neg2_log_likel + 2 * result.nvarys
        result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys
        return result

    def leastsq(self, params=None, **kws):
        """
        Use Levenberg-Marquardt minimization to perform a fit.
        This assumes that Parameters have been stored, and a function to
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
        vars = result.init_vals
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
        result.nvarys = nvars
        # this is -2*loglikelihood
        _neg2_log_likel = result.ndata * np.log(result.chisqr / result.ndata)
        result.aic = _neg2_log_likel + 2 * result.nvarys
        result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys

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
            result.message = '%s. Could not estimate error-bars' % result.message

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


def _lnprior(theta, bounds):
    """
    Calculates an improper uniform log-prior probability

    Parameters
    ----------
    theta : sequence
        float parameter values (only those being varied)
    bounds : np.ndarray
        Lower and upper bounds of parameters that are varying.
        Has shape (nvarys, 2).

    Returns
    -------
    lnprob : float
        Log prior probability
    """
    if (np.any(theta > bounds[:, 1])
        or np.any(theta < bounds[:, 0])):
        return -np.inf
    else:
        return 0


def _lnpost(theta, userfcn, params, var_names, bounds, userargs=(),
            userkws=None, float_behavior='posterior', is_weighted=True,
            nan_policy='raise'):
    """
    Calculates the log-posterior probability. See the `Minimizer.emcee` method
    for more details

    Parameters
    ----------
    theta : sequence
        float parameter values (only those being varied)
    userfcn : callable
        User objective function
    params : lmfit.Parameters
        The entire set of Parameters
    var_names : list
        The names of the parameters that are varying
    bounds : np.ndarray
        Lower and upper bounds of parameters. Has shape (nvarys, 2).
    userargs : tuple, optional
        Extra positional arguments required for user objective function
    userkws : dict, optional
        Extra keyword arguments required for user objective function
    float_behavior : str, optional
        Specifies meaning of objective when it returns a float. One of:

        'posterior' - objective function returnins a log-posterior
                      probability.
        'chi2' - objective function returns a chi2 value.

    is_weighted : bool
        If `userfcn` returns a vector of residuals then `is_weighted`
        specifies if the residuals have been weighted by data uncertainties.
    nan_policy : str, optional
        Specifies action if `userfcn` returns nan
        values. One of:

            'raise' - a `ValueError` is raised
            'propagate' - the values returned from `userfcn` are un-altered
            'omit' - the non-finite values are filtered.


    Returns
    -------
    lnprob : float
        Log posterior probability
    """
    # the comparison has to be done on theta and bounds. DO NOT inject theta
    # values into Parameters, then compare Parameters values to the bounds.
    # Parameters values are clipped to stay within bounds.
    if (np.any(theta > bounds[:, 1])
        or np.any(theta < bounds[:, 0])):
        return -np.inf

    for name, val in zip(var_names, theta):
        params[name].value = val

    userkwargs = {}
    if userkws is not None:
        userkwargs = userkws

    # update the constraints
    params.update_constraints()

    # now calculate the log-likelihood
    out = userfcn(params, *userargs, **userkwargs)
    out = _nan_policy(out, nan_policy=nan_policy, handle_inf=False)

    lnprob = np.asarray(out).ravel()

    if lnprob.size > 1:
        # objective function returns a vector of residuals
        if '__lnsigma' in params and not is_weighted:
            # marginalise over a constant data uncertainty
            __lnsigma = params['__lnsigma'].value
            c = np.log(2 * np.pi) + 2 * __lnsigma
            lnprob = -0.5 * np.sum((lnprob / np.exp(__lnsigma)) ** 2 + c)
        else:
            lnprob = -0.5 * (lnprob * lnprob).sum()
    else:
        # objective function returns a single value.
        # use float_behaviour to figure out if the value is posterior or chi2
        if float_behavior == 'posterior':
            pass
        elif float_behavior == 'chi2':
            lnprob *= -0.5
        else:
            raise ValueError("float_behaviour must be either 'posterior' or"
                             " 'chi2' " + float_behavior)

    return lnprob


def _make_random_gen(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _nan_policy(a, nan_policy='raise', handle_inf=True):
    """
    Specifies behaviour when an array contains np.nan or np.inf

    Parameters
    ----------
    a : array_like
        Input array to consider
    nan_policy : str, optional
        One of:

        'raise' - raise a `ValueError` if `a` contains NaN.
        'propagate' - propagate NaN
        'omit' - filter NaN from input array
    handle_inf : bool, optional
        As well as nan consider +/- inf

    Returns
    -------
    filtered_array : array_like
    """
    """
    This function is copied, then modified, from
    scipy/stats/stats.py/_contains_nan
    """

    policies = ['propagate', 'raise', 'omit']

    if handle_inf:
        handler_func =  lambda a: ~np.isfinite(a)
    else:
        handler_func = np.isnan

    if nan_policy == 'propagate':
        # nan values are ignored.
        return a
    elif nan_policy == 'raise':
        try:
            # Calling np.sum to avoid creating a huge array into memory
            # e.g. np.isnan(a).any()
            with np.errstate(invalid='ignore'):
                contains_nan = handler_func(np.sum(a))
        except TypeError:
            # If the check cannot be properly performed we fallback to omiting
            # nan values and raising a warning. This can happen when attempting to
            # sum things that are not numbers (e.g. as in the function `mode`).
            contains_nan = False
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

        if contains_nan:
            raise ValueError("The input contains nan values")
        return a

    elif nan_policy == 'omit':
        # nans are filtered
        mask = handler_func(a)
        return a[~mask]
    else:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))


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
