"""Implementation of the Minimizer class and fitting algorithms.

Simple minimizer is a wrapper around scipy.leastsq, allowing a user to
build a fitting model as a function of general purpose fit Parameters that
can be fixed or varied, bounded, and written as a simple expression of
other fit Parameters.

The user sets up a model in terms of instance of Parameters and writes a
function-to-be-minimized (residual function) in terms of these Parameters.

Original copyright:
   Copyright (c) 2011 Matthew Newville, The University of Chicago

See LICENSE for more complete authorship information and license.

"""
from collections import namedtuple
from copy import deepcopy
import inspect
import multiprocessing
import numbers
import warnings

import numpy as np
from scipy.linalg import LinAlgError, inv
from scipy.optimize import basinhopping as scipy_basinhopping
from scipy.optimize import brute as scipy_brute
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing as scipy_dual_annealing
from scipy.optimize import least_squares
from scipy.optimize import leastsq as scipy_leastsq
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import shgo as scipy_shgo
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.stats import cauchy as cauchy_dist
from scipy.stats import norm as norm_dist
import uncertainties

from ._ampgo import ampgo
from .parameter import Parameter, Parameters
from .printfuncs import fitreport_html_table

# check for EMCEE
try:
    import emcee
    from emcee.autocorr import AutocorrError
    HAS_EMCEE = int(emcee.__version__[0]) >= 3
except ImportError:
    HAS_EMCEE = False

# check for pandas
try:
    import pandas as pd
    from pandas import isnull
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    isnull = np.isnan

# check for numdifftools
try:
    import numdifftools as ndt
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False

# check for dill
try:
    import dill  # noqa: F401
    HAS_DILL = True
except ImportError:
    HAS_DILL = False


# define the namedtuple here so pickle will work with the MinimizerResult
Candidate = namedtuple('Candidate', ['params', 'score'])

maxeval_warning = "ignoring `{}` argument to `{}()`. Use `max_nfev` instead."


def thisfuncname():
    """Return the name of calling function."""
    try:
        return inspect.stack()[1].function
    except AttributeError:
        return inspect.stack()[1][3]


def asteval_with_uncertainties(*vals, **kwargs):
    """Calculate object value, given values for variables.

    This is used by the uncertainties package to calculate the
    uncertainty in an object even with a complicated expression.

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
    """Evaluate uncertainty and set ``.stderr`` for a parameter `obj`.

    Given the uncertain values `uvars` (list of `uncertainties.ufloats`),
    a list of parameter names that matches `uvars`, and a dictionary of
    parameter objects, keyed by name.

    This uses the uncertainties package wrapped function to evaluate the
    uncertainty for an arbitrary expression (in ``obj._expr_ast``) of
    parameters.

    """
    if not isinstance(obj, Parameter) or getattr(obj, '_expr_ast', None) is None:
        return
    uval = wrap_ueval(*uvars, _obj=obj, _names=_names, _pars=_pars)
    try:
        obj.stderr = uval.std_dev
    except Exception:
        obj.stderr = 0


class MinimizerException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return f"{self.msg}"


class AbortFitException(MinimizerException):
    """Raised when a fit is aborted by the user."""


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
                  'differential_evolution': 'differential_evolution',
                  'trust-constr': 'trust-constr',
                  'trust-exact': 'trust-exact',
                  'trust-krylov': 'trust-krylov'}


def reduce_chisquare(r):
    """Reduce residual array to scalar (chi-square).

    Calculate the chi-square value from residual array `r` as
    ``(r*r).sum()``.

    Parameters
    ----------
    r : numpy.ndarray
        Residual array.

    Returns
    -------
    float
        Chi-square calculated from the residual array.

    """
    return (r*r).sum()


def reduce_negentropy(r):
    """Reduce residual array to scalar (negentropy).

    Reduce residual array `r` to scalar using negative entropy and the
    normal (Gaussian) probability distribution of `r` as pdf:

       ``(norm.pdf(r)*norm.logpdf(r)).sum()``

    since ``pdf(r) = exp(-r*r/2)/sqrt(2*pi)``, this is
       ``((r*r/2 - log(sqrt(2*pi))) * exp(-r*r/2)).sum()``.

    Parameters
    ----------
    r : numpy.ndarray
        Residual array.

    Returns
    -------
    float
        Negative entropy value calculated from the residual array.

    """
    return (norm_dist.pdf(r)*norm_dist.logpdf(r)).sum()


def reduce_cauchylogpdf(r):
    """Reduce residual array to scalar (cauchylogpdf).

    Reduce residual array `r` to scalar using negative log-likelihood and
    a Cauchy (Lorentzian) distribution of `r`:

       ``-scipy.stats.cauchy.logpdf(r)``

    where the Cauchy pdf = ``1/(pi*(1+r*r))``.

    This gives better suppression of outliers compared to the default
    sum-of-squares.

    Parameters
    ----------
    r : numpy.ndarray
        Residual array.

    Returns
    -------
    float
        Negative entropy value calculated from the residual array.

    """
    return -cauchy_dist.logpdf(r).sum()


class MinimizerResult:
    r"""The results of a minimization.

    Minimization results include data such as status and error messages,
    fit statistics, and the updated (i.e., best-fit) parameters themselves
    in the :attr:`params` attribute.

    The list of (possible) `MinimizerResult` attributes is given below:

    Attributes
    ----------
    params : Parameters
        The best-fit parameters resulting from the fit.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    var_names : list
        Ordered list of variable parameter names used in optimization, and
        useful for understanding the values in :attr:`init_vals` and
        :attr:`covar`.
    covar : numpy.ndarray
        Covariance matrix from minimization, with rows and columns
        corresponding to :attr:`var_names`.
    init_vals : list
        List of initial values for variable parameters using
        :attr:`var_names`.
    init_values : dict
        Dictionary of initial values for variable parameters.
    nfev : int
        Number of function evaluations.
    success : bool
        True if the fit succeeded, otherwise False.
    errorbars : bool
        True if uncertainties were estimated, otherwise False.
    message : str
        Message about fit success.
    call_kws : dict
        Keyword arguments sent to underlying solver.
    ier : int
        Integer error value from :scipydoc:`optimize.leastsq` (`'leastsq'`
        method only).
    lmdif_message : str
        Message from :scipydoc:`optimize.leastsq` (`'leastsq'` method only).
    nvarys : int
        Number of variables in fit: :math:`N_{\rm varys}`.
    ndata : int
        Number of data points: :math:`N`.
    nfree : int
        Degrees of freedom in fit: :math:`N - N_{\rm varys}`.
    residual : numpy.ndarray
        Residual array :math:`{\rm Resid_i}`. Return value of the objective
        function when using the best-fit values of the parameters.
    chisqr : float
        Chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`.
    redchi : float
        Reduced chi-square:
        :math:`\chi^2_{\nu}= {\chi^2} / {(N - N_{\rm varys})}`.
    aic : float
        Akaike Information Criterion statistic:
        :math:`N \ln(\chi^2/N) + 2 N_{\rm varys}`.
    bic : float
        Bayesian Information Criterion statistic:
        :math:`N \ln(\chi^2/N) + \ln(N) N_{\rm varys}`.
    flatchain : pandas.DataFrame
        A flatchain view of the sampling chain from the `emcee` method.

    Methods
    -------
    show_candidates
        :meth:`pretty_print` representation of candidates from the `brute`
        fitting method.

    """

    def __init__(self, **kws):
        for key, val in kws.items():
            setattr(self, key, val)

    @property
    def flatchain(self):
        """Show flatchain view of the sampling chain from `emcee` method."""
        if hasattr(self, 'chain'):
            if HAS_PANDAS:
                if len(self.chain.shape) == 4:
                    return pd.DataFrame(self.chain[0, ...].reshape((-1, self.nvarys)),
                                        columns=self.var_names)
                elif len(self.chain.shape) == 3:
                    return pd.DataFrame(self.chain.reshape((-1, self.nvarys)),
                                        columns=self.var_names)
            else:
                raise NotImplementedError('Please install Pandas to see the '
                                          'flattened chain')
        else:
            return None

    def show_candidates(self, candidate_nmb='all'):
        """Show pretty_print() representation of candidates.

        Showing all stored candidates (default) or the specified
        candidate-# obtained from the `brute` fitting method.

        Parameters
        ----------
        candidate_nmb : int or 'all', optional
            The candidate-number to show using the :meth:`pretty_print`
            method (default is 'all').

        """
        if hasattr(self, 'candidates'):
            if candidate_nmb == 'all':
                for i, candidate in enumerate(self.candidates):
                    print(f"\nCandidate #{i + 1}, chisqr = {candidate.score:.3f}")
                    candidate.params.pretty_print()
            elif (candidate_nmb < 1 or candidate_nmb > len(self.candidates)):
                raise ValueError(f"'candidate_nmb' should be between 1 and {len(self.candidates)}.")
            else:
                candidate = self.candidates[candidate_nmb-1]
                print(f"\nCandidate #{candidate_nmb}, chisqr = {candidate.score:.3f}")
                candidate.params.pretty_print()

    def _calculate_statistics(self):
        """Calculate the fitting statistics."""
        self.nvarys = len(self.init_vals)
        if not hasattr(self, 'residual'):
            self.residual = -np.inf
        if isinstance(self.residual, np.ndarray):
            self.chisqr = (self.residual**2).sum()
            self.ndata = len(self.residual)
            self.nfree = self.ndata - self.nvarys
        else:
            self.chisqr = self.residual
            self.ndata = 1
            self.nfree = 1
        self.redchi = self.chisqr / max(1, self.nfree)
        # this is -2*loglikelihood
        self.chisqr = max(self.chisqr, 1.e-250*self.ndata)
        _neg2_log_likel = self.ndata * np.log(self.chisqr / self.ndata)
        self.aic = _neg2_log_likel + 2 * self.nvarys
        self.bic = _neg2_log_likel + np.log(self.ndata) * self.nvarys

    def _repr_html_(self, show_correl=True, min_correl=0.1):
        """Return a HTML representation of parameters data."""
        return fitreport_html_table(self, show_correl=show_correl,
                                    min_correl=min_correl)


class Minimizer:
    """A general minimizer for curve fitting and optimization."""

    _err_nonparam = ("params must be a minimizer.Parameters() instance or"
                     " list of Parameters()")
    _err_max_evals = ("Too many function calls (max set to {:i})! Use:"
                      " minimize(func, params, ..., max_nfev=NNN)"
                      " to increase this maximum.")

    def __init__(self, userfcn, params, fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, nan_policy='raise',
                 reduce_fcn=None, calc_covar=True, max_nfev=None, **kws):
        """
        Parameters
        ----------
        userfcn : callable
            Objective function that returns the residual (difference
            between model and data) to be minimized in a least-squares
            sense. This function must have the signature::

                userfcn(params, *fcn_args, **fcn_kws)

        params : Parameters
            Contains the Parameters for the model.
        fcn_args : tuple, optional
            Positional arguments to pass to `userfcn`.
        fcn_kws : dict, optional
            Keyword arguments to pass to `userfcn`.
        iter_cb : callable, optional
            Function to be called at each fit iteration. This function
            should have the signature::

                iter_cb(params, iter, resid, *fcn_args, **fcn_kws)

            where `params` will have the current parameter values, `iter`
            the iteration number, `resid` the current residual array, and
            `*fcn_args` and `**fcn_kws` are passed to the objective
            function.
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix (default
            is True).
        nan_policy : {'raise', 'propagate', 'omit}, optional
            Specifies action if `userfcn` (or a Jacobian) returns NaN
            values. One of:

            - `'raise'` : a `ValueError` is raised (default)
            - `'propagate'` : the values returned from `userfcn` are un-altered
            - `'omit'` : non-finite values are filtered

        reduce_fcn : str or callable, optional
            Function to convert a residual array to a scalar value for the
            scalar minimizers. Optional values are (where `r` is the
            residual array):

            - None : sum-of-squares of residual (default)

               = (r*r).sum()

            - `'negentropy'` : neg entropy, using normal distribution

               = rho*log(rho).sum()`, where rho = exp(-r*r/2)/(sqrt(2*pi))

            - `'neglogcauchy'` : neg log likelihood, using Cauchy distribution

               = -log(1/(pi*(1+r*r))).sum()

            - callable : must take one argument (`r`) and return a float.

        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than ``'leastsq'`` and ``'least_squares'``.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        **kws : dict, optional
            Options to pass to the minimizer being used.

        Notes
        -----
        The objective function should return the value to be minimized.
        For the Levenberg-Marquardt algorithm from :meth:`leastsq` or
        :meth:`least_squares`, this returned value must be an array, with
        a length greater than or equal to the number of fitting variables
        in the model. For the other methods, the return value can either be
        a scalar or an array. If an array is returned, the sum-of-squares
        of the array will be sent to the underlying fitting method,
        effectively doing a least-squares optimization of the return
        values. If the objective function returns non-finite values then a
        `ValueError` will be raised because the underlying solvers cannot
        deal with them.

        A common use for the `fcn_args` and `fcn_kws` would be to pass in
        other data needed to calculate the residual, including such things
        as the data array, dependent variable, uncertainties in the data,
        and other data structures for the model calculation.

        """
        self.userfcn = userfcn
        self.userargs = fcn_args
        if self.userargs is None:
            self.userargs = []

        self.userkws = fcn_kws
        if self.userkws is None:
            self.userkws = {}
        for maxnfev_alias in ('maxfev', 'maxiter'):
            if maxnfev_alias in kws:
                warnings.warn(maxeval_warning.format(maxnfev_alias, 'Minimizer'),
                              RuntimeWarning)
                kws.pop(maxnfev_alias)

        self.kws = kws
        self.iter_cb = iter_cb
        self.calc_covar = calc_covar
        self.scale_covar = scale_covar
        self.max_nfev = max_nfev
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
        self.reduce_fcn = reduce_fcn
        self.params = params
        self.col_deriv = False
        self.jacfcn = None
        self.nan_policy = nan_policy

    def set_max_nfev(self, max_nfev=None, default_value=100000):
        """Set maximum number of function evaluations.

        If `max_nfev` is None, use the provided `default_value`.

        >>> self.set_max_nfev(max_nfev, 2000*(result.nvarys+1))

        """
        if max_nfev is not None:
            self.max_nfev = max_nfev
        if self.max_nfev in (None, np.inf):
            self.max_nfev = default_value

    @property
    def values(self):
        """Return Parameter values in a simple dictionary."""
        return {name: p.value for name, p in self.result.params.items()}

    def __residual(self, fvars, apply_bounds_transformation=True):
        """Residual function used for least-squares fit.

        With the new, candidate values of `fvars` (the fitting variables),
        this evaluates all parameters, including setting bounds and
        evaluating constraints, and then passes those to the user-supplied
        function to calculate the residual.

        Parameters
        ----------
        fvars : numpy.ndarray
            Array of new parameter values suggested by the minimizer.
        apply_bounds_transformation : bool, optional
            Whether to apply lmfits parameter transformation to constrain
            parameters (default is True). This is needed for solvers
            without built-in support for bounds.

        Returns
        -------
        numpy.ndarray
             The evaluated function values for given `fvars`.

        """
        params = self.result.params

        if fvars.shape == ():
            fvars = fvars.reshape((1,))

        if apply_bounds_transformation:
            for name, val in zip(self.result.var_names, fvars):
                params[name].value = params[name].from_internal(val)
        else:
            for name, val in zip(self.result.var_names, fvars):
                params[name].value = val
        params.update_constraints()

        if self.max_nfev is None:
            self.max_nfev = 200000*(len(fvars)+1)

        self.result.nfev += 1
        self.result.last_internal_values = fvars
        if self.result.nfev > self.max_nfev:
            self.result.aborted = True
            m = f"number of function evaluations > {self.max_nfev}"
            self.result.message = f"Fit aborted: {m}"
            self.result.success = False
            raise AbortFitException(f"fit aborted: too many function evaluations {self.max_nfev}")

        out = self.userfcn(params, *self.userargs, **self.userkws)

        if callable(self.iter_cb):
            abort = self.iter_cb(params, self.result.nfev, out,
                                 *self.userargs, **self.userkws)
            self._abort = self._abort or abort

        if self._abort:
            self.result.residual = out
            self.result.aborted = True
            self.result.message = "Fit aborted by user callback. Could not estimate error-bars."
            self.result.success = False
            raise AbortFitException("fit aborted by user.")
        else:
            return _nan_policy(np.asarray(out).ravel(),
                               nan_policy=self.nan_policy)

    def __jacobian(self, fvars):
        """Return analytical jacobian to be used with Levenberg-Marquardt.

        modified 02-01-2012 by Glenn Jones, Aberystwyth University
        modified 06-29-2015 by M Newville to apply gradient scaling for
        bounded variables (thanks to JJ Helmus, N Mayorov)

        """
        pars = self.result.params
        grad_scale = np.ones_like(fvars)
        for ivar, name in enumerate(self.result.var_names):
            val = fvars[ivar]
            pars[name].value = pars[name].from_internal(val)
            grad_scale[ivar] = pars[name].scale_gradient(val)

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
        """Penalty function for scalar minimizers.

        Parameters
        ----------
        fvars : numpy.ndarray
            Array of values for the variable parameters.

        Returns
        -------
        r : float
            The evaluated user-supplied objective function.

            If the objective function is an array of size greater than 1,
            use the scalar returned by `self.reduce_fcn`. This defaults to
            sum-of-squares, but can be replaced by other options.

        """
        if self.result.method in ['brute', 'shgo', 'dual_annealing']:
            apply_bounds_transformation = False
        else:
            apply_bounds_transformation = True

        r = self.__residual(fvars, apply_bounds_transformation)
        if isinstance(r, np.ndarray) and r.size > 1:
            r = self.reduce_fcn(r)
            if isinstance(r, np.ndarray) and r.size > 1:
                r = r.sum()
        return r

    def prepare_fit(self, params=None):
        """Prepare parameters for fitting.

        Prepares and initializes model and Parameters for subsequent
        fitting. This routine prepares the conversion of
        :class:`Parameters` into fit variables, organizes parameter bounds,
        and parses, "compiles" and checks constrain expressions. The method
        also creates and returns a new instance of a
        :class:`MinimizerResult` object that contains the copy of the
        Parameters that will actually be varied in the fit.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model; if None, then the
            Parameters used to initialize the Minimizer object are used.

        Returns
        -------
        MinimizerResult

        Notes
        -----
        This method is called directly by the fitting methods, and it is
        generally not necessary to call this function explicitly.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.

        """
        self._abort = False

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
                    raise MinimizerException(self._err_nonparam)
                else:
                    result.params[par.name] = par
        elif self.params is None:
            raise MinimizerException(self._err_nonparam)

        # determine which parameters are actually variables
        # and which are defined expressions.
        result.var_names = []  # note that this *does* belong to self...
        result.init_vals = []
        result.params.update_constraints()
        result.nfev = 0
        result.call_kws = {}
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
        result.init_values = {n: v for n, v in zip(result.var_names,
                                                   result.init_vals)}

        # set up reduce function for scalar minimizers
        #    1. user supplied callable
        #    2. string starting with 'neglogc' or 'negent'
        #    3. sum-of-squares
        if not callable(self.reduce_fcn):
            if isinstance(self.reduce_fcn, str):
                if self.reduce_fcn.lower().startswith('neglogc'):
                    self.reduce_fcn = reduce_cauchylogpdf
                elif self.reduce_fcn.lower().startswith('negent'):
                    self.reduce_fcn = reduce_negentropy
            if self.reduce_fcn is None:
                self.reduce_fcn = reduce_chisquare
        return result

    def unprepare_fit(self):
        """Clean fit state.

        This methods Removes AST compilations of constraint expressions.
        Subsequent fits will need to call prepare_fit().


        """

    def _calculate_covariance_matrix(self, fvars):
        """Calculate the covariance matrix.

        The ``numdiftoools`` package is used to estimate the Hessian
        matrix, and the covariance matrix is calculated as:

        .. math::

            cov_x = inverse(Hessian) * 2.0

        Parameters
        ----------
        fvars : numpy.ndarray
            Array of the optimal internal, freely variable parameters.

        Returns
        -------
        cov_x : numpy.ndarray or None
            Covariance matrix if successful, otherwise None.

        """
        warnings.filterwarnings(action="ignore", module="scipy",
                                message="^internal gelsd")

        nfev = deepcopy(self.result.nfev)
        best_vals = self.result.params.valuesdict()

        try:
            Hfun = ndt.Hessian(self.penalty, step=1.e-4)
            hessian_ndt = Hfun(fvars)
            cov_x = inv(hessian_ndt) * 2.0
        except (LinAlgError, ValueError):
            return None
        finally:
            self.result.nfev = nfev

        # restore original values
        for name in self.result.var_names:
            self.result.params[name].value = best_vals[name]
        return cov_x

    def _int2ext_cov_x(self, cov_int, fvars):
        """Transform covariance matrix to external parameter space.

        It makes use of the gradient scaling according to the MINUIT
        recipe:

            cov_ext = np.dot(grad.T, grad) * cov_int

        Parameters
        ----------
        cov_int : numpy.ndarray
            Covariance matrix in the internal parameter space.
        fvars : numpy.ndarray
            Array of the optimal internal, freely variable, parameter
            values.

        Returns
        -------
        cov_ext : numpy.ndarray
            Covariance matrix, transformed to external parameter space.

        """
        g = [self.result.params[name].scale_gradient(fvars[i]) for i, name in
             enumerate(self.result.var_names)]
        grad2d = np.atleast_2d(g)
        grad = np.dot(grad2d.T, grad2d)

        cov_ext = cov_int * grad
        return cov_ext

    def _calculate_uncertainties_correlations(self):
        """Calculate parameter uncertainties and correlations."""
        self.result.errorbars = True

        if self.scale_covar:
            self.result.covar *= self.result.redchi

        vbest = np.atleast_1d([self.result.params[name].value for name in
                               self.result.var_names])

        has_expr = False
        for par in self.result.params.values():
            par.stderr, par.correl = 0, None
            has_expr = has_expr or par.expr is not None

        for ivar, name in enumerate(self.result.var_names):
            par = self.result.params[name]
            par.stderr = np.sqrt(self.result.covar[ivar, ivar])
            par.correl = {}
            try:
                self.result.errorbars = self.result.errorbars and (par.stderr > 0.0)
                for jvar, varn2 in enumerate(self.result.var_names):
                    if jvar != ivar:
                        par.correl[varn2] = (self.result.covar[ivar, jvar] /
                                             (par.stderr * np.sqrt(self.result.covar[jvar, jvar])))
            except ZeroDivisionError:
                self.result.errorbars = False

        if has_expr:
            try:
                uvars = uncertainties.correlated_values(vbest, self.result.covar)
            except (LinAlgError, ValueError):
                uvars = None

            # for uncertainties on constrained parameters, use the calculated
            # "correlated_values", evaluate the uncertainties on the constrained
            # parameters and reset the Parameters to best-fit value
            if uvars is not None:
                for par in self.result.params.values():
                    eval_stderr(par, uvars, self.result.var_names, self.result.params)
                # restore nominal values
                for v, nam in zip(uvars, self.result.var_names):
                    self.result.params[nam].value = v.nominal_value

    def scalar_minimize(self, method='Nelder-Mead', params=None, max_nfev=None,
                        **kws):
        """Scalar minimization using :scipydoc:`optimize.minimize`.

        Perform fit with any of the scalar minimization algorithms
        supported by :scipydoc:`optimize.minimize`. Default argument
        values are:

        +-------------------------+---------------+-----------------------+
        | :meth:`scalar_minimize` | Default Value | Description           |
        | arg                     |               |                       |
        +=========================+===============+=======================+
        |  `method`               | 'Nelder-Mead' | fitting method        |
        +-------------------------+---------------+-----------------------+
        |  `tol`                  | 1.e-7         | fitting and parameter |
        |                         |               | tolerance             |
        +-------------------------+---------------+-----------------------+
        |  `hess`                 | None          | Hessian of objective  |
        |                         |               | function              |
        +-------------------------+---------------+-----------------------+


        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use. One of:

            - `'Nelder-Mead'` (default)
            - `'L-BFGS-B'`
            - `'Powell'`
            - `'CG'`
            - `'Newton-CG'`
            - `'COBYLA'`
            - `'BFGS'`
            - `'TNC'`
            - `'trust-ncg'`
            - `'trust-exact'`
            - `'trust-krylov'`
            - `'trust-constr'`
            - `'dogleg'`
            - `'SLSQP'`
            - `'differential_evolution'`

        params : Parameters, optional
            Parameters to use as starting point.
        max_nfev : int or None, optional
            Maximum number of function evaluations. Defaults to
            ``2000*(nvars+1)``, where ``nvars`` is the number of variable
            parameters.
        **kws : dict, optional
            Minimizer options pass to :scipydoc:`optimize.minimize`.

        Returns
        -------
        MinimizerResult
            Object containing the optimized parameters and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.


        Notes
        -----
        If the objective function returns a NumPy array instead of the
        expected scalar, the sum-of-squares of the array will be used.

        Note that bounds and constraints can be set on Parameters for any
        of these methods, so are not supported separately for those
        designed to use bounds. However, if you use the
        ``differential_evolution`` method you must specify finite
        ``(min, max)`` for each varying Parameter.

        """
        result = self.prepare_fit(params=params)
        result.method = method
        variables = result.init_vals
        params = result.params

        self.set_max_nfev(max_nfev, 2000*(result.nvarys+1))
        fmin_kws = dict(method=method, options={'maxiter': 2*self.max_nfev})
        fmin_kws.update(self.kws)

        if 'maxiter' in kws:
            warnings.warn(maxeval_warning.format('maxiter', thisfuncname()),
                          RuntimeWarning)
            kws.pop('maxiter')
        fmin_kws.update(kws)

        # hess supported only in some methods
        if 'hess' in fmin_kws and method not in ('Newton-CG', 'dogleg',
                                                 'trust-constr', 'trust-ncg',
                                                 'trust-krylov', 'trust-exact'):
            fmin_kws.pop('hess')

        # Accept Jacobians given as Dfun argument
        if 'jac' not in fmin_kws and fmin_kws.get('Dfun', None) is not None:
            fmin_kws['jac'] = fmin_kws.pop('Dfun')

        # Wrap Jacobian function to deal with bounds
        if 'jac' in fmin_kws:
            self.jacfcn = fmin_kws.pop('jac')
            fmin_kws['jac'] = self.__jacobian

        # Ignore jac argument for methods that do not support it
        if 'jac' in fmin_kws and method not in ('CG', 'BFGS', 'Newton-CG',
                                                'L-BFGS-B', 'TNC', 'SLSQP',
                                                'dogleg', 'trust-ncg',
                                                'trust-krylov', 'trust-exact'):
            self.jacfcn = None
            fmin_kws.pop('jac')

        # workers / updating keywords only supported in differential_evolution
        for kwd in ('workers', 'updating'):
            if kwd in fmin_kws and method != 'differential_evolution':
                fmin_kws.pop(kwd)

        if method == 'differential_evolution':
            for par in params.values():
                if (par.vary and
                        not (np.isfinite(par.min) and np.isfinite(par.max))):
                    raise ValueError('differential_evolution requires finite '
                                     'bound for all varying parameters')

            _bounds = [(-np.pi / 2., np.pi / 2.)] * len(variables)
            kwargs = dict(args=(), strategy='best1bin', maxiter=self.max_nfev,
                          popsize=15, tol=0.01, mutation=(0.5, 1),
                          recombination=0.7, seed=None, callback=None,
                          disp=False, polish=True, init='latinhypercube',
                          atol=0, updating='immediate', workers=1)

            for k, v in fmin_kws.items():
                if k in kwargs:
                    kwargs[k] = v

            fmin_kws = kwargs
            result.call_kws = fmin_kws
            try:
                ret = differential_evolution(self.penalty, _bounds, **fmin_kws)
            except AbortFitException:
                pass

        else:
            result.call_kws = fmin_kws
            try:
                ret = scipy_minimize(self.penalty, variables, **fmin_kws)
            except AbortFitException:
                pass

        if not result.aborted:
            if isinstance(ret, dict):
                for attr, value in ret.items():
                    setattr(result, attr, value)
            else:
                for attr in dir(ret):
                    if not attr.startswith('_'):
                        setattr(result, attr, getattr(ret, attr))

            result.x = np.atleast_1d(result.x)
            result.residual = self.__residual(result.x)
            result.nfev -= 1
        else:
            result.x = result.last_internal_values
            self.result.nfev -= 2
            self._abort = False
            result.residual = self.__residual(result.x)
            result.nfev += 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
                len(result.residual) > len(result.var_names)):
            _covar_ndt = self._calculate_covariance_matrix(result.x)
            if _covar_ndt is not None:
                result.covar = self._int2ext_cov_x(_covar_ndt, result.x)
                self._calculate_uncertainties_correlations()

        return result

    def _lnprob(self, theta, userfcn, params, var_names, bounds, userargs=(),
                userkws=None, float_behavior='posterior', is_weighted=True,
                nan_policy='raise'):
        """Calculate the log-posterior probability.

        See the `Minimizer.emcee` method for more details.

        Parameters
        ----------
        theta : sequence
            Float parameter values (only those being varied).
        userfcn : callable
            User objective function.
        params : Parameters
            The entire set of Parameters.
        var_names : list
            The names of the parameters that are varying.
        bounds : numpy.ndarray
            Lower and upper bounds of parameters, with shape
            ``(nvarys, 2)``.
        userargs : tuple, optional
            Extra positional arguments required for user objective function.
        userkws : dict, optional
            Extra keyword arguments required for user objective function.
        float_behavior : {'posterior', 'chi2'}, optional
            Specifies meaning of objective when it returns a float. Use
            `'posterior'` if objective function returns a log-posterior
            probability (default) or `'chi2'` if it returns a chi2 value.
        is_weighted : bool, optional
            If `userfcn` returns a vector of residuals then `is_weighted`
            (default is True) specifies if the residuals have been weighted
            by data uncertainties.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            Specifies action if `userfcn` returns NaN values. Use `'raise'`
            (default) to raise a `ValueError`, `'propagate'` to use values
            as-is, or `'omit'` to filter out the non-finite values.

        Returns
        -------
        lnprob : float
            Log posterior probability.

        """
        # the comparison has to be done on theta and bounds. DO NOT inject theta
        # values into Parameters, then compare Parameters values to the bounds.
        # Parameters values are clipped to stay within bounds.
        if np.any(theta > bounds[:, 1]) or np.any(theta < bounds[:, 0]):
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
        self.result.nfev += 1
        if callable(self.iter_cb):
            abort = self.iter_cb(params, self.result.nfev, out,
                                 *userargs, **userkwargs)
            self._abort = self._abort or abort
        if self._abort:
            self.result.residual = out
            self._lastpos = theta
            raise AbortFitException("fit aborted by user.")
        else:
            out = _nan_policy(np.asarray(out).ravel(),
                              nan_policy=self.nan_policy)
        lnprob = np.asarray(out).ravel()
        if len(lnprob) == 0:
            lnprob = np.array([-1.e100])
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
                raise ValueError("float_behaviour must be either 'posterior' "
                                 "or 'chi2' " + float_behavior)
        return lnprob

    def emcee(self, params=None, steps=1000, nwalkers=100, burn=0, thin=1,
              ntemps=1, pos=None, reuse_sampler=False, workers=1,
              float_behavior='posterior', is_weighted=True, seed=None,
              progress=True, run_mcmc_kwargs={}):
        r"""Bayesian sampling of the posterior distribution.

        The method uses the ``emcee`` Markov Chain Monte Carlo package and
        assumes that the prior is Uniform. You need to have ``emcee``
        version 3 or newer installed to use this method.

        Parameters
        ----------
        params : Parameters, optional
            Parameters to use as starting point. If this is not specified
            then the Parameters used to initialize the Minimizer object
            are used.
        steps : int, optional
            How many samples you would like to draw from the posterior
            distribution for each of the walkers?
        nwalkers : int, optional
            Should be set so :math:`nwalkers >> nvarys`, where ``nvarys``
            are the number of parameters being varied during the fit.
            'Walkers are the members of the ensemble. They are almost like
            separate Metropolis-Hastings chains but, of course, the proposal
            distribution for a given walker depends on the positions of all
            the other walkers in the ensemble.' - from the `emcee` webpage.
        burn : int, optional
            Discard this many samples from the start of the sampling regime.
        thin : int, optional
            Only accept 1 in every `thin` samples.
        ntemps : int, deprecated
            ntemps has no effect.
        pos : numpy.ndarray, optional
            Specify the initial positions for the sampler, an ndarray of
            shape ``(nwalkers, nvarys)``. You can also initialise using a
            previous chain of the same `nwalkers` and ``nvarys``. Note that
            ``nvarys`` may be one larger than you expect it to be if your
            ``userfcn`` returns an array and ``is_weighted=False``.
        reuse_sampler : bool, optional
            Set to True if you have already run `emcee` with the
            `Minimizer` instance and want to continue to draw from its
            ``sampler`` (and so retain the chain history). If False, a
            new sampler is created. The keywords `nwalkers`, `pos`, and
            `params` will be ignored when this is set, as they will be set
            by the existing sampler.
            **Important**: the Parameters used to create the sampler must
            not change in-between calls to `emcee`. Alteration of Parameters
            would include changed ``min``, ``max``, ``vary`` and ``expr``
            attributes. This may happen, for example, if you use an altered
            Parameters object and call the `minimize` method in-between
            calls to `emcee`.
        workers : Pool-like or int, optional
            For parallelization of sampling. It can be any Pool-like object
            with a map method that follows the same calling sequence as the
            built-in `map` function. If int is given as the argument, then
            a multiprocessing-based pool is spawned internally with the
            corresponding number of parallel processes. 'mpi4py'-based
            parallelization and 'joblib'-based parallelization pools can
            also be used here. **Note**: because of multiprocessing
            overhead it may only be worth parallelising if the objective
            function is expensive to calculate, or if there are a large
            number of objective evaluations per step
            (``nwalkers * nvarys``).
        float_behavior : str, optional
            Meaning of float (scalar) output of objective function. Use
            `'posterior'` if it returns a log-posterior probability or
            `'chi2'` if it returns :math:`\chi^2`. See Notes for further
            details.
        is_weighted : bool, optional
            Has your objective function been weighted by measurement
            uncertainties? If ``is_weighted=True`` then your objective
            function is assumed to return residuals that have been divided
            by the true measurement uncertainty ``(data - model) / sigma``.
            If ``is_weighted=False`` then the objective function is
            assumed to return unweighted residuals, ``data - model``. In
            this case `emcee` will employ a positive measurement
            uncertainty during the sampling. This measurement uncertainty
            will be present in the output params and output chain with the
            name ``__lnsigma``. A side effect of this is that you cannot
            use this parameter name yourself.
            **Important**: this parameter only has any effect if your
            objective function returns an array. If your objective function
            returns a float, then this parameter is ignored. See Notes for
            more details.
        seed : int or numpy.random.RandomState, optional
            If `seed` is an ``int``, a new `numpy.random.RandomState`
            instance is used, seeded with `seed`.
            If `seed` is already a `numpy.random.RandomState` instance,
            then that `numpy.random.RandomState` instance is used. Specify
            `seed` for repeatable minimizations.
        progress : bool, optional
            Print a progress bar to the console while running.
        run_mcmc_kwargs : dict, optional
            Additional (optional) keyword arguments that are passed to
            ``emcee.EnsembleSampler.run_mcmc``.

        Returns
        -------
        MinimizerResult
            MinimizerResult object containing updated params, statistics,
            etc. The updated params represent the median of the samples,
            while the uncertainties are half the difference of the 15.87
            and 84.13 percentiles. The `MinimizerResult` contains a few
            additional attributes: `chain` contain the samples and has
            shape ``((steps - burn) // thin, nwalkers, nvarys)``.
            `flatchain` is a `pandas.DataFrame` of the flattened chain,
            that can be accessed with `result.flatchain[parname]`.
            `lnprob` contains the log probability for each sample in
            `chain`. The sample with the highest probability corresponds
            to the maximum likelihood estimate. `acor` is an array
            containing the auto-correlation time for each parameter if the
            auto-correlation time can be computed from the chain. Finally,
            `acceptance_fraction` (an array of the fraction of steps
            accepted for each walker).

        Notes
        -----
        This method samples the posterior distribution of the parameters
        using Markov Chain Monte Carlo. It calculates the log-posterior
        probability of the model parameters, `F`, given the data, `D`,
        :math:`\ln p(F_{true} | D)`. This 'posterior probability' is
        given by:

        .. math::

            \ln p(F_{true} | D) \propto \ln p(D | F_{true}) + \ln p(F_{true})

        where :math:`\ln p(D | F_{true})` is the 'log-likelihood' and
        :math:`\ln p(F_{true})` is the 'log-prior'. The default log-prior
        encodes prior information known about the model that the log-prior
        probability is ``-numpy.inf`` (impossible) if any of the parameters
        is outside its limits, and is zero if all the parameters are inside
        their bounds (uniform prior). The log-likelihood function is [1]_:

        .. math::

            \ln p(D|F_{true}) = -\frac{1}{2}\sum_n \left[\frac{(g_n(F_{true}) - D_n)^2}{s_n^2}+\ln (2\pi s_n^2)\right]

        The first term represents the residual (:math:`g` being the
        generative model, :math:`D_n` the data and :math:`s_n` the
        measurement uncertainty). This gives :math:`\chi^2` when summed
        over all data points. The objective function may also return the
        log-posterior probability, :math:`\ln p(F_{true} | D)`. Since the
        default log-prior term is zero, the objective function can also
        just return the log-likelihood, unless you wish to create a
        non-uniform prior.

        If the objective function returns a float value, this is assumed
        by default to be the log-posterior probability, (`float_behavior`
        default is 'posterior'). If your objective function returns
        :math:`\chi^2`, then you should use ``float_behavior='chi2'``
        instead.

        By default objective functions may return an ndarray of (possibly
        weighted) residuals. In this case, use `is_weighted` to select
        whether these are correctly weighted by measurement uncertainty.
        Note that this ignores the second term above, so that to calculate
        a correct log-posterior probability value your objective function
        should return a float value. With ``is_weighted=False`` the data
        uncertainty, `s_n`, will be treated as a nuisance parameter to be
        marginalized out. This uses strictly positive uncertainty
        (homoscedasticity) for each data point,
        :math:`s_n = \exp(\rm{\_\_lnsigma})`. ``__lnsigma`` will be
        present in `MinimizerResult.params`, as well as `Minimizer.chain`
        and ``nvarys`` will be increased by one.

        References
        ----------
        .. [1] https://emcee.readthedocs.io

        """
        if not HAS_EMCEE:
            raise NotImplementedError('emcee version 3 is required.')

        if ntemps > 1:
            msg = ("'ntemps' has no effect anymore, since the PTSampler was "
                   "removed from emcee version 3.")
            raise DeprecationWarning(msg)

        tparams = params
        # if you're reusing the sampler then nwalkers have to be
        # determined from the previous sampling
        if reuse_sampler:
            if not hasattr(self, 'sampler') or not hasattr(self, '_lastpos'):
                raise ValueError("You wanted to use an existing sampler, but "
                                 "it hasn't been created yet")
            if len(self._lastpos.shape) == 2:
                nwalkers = self._lastpos.shape[0]
            elif len(self._lastpos.shape) == 3:
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
                params.add('__lnsigma', value=0.01, min=-np.inf, max=np.inf,
                           vary=True)
                # have to re-prepare the fit
                result = self.prepare_fit(params)
                params = result.params

        result.method = 'emcee'

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
        if isinstance(workers, int) and workers > 1 and HAS_DILL:
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
                raise ValueError("You cannot reuse the sampler if the number "
                                 "of varying parameters has changed")

        else:
            p0 = 1 + rng.randn(nwalkers, self.nvarys) * 1.e-4
            p0 *= var_arr
            sampler_kwargs.setdefault('pool', auto_pool)
            self.sampler = emcee.EnsembleSampler(nwalkers, self.nvarys,
                                                 self._lnprob, **sampler_kwargs)

        # user supplies an initialisation position for the chain
        # If you try to run the sampler with p0 of a wrong size then you'll get
        # a ValueError. Note, you can't initialise with a position if you are
        # reusing the sampler.
        if pos is not None and not reuse_sampler:
            tpos = np.asfarray(pos)
            if p0.shape == tpos.shape:
                pass
            # trying to initialise with a previous chain
            elif tpos.shape[-1] == self.nvarys:
                tpos = tpos[-1]
            else:
                raise ValueError('pos should have shape (nwalkers, nvarys)')
            p0 = tpos

        # if you specified a seed then you also need to seed the sampler
        if seed is not None:
            self.sampler.random_state = rng.get_state()

        if not isinstance(run_mcmc_kwargs, dict):
            raise ValueError('run_mcmc_kwargs should be a dict of keyword arguments')

        # now do a production run, sampling all the time
        try:
            output = self.sampler.run_mcmc(p0, steps, progress=progress, **run_mcmc_kwargs)
            self._lastpos = output.coords
        except AbortFitException:
            result.aborted = True
            result.message = "Fit aborted by user callback. Could not estimate error-bars."
            result.success = False
            result.nfev = self.result.nfev

        # discard the burn samples and thin
        chain = self.sampler.get_chain(thin=thin, discard=burn)[..., :, :]
        lnprobability = self.sampler.get_log_prob(thin=thin, discard=burn)[..., :]
        flatchain = chain.reshape((-1, self.nvarys))
        if not result.aborted:
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
        result.nfev = nwalkers*steps

        try:
            result.acor = self.sampler.get_autocorr_time()
        except AutocorrError as e:
            print(str(e))
        result.acceptance_fraction = self.sampler.acceptance_fraction

        # Calculate the residual with the "best fit" parameters
        out = self.userfcn(params, *self.userargs, **self.userkws)
        result.residual = _nan_policy(out, nan_policy=self.nan_policy,
                                      handle_inf=False)

        # If uncertainty was automatically estimated, weight the residual properly
        if (not is_weighted) and (result.residual.size > 1):
            if '__lnsigma' in params:
                result.residual = result.residual/np.exp(params['__lnsigma'].value)

        # Calculate statistics for the two standard cases:
        if isinstance(result.residual, np.ndarray) or (float_behavior == 'chi2'):
            result._calculate_statistics()

        # Handle special case unique to emcee:
        # This should eventually be moved into result._calculate_statistics.
        elif float_behavior == 'posterior':
            result.ndata = 1
            result.nfree = 1

            # assuming prior prob = 1, this is true
            _neg2_log_likel = -2*result.residual

            # assumes that residual is properly weighted, avoid overflowing np.exp()
            result.chisqr = np.exp(min(650, _neg2_log_likel))

            result.redchi = result.chisqr / result.nfree
            result.aic = _neg2_log_likel + 2 * result.nvarys
            result.bic = _neg2_log_likel + np.log(result.ndata) * result.nvarys

        if auto_pool is not None:
            auto_pool.terminate()

        return result

    def least_squares(self, params=None, max_nfev=None, **kws):
        """Least-squares minimization using :scipydoc:`optimize.least_squares`.

        This method wraps :scipydoc:`optimize.least_squares`, which has
        built-in support for bounds and robust loss functions. By default
        it uses the Trust Region Reflective algorithm with a linear loss
        function (i.e., the standard least-squares problem).

        Parameters
        ----------
        params : Parameters, optional
            Parameters to use as starting point.
        max_nfev : int or None, optional
            Maximum number of function evaluations. Defaults to
            ``2000*(nvars+1)``, where ``nvars`` is the number of variable
            parameters.
        **kws : dict, optional
            Minimizer options to pass to :scipydoc:`optimize.least_squares`.

        Returns
        -------
        MinimizerResult
            Object containing the optimized parameters and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.

        """
        result = self.prepare_fit(params)
        result.method = 'least_squares'

        replace_none = lambda x, sign: sign*np.inf if x is None else x
        self.set_max_nfev(max_nfev, 2000*(result.nvarys+1))

        start_vals, lower_bounds, upper_bounds = [], [], []
        for vname in result.var_names:
            par = self.params[vname]
            start_vals.append(par.value)
            lower_bounds.append(replace_none(par.min, -1))
            upper_bounds.append(replace_none(par.max, 1))

        result.call_kws = kws
        try:
            ret = least_squares(self.__residual, start_vals,
                                bounds=(lower_bounds, upper_bounds),
                                kwargs=dict(apply_bounds_transformation=False),
                                max_nfev=2*self.max_nfev, **kws)
            result.residual = ret.fun
        except AbortFitException:
            pass

        # note: upstream least_squares is actually returning
        # "last evaluation", not "best result", do we do that
        # here for consistency
        if not result.aborted:
            result.nfev -= 1
            result.residual = self.__residual(ret.x, False)
        result._calculate_statistics()

        if not result.aborted:
            for attr in ret:
                setattr(result, attr, ret[attr])

            result.x = np.atleast_1d(result.x)

            # calculate the cov_x and estimate uncertainties/correlations
            try:
                if issparse(ret.jac):
                    hess = (ret.jac.T * ret.jac).toarray()
                elif isinstance(ret.jac, LinearOperator):
                    identity = np.eye(ret.jac.shape[1], dtype=ret.jac.dtype)
                    hess = (ret.jac.T * ret.jac) * identity
                else:
                    hess = np.matmul(ret.jac.T, ret.jac)
                result.covar = np.linalg.inv(hess)
                self._calculate_uncertainties_correlations()
            except LinAlgError:
                pass

        return result

    def leastsq(self, params=None, max_nfev=None, **kws):
        """Use Levenberg-Marquardt minimization to perform a fit.

        It assumes that the input Parameters have been initialized, and a
        function to minimize has been properly set up. When possible, this
        calculates the estimated uncertainties and variable correlations
        from the covariance matrix.

        This method calls :scipydoc:`optimize.leastsq` and, by default,
        numerical derivatives are used, and the following arguments are
        set:

        +------------------+----------------+------------------------+
        | :meth:`leastsq`  |  Default Value | Description            |
        | arg              |                |                        |
        +==================+================+========================+
        |  `xtol`          |  1.e-7         | Relative error in the  |
        |                  |                | approximate solution   |
        +------------------+----------------+------------------------+
        |  `ftol`          |  1.e-7         | Relative error in the  |
        |                  |                | desired sum-of-squares |
        +------------------+----------------+------------------------+
        |  `Dfun`          | None           | Function to call for   |
        |                  |                | Jacobian calculation   |
        +------------------+----------------+------------------------+

        Parameters
        ----------
        params : Parameters, optional
            Parameters to use as starting point.
        max_nfev : int or None, optional
            Maximum number of function evaluations. Defaults to
            ``2000*(nvars+1)``, where ``nvars`` is the number of variable
            parameters.
        **kws : dict, optional
            Minimizer options to pass to :scipydoc:`optimize.leastsq`.

        Returns
        -------
        MinimizerResult
            Object containing the optimized parameters and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.

        """
        result = self.prepare_fit(params=params)
        result.method = 'leastsq'
        result.nfev -= 2  # correct for "pre-fit" initialization/checks
        variables = result.init_vals

        # note we set the max number of function evaluations here, and send twice that
        # value to the solver so it essentially never stops on its own
        self.set_max_nfev(max_nfev, 2000*(result.nvarys+1))

        lskws = dict(full_output=1, xtol=1.e-7, ftol=1.e-7, col_deriv=False,
                     gtol=1.e-7, maxfev=2*self.max_nfev, Dfun=None)
        if 'maxfev' in kws:
            warnings.warn(maxeval_warning.format('maxfev', thisfuncname()),
                          RuntimeWarning)
            kws.pop('maxfev')

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
        result.call_kws = lskws
        try:
            lsout = scipy_leastsq(self.__residual, variables, **lskws)
        except AbortFitException:
            pass

        if not result.aborted:
            _best, _cov, _infodict, errmsg, ier = lsout
        else:
            _best = result.last_internal_values
            _cov = None
            ier = -1
            errmsg = 'Fit aborted.'

        result.nfev -= 1
        if result.nfev >= self.max_nfev:
            result.nfev = self.max_nfev - 1
        self.result.nfev = result.nfev
        try:
            result.residual = self.__residual(_best)
            result._calculate_statistics()
        except AbortFitException:
            pass

        result.ier = ier
        result.lmdif_message = errmsg
        result.success = ier in [1, 2, 3, 4]
        if ier in {1, 2, 3}:
            result.message = 'Fit succeeded.'
        elif ier == 0:
            result.message = ('Invalid Input Parameters. I.e. more variables '
                              'than data points given, tolerance < 0.0, or '
                              'no data provided.')
        elif ier == 4:
            result.message = 'One or more variable did not affect the fit.'
        elif ier == 5:
            result.message = self._err_max_evals.format(lskws['maxfev'])
        else:
            result.message = 'Tolerance seems to be too small.'

        # self.errorbars = error bars were successfully estimated
        result.errorbars = (_cov is not None)
        if result.errorbars:
            # transform the covariance matrix to "external" parameter space
            result.covar = self._int2ext_cov_x(_cov, _best)
            # calculate parameter uncertainties and correlations
            self._calculate_uncertainties_correlations()
        else:
            result.message = f'{result.message} Could not estimate error-bars.'

        np.seterr(**orig_warn_settings)

        return result

    def basinhopping(self, params=None, max_nfev=None, **kws):
        """Use the `basinhopping` algorithm to find the global minimum.

        This method calls :scipydoc:`optimize.basinhopping` using the
        default arguments. The default minimizer is ``BFGS``, but since
        lmfit supports parameter bounds for all minimizers, the user can
        choose any of the solvers present in :scipydoc:`optimize.minimize`.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model. If None, then the
            Parameters used to initialize the Minimizer object are used.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). Defaults
            to ``200000*(nvarys+1)``.
        **kws : dict, optional
            Minimizer options to pass to :scipydoc:`optimize.basinhopping`.

        Returns
        -------
        MinimizerResult
            Object containing the optimization results from the
            basinhopping algorithm.


        .. versionadded:: 0.9.10

        """
        result = self.prepare_fit(params=params)
        result.method = 'basinhopping'
        self.set_max_nfev(max_nfev, 200000*(result.nvarys+1))
        basinhopping_kws = dict(niter=100, T=1.0, stepsize=0.5,
                                minimizer_kwargs={}, take_step=None,
                                accept_test=None, callback=None, interval=50,
                                disp=False, niter_success=None, seed=None)

        basinhopping_kws.update(self.kws)
        basinhopping_kws.update(kws)

        x0 = result.init_vals
        result.call_kws = basinhopping_kws
        try:
            ret = scipy_basinhopping(self.penalty, x0, **basinhopping_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            result.message = ret.message
            result.residual = self.__residual(ret.x)
            result.nfev -= 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
                len(result.residual) > len(result.var_names)):
            _covar_ndt = self._calculate_covariance_matrix(ret.x)
            if _covar_ndt is not None:
                result.covar = self._int2ext_cov_x(_covar_ndt, ret.x)
                self._calculate_uncertainties_correlations()

        return result

    def brute(self, params=None, Ns=20, keep=50, workers=1, max_nfev=None):
        """Use the `brute` method to find the global minimum of a function.

        The following parameters are passed to :scipydoc:`optimize.brute`
        and cannot be changed:

        +-------------------+-------+------------------------------------+
        | :meth:`brute` arg | Value | Description                        |
        +===================+=======+====================================+
        |  `full_output`    | 1     | Return the evaluation grid and the |
        |                   |       | objective function's values on it. |
        +-------------------+-------+------------------------------------+
        |  `finish`         | None  | No "polishing" function is to be   |
        |                   |       | used after the grid search.        |
        +-------------------+-------+------------------------------------+
        |  `disp`           | False | Do not print convergence messages  |
        |                   |       | (when finish is not None).         |
        +-------------------+-------+------------------------------------+

        It assumes that the input Parameters have been initialized, and a
        function to minimize has been properly set up.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model. If None, then the
            Parameters used to initialize the Minimizer object are used.
        Ns : int, optional
            Number of grid points along the axes, if not otherwise
            specified (see Notes).
        keep : int, optional
            Number of best candidates from the brute force method that are
            stored in the :attr:`candidates` attribute. If `'all'`, then
            all grid points from :scipydoc:`optimize.brute` are stored as
            candidates.
        workers : int or map-like callable, optional
            For parallel evaluation of the grid (see :scipydoc:`optimize.brute`
            for more details).
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). Defaults
            to ``200000*(nvarys+1)``.

        Returns
        -------
        MinimizerResult
            Object containing the parameters from the brute force method.
            The return values (``x0``, ``fval``, ``grid``, ``Jout``) from
            :scipydoc:`optimize.brute` are stored as ``brute_<parname>``
            attributes. The `MinimizerResult` also contains the
            :attr:``candidates`` attribute and :meth:`show_candidates`
            method. The :attr:`candidates` attribute contains the
            parameters and chisqr from the brute force method as a
            namedtuple, ``('Candidate', ['params', 'score'])`` sorted on
            the (lowest) chisqr value. To access the values for a
            particular candidate one can use ``result.candidate[#].params``
            or ``result.candidate[#].score``, where a lower # represents a
            better candidate. The :meth:`show_candidates` method uses the
            :meth:`pretty_print` method to show a specific candidate-# or
            all candidates when no number is specified.


        .. versionadded:: 0.9.6


        Notes
        -----
        The :meth:`brute` method evaluates the function at each point of a
        multidimensional grid of points. The grid points are generated from
        the parameter ranges using `Ns` and (optional) `brute_step`.
        The implementation in :scipydoc:`optimize.brute` requires finite
        bounds and the ``range`` is specified as a two-tuple ``(min, max)``
        or slice-object ``(min, max, brute_step)``. A slice-object is used
        directly, whereas a two-tuple is converted to a slice object that
        interpolates `Ns` points from ``min`` to ``max``, inclusive.

        In addition, the :meth:`brute` method in lmfit, handles three other
        scenarios given below with their respective slice-object:

            - lower bound (:attr:`min`) and :attr:`brute_step` are specified:
                ``range = (min, min + Ns * brute_step, brute_step)``.
            - upper bound (:attr:`max`) and :attr:`brute_step` are specified:
                ``range = (max - Ns * brute_step, max, brute_step)``.
            - numerical value (:attr:`value`) and :attr:`brute_step` are specified:
                ``range = (value - (Ns//2) * brute_step`, value +
                (Ns//2) * brute_step, brute_step)``.

        """
        result = self.prepare_fit(params=params)
        result.method = 'brute'
        self.set_max_nfev(max_nfev, 200000*(result.nvarys+1))

        brute_kws = dict(full_output=1, finish=None, disp=False, Ns=Ns,
                         workers=workers)

        varying = np.asarray([par.vary for par in self.params.values()])
        replace_none = lambda x, sign: sign*np.inf if x is None else x
        lower_bounds = np.asarray([replace_none(i.min, -1) for i in
                                   self.params.values()])[varying]
        upper_bounds = np.asarray([replace_none(i.max, 1) for i in
                                   self.params.values()])[varying]
        value = np.asarray([i.value for i in self.params.values()])[varying]
        stepsize = np.asarray([i.brute_step for i in self.params.values()])[varying]

        ranges = []
        for i, step in enumerate(stepsize):
            if np.all(np.isfinite([lower_bounds[i], upper_bounds[i]])):
                # lower AND upper bounds are specified (brute_step optional)
                par_range = ((lower_bounds[i], upper_bounds[i], step)
                             if step else (lower_bounds[i], upper_bounds[i]))
            elif np.isfinite(lower_bounds[i]) and step:
                # lower bound AND brute_step are specified
                par_range = (lower_bounds[i], lower_bounds[i] + Ns*step, step)
            elif np.isfinite(upper_bounds[i]) and step:
                # upper bound AND brute_step are specified
                par_range = (upper_bounds[i] - Ns*step, upper_bounds[i], step)
            elif np.isfinite(value[i]) and step:
                # no bounds, but an initial value is specified
                par_range = (value[i] - (Ns//2)*step, value[i] + (Ns//2)*step,
                             step)
            else:
                raise ValueError('Not enough information provided for the brute '
                                 'force method. Please specify bounds or at '
                                 'least an initial value and brute_step for '
                                 'parameter "{}".'.format(result.var_names[i]))
            ranges.append(par_range)
        result.call_kws = brute_kws
        try:
            ret = scipy_brute(self.penalty, tuple(ranges), **brute_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            result.brute_x0 = ret[0]
            result.brute_fval = ret[1]
            result.brute_grid = ret[2]
            result.brute_Jout = ret[3]

            # sort the results of brute and populate .candidates attribute
            grid_score = ret[3].ravel()  # chisqr
            grid_points = [par.ravel() for par in ret[2]]

            if len(result.var_names) == 1:
                grid_result = np.array([res for res in zip(zip(grid_points), grid_score)],
                                       dtype=[('par', 'O'), ('score', 'float')])
            else:
                grid_result = np.array([res for res in zip(zip(*grid_points), grid_score)],
                                       dtype=[('par', 'O'), ('score', 'float')])
            grid_result_sorted = grid_result[grid_result.argsort(order='score')]

            result.candidates = []

            if keep == 'all':
                keep_candidates = len(grid_result_sorted)
            else:
                keep_candidates = min(len(grid_result_sorted), keep)

            for data in grid_result_sorted[:keep_candidates]:
                pars = deepcopy(self.params)
                for i, par in enumerate(result.var_names):
                    pars[par].value = data[0][i]
                result.candidates.append(Candidate(params=pars, score=data[1]))

            result.params = result.candidates[0].params
            result.residual = self.__residual(result.brute_x0,
                                              apply_bounds_transformation=False)
            result.nfev = len(result.brute_Jout.ravel())

        result._calculate_statistics()

        return result

    def ampgo(self, params=None, max_nfev=None, **kws):
        """Find the global minimum of a multivariate function using AMPGO.

        AMPGO stands for 'Adaptive Memory Programming for Global
        Optimization' and is an efficient algorithm to find the global
        minimum.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model. If None, then the
            Parameters used to initialize the Minimizer object are used.
        max_nfev : int, optional
            Maximum number of total function evaluations. If None
            (default), the optimization will stop after `totaliter` number
            of iterations (see below)..
        **kws : dict, optional
            Minimizer options to pass to the ampgo algorithm, the options
            are listed below::

                local: str, optional
                    Name of the local minimization method. Valid options
                    are:
                    - `'L-BFGS-B'` (default)
                    - `'Nelder-Mead'`
                    - `'Powell'`
                    - `'TNC'`
                    - `'SLSQP'`
                local_opts: dict, optional
                    Options to pass to the local minimizer (default is
                    None).
                maxfunevals: int, optional
                    Maximum number of function evaluations. If None
                    (default), the optimization will stop after
                    `totaliter` number of iterations (deprecated: use
                    `max_nfev` instead).
                totaliter: int, optional
                    Maximum number of global iterations (default is 20).
                maxiter: int, optional
                    Maximum number of `Tabu Tunneling` iterations during
                    each global iteration (default is 5).
                glbtol: float, optional
                    Tolerance whether or not to accept a solution after a
                    tunneling phase (default is 1e-5).
                eps1: float, optional
                    Constant used to define an aspiration value for the
                    objective function during the Tunneling phase (default
                    is 0.02).
                eps2: float, optional
                    Perturbation factor used to move away from the latest
                    local minimum at the start of a Tunneling phase
                    (default is 0.1).
                tabulistsize: int, optional
                    Size of the (circular) tabu search list (default is 5).
                tabustrategy: {'farthest', 'oldest'}, optional
                    Strategy to use when the size of the tabu list exceeds
                    `tabulistsize`. It can be `'oldest'` to drop the oldest
                    point from the tabu list or `'farthest'` (defauilt) to
                    drop the element farthest from the last local minimum
                    found.
                disp: bool, optional
                    Set to True to print convergence messages (default is
                    False).

        Returns
        -------
        MinimizerResult
            Object containing the parameters from the ampgo method, with
            fit parameters, statistics and such. The return values
            (``x0``, ``fval``, ``eval``, ``msg``, ``tunnel``) are stored
            as ``ampgo_<parname>`` attributes.


        .. versionadded:: 0.9.10


        Notes
        -----
        The Python implementation was written by Andrea Gavana in 2014
        (http://infinity77.net/global_optimization/index.html).

        The details of the AMPGO algorithm are described in the paper
        "Adaptive Memory Programming for Constrained Global Optimization"
        located here:

        http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

        """
        result = self.prepare_fit(params=params)
        self.set_max_nfev(max_nfev, 200000*(result.nvarys+1))

        ampgo_kws = dict(local='L-BFGS-B', local_opts=None, maxfunevals=None,
                         totaliter=20, maxiter=5, glbtol=1e-5, eps1=0.02,
                         eps2=0.1, tabulistsize=5, tabustrategy='farthest',
                         disp=False)
        ampgo_kws.update(self.kws)
        ampgo_kws.update(kws)

        values = result.init_vals
        result.method = f"ampgo, with {ampgo_kws['local']} as local solver"
        result.call_kws = ampgo_kws
        try:
            ret = ampgo(self.penalty, values, **ampgo_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            result.ampgo_x0 = ret[0]
            result.ampgo_fval = ret[1]
            result.ampgo_eval = ret[2]
            result.ampgo_msg = ret[3]
            result.ampgo_tunnel = ret[4]

            for i, par in enumerate(result.var_names):
                result.params[par].value = result.ampgo_x0[i]

            result.residual = self.__residual(result.ampgo_x0)
            result.nfev -= 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
                len(result.residual) > len(result.var_names)):
            _covar_ndt = self._calculate_covariance_matrix(result.ampgo_x0)
            if _covar_ndt is not None:
                result.covar = self._int2ext_cov_x(_covar_ndt, result.ampgo_x0)
                self._calculate_uncertainties_correlations()

        return result

    def shgo(self, params=None, max_nfev=None, **kws):
        """Use the `SHGO` algorithm to find the global minimum.

        SHGO stands for "simplicial homology global optimization" and
        calls :scipydoc:`optimize.shgo` using its default arguments.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model. If None, then the
            Parameters used to initialize the Minimizer object are used.
        max_nfev : int or None, optional
            Maximum number of function evaluations. Defaults to
            ``200000*(nvars+1)``, where ``nvars`` is the number of variable
            parameters.
        **kws : dict, optional
            Minimizer options to pass to the SHGO algorithm.

        Returns
        -------
        MinimizerResult
            Object containing the parameters from the SHGO method.
            The return values specific to :scipydoc:`optimize.shgo`
            (``x``, ``xl``, ``fun``, ``funl``, ``nfev``, ``nit``,
            ``nlfev``, ``nlhev``, and ``nljev``) are stored as
            ``shgo_<parname>`` attributes.


        .. versionadded:: 0.9.14

        """
        result = self.prepare_fit(params=params)
        result.method = 'shgo'

        self.set_max_nfev(max_nfev, 200000*(result.nvarys+1))

        shgo_kws = dict(constraints=None, n=100, iters=1, callback=None,
                        minimizer_kwargs=None, options=None,
                        sampling_method='simplicial')

        shgo_kws.update(self.kws)
        shgo_kws.update(kws)

        varying = np.asarray([par.vary for par in self.params.values()])
        bounds = np.asarray([(par.min, par.max) for par in
                             self.params.values()])[varying]
        result.call_kws = shgo_kws
        try:
            ret = scipy_shgo(self.penalty, bounds, **shgo_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            for attr, value in ret.items():
                if attr in ['success', 'message']:
                    setattr(result, attr, value)
                else:
                    setattr(result, f'shgo_{attr}', value)

            result.residual = self.__residual(result.shgo_x, False)
            result.nfev -= 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
                len(result.residual) > len(result.var_names)):
            result.covar = self._calculate_covariance_matrix(result.shgo_x)
            if result.covar is not None:
                self._calculate_uncertainties_correlations()

        return result

    def dual_annealing(self, params=None, max_nfev=None, **kws):
        """Use the `dual_annealing` algorithm to find the global minimum.

        This method calls :scipydoc:`optimize.dual_annealing` using its
        default arguments.

        Parameters
        ----------
        params : Parameters, optional
            Contains the Parameters for the model. If None, then the
            Parameters used to initialize the Minimizer object are used.
        max_nfev : int or None, optional
            Maximum number of function evaluations. Defaults to
            ``200000*(nvars+1)``, where ``nvars`` is the number of variables.
        **kws : dict, optional
            Minimizer options to pass to the dual_annealing algorithm.

        Returns
        -------
        MinimizerResult
            Object containing the parameters from the dual_annealing
            method. The return values specific to
            :scipydoc:`optimize.dual_annealing` (``x``, ``fun``, ``nfev``,
            ``nhev``, ``njev``, and ``nit``) are stored as
            ``da_<parname>`` attributes.


        .. versionadded:: 0.9.14

        """
        result = self.prepare_fit(params=params)
        result.method = 'dual_annealing'
        self.set_max_nfev(max_nfev, 200000*(result.nvarys+1))

        da_kws = dict(maxiter=1000, local_search_options={},
                      initial_temp=5230.0, restart_temp_ratio=2e-05,
                      visit=2.62, accept=-5.0, maxfun=2*self.max_nfev,
                      seed=None, no_local_search=False, callback=None,
                      x0=None)

        da_kws.update(self.kws)
        da_kws.update(kws)

        varying = np.asarray([par.vary for par in self.params.values()])
        bounds = np.asarray([(par.min, par.max) for par in
                             self.params.values()])[varying]

        if not np.all(np.isfinite(bounds)):
            raise ValueError('dual_annealing requires finite bounds for all'
                             ' varying parameters')
        result.call_kws = da_kws
        try:
            ret = scipy_dual_annealing(self.penalty, bounds, **da_kws)
        except AbortFitException:
            pass

        if not result.aborted:
            for attr, value in ret.items():
                if attr in ['success', 'message']:
                    setattr(result, attr, value)
                else:
                    setattr(result, f'da_{attr}', value)

            result.residual = self.__residual(result.da_x, False)
            result.nfev -= 1

        result._calculate_statistics()

        # calculate the cov_x and estimate uncertainties/correlations
        if (not result.aborted and self.calc_covar and HAS_NUMDIFFTOOLS and
                len(result.residual) > len(result.var_names)):
            result.covar = self._calculate_covariance_matrix(result.da_x)
            if result.covar is not None:
                self._calculate_uncertainties_correlations()

        return result

    def minimize(self, method='leastsq', params=None, **kws):
        """Perform the minimization.

        Parameters
        ----------
        method : str, optional
            Name of the fitting method to use. Valid values are:

            - `'leastsq'`: Levenberg-Marquardt (default)
            - `'least_squares'`: Least-Squares minimization, using Trust
              Region Reflective method
            - `'differential_evolution'`: differential evolution
            - `'brute'`: brute force method
            - `'basinhopping'`: basinhopping
            - `'ampgo'`: Adaptive Memory Programming for Global
              Optimization
            - '`nelder`': Nelder-Mead
            - `'lbfgsb'`: L-BFGS-B
            - `'powell'`: Powell
            - `'cg'`: Conjugate-Gradient
            - `'newton'`: Newton-CG
            - `'cobyla'`: Cobyla
            - `'bfgs'`: BFGS
            - `'tnc'`: Truncated Newton
            - `'trust-ncg'`: Newton-CG trust-region
            - `'trust-exact'`: nearly exact trust-region
            - `'trust-krylov'`: Newton GLTR trust-region
            - `'trust-constr'`: trust-region for constrained optimization
            - `'dogleg'`: Dog-leg trust-region
            - `'slsqp'`: Sequential Linear Squares Programming
            - `'emcee'`: Maximum likelihood via Monte-Carlo Markov Chain
            - `'shgo'`: Simplicial Homology Global Optimization
            - `'dual_annealing'`: Dual Annealing optimization

            In most cases, these methods wrap and use the method with the
            same name from `scipy.optimize`, or use
            `scipy.optimize.minimize` with the same `method` argument.
            Thus `'leastsq'` will use `scipy.optimize.leastsq`, while
            `'powell'` will use `scipy.optimize.minimizer(...,
            method='powell')`.

            For more details on the fitting methods please refer to the
            `SciPy documentation
            <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__.

        params : Parameters, optional
            Parameters of the model to use as starting values.
        **kws : optional
            Additional arguments are passed to the underlying minimization
            method.

        Returns
        -------
        MinimizerResult
            Object containing the optimized parameters and several
            goodness-of-fit statistics.


        .. versionchanged:: 0.9.0
           Return value changed to :class:`MinimizerResult`.

        """
        kwargs = {'params': params}
        kwargs.update(self.kws)
        for maxnfev_alias in ('maxfev', 'maxiter'):
            if maxnfev_alias in kws:
                warnings.warn(maxeval_warning.format(maxnfev_alias, thisfuncname()),
                              RuntimeWarning)
                kws.pop(maxnfev_alias)

        kwargs.update(kws)

        user_method = method.lower()
        if user_method.startswith('leasts'):
            function = self.leastsq
        elif user_method.startswith('least_s'):
            function = self.least_squares
        elif user_method == 'brute':
            function = self.brute
        elif user_method == 'basinhopping':
            function = self.basinhopping
        elif user_method == 'ampgo':
            function = self.ampgo
        elif user_method == 'emcee':
            function = self.emcee
        elif user_method == 'shgo':
            function = self.shgo
        elif user_method == 'dual_annealing':
            function = self.dual_annealing
        else:
            function = self.scalar_minimize
            for key, val in SCALAR_METHODS.items():
                if (key.lower().startswith(user_method) or
                        val.lower().startswith(user_method)):
                    kwargs['method'] = val
        return function(**kwargs)


def _make_random_gen(seed):
    """Turn seed into a numpy.random.RandomState instance.

    If `seed` is None, return the RandomState singleton used by numpy.random.
    If `seed` is an int, return a new RandomState instance seeded with seed.
    If `seed` is already a RandomState instance, return it.
    Otherwise raise a `ValueError`.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f'{seed:r} cannot be used to seed a numpy.random.RandomState'
                     ' instance')


def _nan_policy(arr, nan_policy='raise', handle_inf=True):
    """Specify behaviour when array contains ``numpy.nan`` or ``numpy.inf``.

    Parameters
    ----------
    arr : array_like
        Input array to consider.
    nan_policy : {'raise', 'propagate', 'omit'}, optional
        One of:

        `'raise'` - raise a `ValueError` if `arr` contains NaN (default)
        `'propagate'` - propagate NaN
        `'omit'` - filter NaN from input array

    handle_inf : bool, optional
        Whether to apply the `nan_policy` to +/- infinity (default is
        True).

    Returns
    -------
    array_like
        Result of `arr` after applying the `nan_policy`.

    Notes
    -----
    This function is copied, then modified, from
    scipy/stats/stats.py/_contains_nan

    """
    if nan_policy not in ('propagate', 'omit', 'raise'):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")

    if handle_inf:
        handler_func = lambda x: ~np.isfinite(x)
    else:
        handler_func = isnull

    if nan_policy == 'omit':
        # mask locates any values to remove
        mask = ~handler_func(arr)
        if not np.all(mask):  # there are some NaNs/infs/missing values
            return arr[mask]

    if nan_policy == 'raise':
        try:
            # Calling np.sum to avoid creating a huge array into memory
            # e.g. np.isnan(a).any()
            with np.errstate(invalid='ignore'):
                contains_nan = handler_func(np.sum(arr))
        except TypeError:
            # If the check cannot be properly performed we fallback to omitting
            # nan values and raising a warning. This can happen when attempting to
            # sum things that are not numbers (e.g. as in the function `mode`).
            contains_nan = False
            warnings.warn("The input array could not be checked for NaNs. "
                          "NaNs will be ignored.", RuntimeWarning)

        if contains_nan:
            msg = ('NaN values detected in your input data or the output of '
                   'your objective/model function - fitting algorithms cannot '
                   'handle this! Please read https://lmfit.github.io/lmfit-py/faq.html#i-get-errors-from-nan-in-my-fit-what-can-i-do '
                   'for more information.')
            raise ValueError(msg)
    return arr


def minimize(fcn, params, method='leastsq', args=None, kws=None, iter_cb=None,
             scale_covar=True, nan_policy='raise', reduce_fcn=None,
             calc_covar=True, max_nfev=None, **fit_kws):
    """Perform the minimization of the objective function.

    The minimize function takes an objective function to be minimized,
    a dictionary (:class:`~lmfit.parameter.Parameters` ; Parameters) containing
    the model parameters, and several optional arguments including the fitting
    method.

    Parameters
    ----------
    fcn : callable
        Objective function to be minimized. When method is `'leastsq'` or
        '`least_squares`', the objective function should return an array
        of residuals (difference between model and data) to be minimized
        in a least-squares sense. With the scalar methods the objective
        function can either return the residuals array or a single scalar
        value. The function must have the signature::

            fcn(params, *args, **kws)

    params : Parameters
        Contains the Parameters for the model.
    method : str, optional
        Name of the fitting method to use. Valid values are:

        - `'leastsq'`: Levenberg-Marquardt (default)
        - `'least_squares'`: Least-Squares minimization, using Trust Region Reflective method
        - `'differential_evolution'`: differential evolution
        - `'brute'`: brute force method
        - `'basinhopping'`: basinhopping
        - `'ampgo'`: Adaptive Memory Programming for Global Optimization
        - '`nelder`': Nelder-Mead
        - `'lbfgsb'`: L-BFGS-B
        - `'powell'`: Powell
        - `'cg'`: Conjugate-Gradient
        - `'newton'`: Newton-CG
        - `'cobyla'`: Cobyla
        - `'bfgs'`: BFGS
        - `'tnc'`: Truncated Newton
        - `'trust-ncg'`: Newton-CG trust-region
        - `'trust-exact'`: nearly exact trust-region
        - `'trust-krylov'`: Newton GLTR trust-region
        - `'trust-constr'`: trust-region for constrained optimization
        - `'dogleg'`: Dog-leg trust-region
        - `'slsqp'`: Sequential Linear Squares Programming
        - `'emcee'`: Maximum likelihood via Monte-Carlo Markov Chain
        - `'shgo'`: Simplicial Homology Global Optimization
        - `'dual_annealing'`: Dual Annealing optimization

        In most cases, these methods wrap and use the method of the same
        name from `scipy.optimize`, or use `scipy.optimize.minimize` with
        the same `method` argument. Thus `'leastsq'` will use
        `scipy.optimize.leastsq`, while `'powell'` will use
        `scipy.optimize.minimizer(..., method='powell')`

        For more details on the fitting methods please refer to the
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/optimize.html>`__.

    args : tuple, optional
        Positional arguments to pass to `fcn`.
    kws : dict, optional
        Keyword arguments to pass to `fcn`.
    iter_cb : callable, optional
        Function to be called at each fit iteration. This function should
        have the signature::

            iter_cb(params, iter, resid, *args, **kws),

        where `params` will have the current parameter values, `iter` the
        iteration number, `resid` the current residual array, and `*args`
        and `**kws` as passed to the objective function.
    scale_covar : bool, optional
        Whether to automatically scale the covariance matrix (default is
        True).
    nan_policy : {'raise', 'propagate', 'omit'}, optional
        Specifies action if `fcn` (or a Jacobian) returns NaN values. One
        of:

        - `'raise'` : a `ValueError` is raised
        - `'propagate'` : the values returned from `userfcn` are un-altered
        - `'omit'` : non-finite values are filtered

    reduce_fcn : str or callable, optional
        Function to convert a residual array to a scalar value for the
        scalar minimizers. See Notes in `Minimizer`.
    calc_covar : bool, optional
        Whether to calculate the covariance matrix (default is True) for
        solvers other than `'leastsq'` and `'least_squares'`. Requires the
        `numdifftools` package to be installed.
    max_nfev : int or None, optional
        Maximum number of function evaluations (default is None). The
        default value depends on the fitting method.
    **fit_kws : dict, optional
        Options to pass to the minimizer being used.

    Returns
    -------
    MinimizerResult
        Object containing the optimized parameters and several
        goodness-of-fit statistics.


    .. versionchanged:: 0.9.0
       Return value changed to :class:`MinimizerResult`.


    Notes
    -----
    The objective function should return the value to be minimized. For
    the Levenberg-Marquardt algorithm from leastsq(), this returned value
    must be an array, with a length greater than or equal to the number of
    fitting variables in the model. For the other methods, the return
    value can either be a scalar or an array. If an array is returned, the
    sum-of- squares of the array will be sent to the underlying fitting
    method, effectively doing a least-squares optimization of the return
    values.

    A common use for `args` and `kws` would be to pass in other data needed
    to calculate the residual, including such things as the data array,
    dependent variable, uncertainties in the data, and other data structures
    for the model calculation.

    On output, `params` will be unchanged. The best-fit values and, where
    appropriate, estimated uncertainties and correlations, will all be
    contained in the returned :class:`MinimizerResult`. See
    :ref:`fit-results-label` for further details.

    This function is simply a wrapper around :class:`Minimizer` and is
    equivalent to::

        fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                           iter_cb=iter_cb, scale_covar=scale_covar,
                           nan_policy=nan_policy, reduce_fcn=reduce_fcn,
                           calc_covar=calc_covar, **fit_kws)
        fitter.minimize(method=method)

    """
    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                       iter_cb=iter_cb, scale_covar=scale_covar,
                       nan_policy=nan_policy, reduce_fcn=reduce_fcn,
                       calc_covar=calc_covar, max_nfev=max_nfev, **fit_kws)
    return fitter.minimize(method=method)
