"""Contains functions to calculate confidence intervals."""

from warnings import warn

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import erf
from scipy.stats import f

from .minimizer import MinimizerException

CONF_ERR_GEN = 'Cannot determine Confidence Intervals'
CONF_ERR_STDERR = f'{CONF_ERR_GEN} without sensible uncertainty estimates'
CONF_ERR_NVARS = f'{CONF_ERR_GEN} with < 2 variables'


def f_compare(best_fit, new_fit):
    """Return the probability calculated using the F-test.

    The null model (i.e., best-fit solution) is compared to an alternate
    model where one or more parameters are fixed.

    Parameters
    ----------
    best_fit : MinimizerResult
        The result from the best-fit.
    new_fit : MinimizerResult
        The result from fit with the fixed parameter(s).

    Returns
    -------
    float
        Value of the calculated probability.

    """
    nfree = best_fit.nfree
    nfix = best_fit.nvarys - new_fit.nvarys
    dchi = new_fit.chisqr / best_fit.chisqr - 1.0
    return f.cdf(dchi * nfree / nfix, nfix, nfree)


def copy_vals(params):
    """Save values/stderrs of parameters in a temporary dictionary."""
    tmp_params = {}
    for para_key in params:
        tmp_params[para_key] = (params[para_key].value,
                                params[para_key].stderr)
    return tmp_params


def restore_vals(tmp_params, params):
    """Restore values/stderrs of parameters from a temporary dictionary."""
    for para_key in params:
        params[para_key].value, params[para_key].stderr = tmp_params[para_key]


def conf_interval(minimizer, result, p_names=None, sigmas=None, trace=False,
                  maxiter=200, verbose=False, prob_func=None,
                  min_rel_change=1e-5):
    """Calculate the confidence interval (CI) for parameters.

    The parameter for which the CI is calculated will be varied, while the
    remaining parameters are re-optimized to minimize the chi-square. The
    resulting chi-square is used to calculate the probability with a given
    statistic (e.g., F-test). This function uses a 1d-rootfinder from SciPy
    to find the values resulting in the searched confidence region.

    Parameters
    ----------
    minimizer : Minimizer
        The minimizer to use, holding objective function.
    result : MinimizerResult
        The result of running minimize().
    p_names : list, optional
        Names of the parameters for which the CI is calculated. If None
        (default), the CI is calculated for every parameter.
    sigmas : list, optional
        The sigma-levels to find (default is [1, 2, 3]). See Notes below.
    trace : bool, optional
        Defaults to False; if True, each result of a probability
        calculation is saved along with the parameter. This can be used to
        plot so-called "profile traces".
    maxiter : int, optional
        Maximum of iteration to find an upper limit (default is 200).
    verbose : bool, optional
        Print extra debugging information (default is False).
    prob_func : None or callable, optional
        Function to calculate the probability from the optimized chi-square.
        Default is None and uses the built-in function `f_compare`
        (i.e., F-test).
    min_rel_change : float, optional
        Minimum relative change in probability (default is 1e-5).

    Returns
    -------
    output : dict
        A dictionary containing a list of ``(sigma, vals)``-tuples for
        each parameter.
    trace_dict : dict, optional
        Only if trace is True. Is a dictionary, the key is the parameter
        which was fixed. The values are again a dict with the names as
        keys, but with an additional key 'prob'. Each contains an array
        of the corresponding values.

    See Also
    --------
    conf_interval2d

    Notes
    -----
    The values for `sigma` are taken as the number of standard deviations
    for a normal distribution and converted to probabilities. That is, the
    default ``sigma=[1, 2, 3]`` will use probabilities of 0.6827, 0.9545,
    and 0.9973. If any of the sigma values is less than 1, that will be
    interpreted as a probability. That is, a value of 1 and 0.6827 will
    give the same results, within precision.

    Examples
    --------
    >>> from lmfit.printfuncs import *
    >>> mini = minimize(some_func, params)
    >>> mini.leastsq()
    True
    >>> report_errors(params)
    ... #report
    >>> ci = conf_interval(mini)
    >>> report_ci(ci)
    ... #report

    Now with quantiles for the sigmas and using the trace.

    >>> ci, trace = conf_interval(mini, sigmas=[0.5, 1, 2, 3], trace=True)
    >>> fixed = trace['para1']['para1']
    >>> free = trace['para1']['not_para1']
    >>> prob = trace['para1']['prob']

    This makes it possible to plot the dependence between free and fixed
    parameters.

    """
    if sigmas is None:
        sigmas = [1, 2, 3]

    ci = ConfidenceInterval(minimizer, result, p_names, prob_func, sigmas,
                            trace, verbose, maxiter, min_rel_change)
    output = ci.calc_all_ci()
    if trace:
        return output, ci.trace_dict
    return output


def map_trace_to_names(trace, params):
    """Map trace to parameter names."""
    out = {}
    allnames = list(params.keys()) + ['prob']
    for name in trace.keys():
        tmp_dict = {}
        tmp = np.array(trace[name])
        for para_name, values in zip(allnames, tmp.T):
            tmp_dict[para_name] = values
        out[name] = tmp_dict
    return out


class ConfidenceInterval:
    """Class used to calculate the confidence interval."""

    def __init__(self, minimizer, result, p_names=None, prob_func=None,
                 sigmas=None, trace=False, verbose=False, maxiter=50,
                 min_rel_change=1e-5):
        """Initialize the ConfidenceInterval class.

        Parameters
        ----------
        minimizer : Minimizer
            The minimizer to use, holding objective function.
        result : MinimizerResult
            The result of running minimize().
        p_names : list, optional
            Names of the parameters for which the CI is calculated. If None
            (default), the CI is calculated for every parameter.
        prob_func : None or callable, optional
            Function to calculate the probability from the optimized chi-square.
            Default is None and uses the built-in function `f_compare`
            (i.e., F-test).
        sigmas : list, optional
            The sigma-levels to find (default is [1, 2, 3]).
        trace : bool, optional
            Defaults to False; if True, each result of a probability
            calculation is saved along with the parameter. This can be used to
            plot so-called "profile traces".
        verbose : bool, optional
            Print extra debugging information (default is False).
        maxiter : int, optional
            Maximum of iteration to find an upper limit (default is 50).
        min_rel_change : float, optional
            Minimum relative change in probability (default is 1e-5).

        Raises
        ------
        MinimizerException
            If there are less than two variables or if the stderrs are not
            sensible.

        """
        self.verbose = verbose
        self.minimizer = minimizer
        self.result = result
        self.params = result.params.copy()
        self.org = copy_vals(self.params)
        self.best_chi = result.chisqr

        if p_names is None:
            p_names = [i for i in self.params if self.params[i].vary]

        self.p_names = p_names
        self.fit_params = [self.params[p] for p in self.p_names]

        # check that there are at least 2 true variables!
        # check that all stderrs are sensible (including not None or NaN)

        for par in self.fit_params:
            if par.vary and (par.stderr is None or par.stderr is np.nan):
                raise MinimizerException(CONF_ERR_STDERR)
        nvars = len([p for p in self.params.values() if p.vary])
        if nvars < 2:
            raise MinimizerException(CONF_ERR_NVARS)

        if prob_func is None:
            self.prob_func = f_compare
        else:
            self.prob_func = prob_func
        if trace:
            self.trace_dict = {i: [] for i in self.p_names}

        self.trace = trace
        self.maxiter = maxiter
        self.min_rel_change = min_rel_change

        if sigmas is None:
            sigmas = [1, 2, 3]
        self.sigmas = list(sigmas)
        self.sigmas.sort()
        self.probs = []
        for sigma in self.sigmas:
            if sigma < 1:
                prob = sigma
            else:
                prob = erf(sigma/np.sqrt(2))
            self.probs.append(prob)

    def calc_all_ci(self):
        """Calculate all confidence intervals."""
        out = {}

        for p in self.p_names:
            out[p] = (self.calc_ci(p, -1)[::-1] +
                      [(0., self.params[p].value)] +
                      self.calc_ci(p, 1))
        if self.trace:
            self.trace_dict = map_trace_to_names(self.trace_dict, self.params)

        return out

    def calc_ci(self, para, direction):
        """Calculate the CI for a single parameter in a single direction.

        Direction is either positive or negative 1.

        """
        if isinstance(para, str):
            para = self.params[para]

        # function used to calculate the probability
        cache = {}

        def calc_prob(val, target_prob):
            if val not in cache:
                cache[val] = self.calc_prob(para, val, 0)
            return cache[val] - target_prob

        if self.trace:
            x = [i.value for i in self.params.values()]
            self.trace_dict[para.name].append(x + [0])

        para.vary = False
        limit, max_prob = self.find_limit(para, direction)
        a_limit = float(para.value)
        ret = []
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')
        for prob in self.probs:
            if prob > max_prob:
                ret.append((prob, direction*np.inf))
                continue

            sol = root_scalar(calc_prob, method='toms748', bracket=sorted([limit, a_limit]), rtol=.5e-4, args=(prob,))
            if sol.converged:
                val = sol.root
            else:
                val = np.nan
                break
            a_limit = val
            ret.append((prob, val))

        para.vary = True
        self.reset_vals()
        np.seterr(**orig_warn_settings)
        return ret

    def reset_vals(self):
        """Reset parameter values to best-fit values."""
        restore_vals(self.org, self.params)

    def find_limit(self, para, direction):
        """Find a value for given parameter so that prob(val) > sigmas."""
        if self.verbose:
            print(f'Calculating CI for {para.name}')
        self.reset_vals()

        # determine starting step
        if para.stderr > 0 and para.stderr < abs(para.value):
            step = para.stderr
        else:
            step = max(abs(para.value) * 0.2, 0.001)
        para.vary = False
        start_val = para.value

        old_prob = 0
        limit = start_val
        i = 0
        bound_reached = False
        max_prob = max(self.probs)

        while old_prob < max_prob:
            i += 1
            limit += step * direction
            if limit > para.max:
                limit = para.max
                bound_reached = True
            elif limit < para.min:
                limit = para.min
                bound_reached = True

            new_prob = self.calc_prob(para, limit)
            rel_change = (new_prob - old_prob) / max(new_prob, old_prob, 1e-12)
            old_prob = new_prob
            if self.verbose:
                print(f'P({para.name}={limit}) = {new_prob}, '
                      f'max. prob={max_prob}')

            # check for convergence
            if bound_reached and new_prob < max(self.probs):
                errmsg = (f'Bound reached with prob({para.name}={limit}) '
                          f'= {new_prob} < max(sigmas)')
                warn(errmsg)
                break

            if i > self.maxiter:
                errmsg = (f'maxiter={self.maxiter} reached and prob('
                          f'{para.name}={limit}) = {new_prob} < max(sigmas)')
                warn(errmsg)
                break

            if rel_change < self.min_rel_change:
                errmsg = (f'rel_change={rel_change} < {self.min_rel_change} '
                          f'at iteration {i} and prob({para.name}={limit}) = '
                          f'{new_prob} < max(sigmas)')
                warn(errmsg)
                break

        self.reset_vals()

        return limit, new_prob

    def calc_prob(self, para, val, offset=0., restore=False):
        """Calculate the probability for given value."""
        if restore:
            restore_vals(self.org, self.params)
        para.value = val
        save_para = self.params[para.name]
        self.params[para.name] = para
        self.minimizer.prepare_fit(self.params)
        out = self.minimizer.leastsq()
        prob = self.prob_func(self.result, out)

        if self.trace:
            x = [i.value for i in out.params.values()]
            self.trace_dict[para.name].append(x + [prob])
        self.params[para.name] = save_para
        return prob - offset


def conf_interval2d(minimizer, result, x_name, y_name, nx=10, ny=10,
                    limits=None, prob_func=None, nsigma=5, chi2_out=False):
    r"""Calculate confidence regions for two fixed parameters.

    The method itself is explained in `conf_interval`: here we are fixing
    two parameters.

    Parameters
    ----------
    minimizer : Minimizer
        The minimizer to use, holding objective function.
    result : MinimizerResult
        The result of running minimize().
    x_name : str
        The name of the parameter which will be the x direction.
    y_name : str
        The name of the parameter which will be the y direction.
    nx : int, optional
        Number of points in the x direction (default is 10).
    ny : int, optional
        Number of points in the y direction (default is 10).
    limits : tuple, optional
        Should have the form ``((x_upper, x_lower), (y_upper, y_lower))``.
        If not given, the default is nsigma*stderr in each direction.
    prob_func : None or callable, deprecated
        Starting with version 1.2, this argument is unused and has no effect.
    nsigma : float or int, optional
        Multiplier of stderr for limits (default is 5).
    chi2_out: bool
        Whether to return chi-square at each coordinate instead of probability.

    Returns
    -------
    x : numpy.ndarray
        X-coordinates (same shape as `nx`).
    y : numpy.ndarray
        Y-coordinates (same shape as `ny`).
    grid : numpy.ndarray
        2-D array (with shape ``(nx, ny)``) containing the calculated
        probabilities or chi-square.

    See Also
    --------
    conf_interval

    Examples
    --------
    >>> mini = Minimizer(some_func, params)
    >>> result = mini.leastsq()
    >>> x, y, gr = conf_interval2d(mini, result, 'para1','para2')
    >>> plt.contour(x,y,gr)

    """
    if prob_func is not None:
        msg = "'prob_func' has no effect and will be removed in version 1.4."
        raise DeprecationWarning(msg)

    params = result.params

    best_chisqr = result.chisqr
    redchi = result.redchi
    org = copy_vals(result.params)

    x = params[x_name]
    y = params[y_name]

    if limits is None:
        (x_upper, x_lower) = (x.value + nsigma * x.stderr, x.value - nsigma * x.stderr)
        (y_upper, y_lower) = (y.value + nsigma * y.stderr, y.value - nsigma * y.stderr)
    elif len(limits) == 2:
        (x_upper, x_lower) = limits[0]
        (y_upper, y_lower) = limits[1]

    x_points = np.linspace(x_lower, x_upper, nx)
    y_points = np.linspace(y_lower, y_upper, ny)
    grid = np.dstack(np.meshgrid(x_points, y_points))

    x.vary, y.vary = False, False

    def calc_chisqr(vals, restore=False):
        """Calculate chi-square for a set of parameter values."""
        save_x = x.value
        save_y = y.value
        result.params[x.name].value = vals[0]
        result.params[y.name].value = vals[1]
        minimizer.prepare_fit(params=result.params)
        out = minimizer.leastsq()
        result.params[x.name].value = save_x
        result.params[y.name].value = save_y
        return out.chisqr

    # grid of chi-square
    out_mat = np.apply_along_axis(calc_chisqr, -1, grid)

    # compute grid of sigma values from chi-square
    if not chi2_out:
        chisqr0 = out_mat.min()
        chisqr0 = min(best_chisqr, chisqr0)
        out_mat = np.sqrt((out_mat-chisqr0)/redchi)

    x.vary, y.vary = True, True
    restore_vals(org, result.params)
    result.chisqr = best_chisqr
    return x_points, y_points, out_mat
