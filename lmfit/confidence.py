#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains functions to calculate confidence intervals.
"""
from __future__ import print_function
from warnings import warn
import numpy as np
from scipy.stats import f
from scipy.optimize import brentq
from .minimizer import MinimizerException

CONF_ERR_GEN    = 'Cannot determine Confidence Intervals'
CONF_ERR_STDERR = '%s without sensible uncertainty estimates' % CONF_ERR_GEN
CONF_ERR_NVARS  = '%s with < 2 variables' % CONF_ERR_GEN

def f_compare(ndata, nparas, new_chi, best_chi, nfix=1.):
    """
    Returns the probalitiy for two given parameter sets.
    nfix is the number of fixed parameters.
    """
    nparas = nparas + nfix
    nfree = ndata - nparas
    nfix = 1.0*nfix
    dchi = new_chi / best_chi - 1.0
    return f.cdf(dchi * nfree / nfix, nfix, nfree)


def copy_vals(params):
    """Saves the values and stderrs of params in temporay dict"""
    tmp_params = {}
    for para_key in params:
        tmp_params[para_key] = (params[para_key].value,
                                params[para_key].stderr)
    return tmp_params


def restore_vals(tmp_params, params):
    """Restores values and stderrs of params in temporay dict"""
    for para_key in params:
        params[para_key].value, params[para_key].stderr = tmp_params[para_key]


def conf_interval(minimizer, result, p_names=None, sigmas=(0.674, 0.95, 0.997),
                  trace=False, maxiter=200, verbose=False, prob_func=None):
    r"""Calculates the confidence interval for parameters
    from the given a MinimizerResult, output from minimize.

    The parameter for which the ci is calculated will be varied, while
    the remaining parameters are re-optimized for minimizing chi-square.
    The resulting chi-square is used  to calculate the probability with
    a given statistic e.g. F-statistic. This function uses a 1d-rootfinder
    from scipy to find the values resulting in the searched confidence
    region.

    Parameters
    ----------
    minimizer : Minimizer
        The minimizer to use, holding objective function.
    result : MinimizerResult
        The result of running minimize().
    p_names : list, optional
        Names of the parameters for which the ci is calculated. If None,
        the ci is calculated for every parameter.
    sigmas : list, optional
        The probabilities (1-alpha) to find. Default is 1,2 and 3-sigma.
    trace : bool, optional
        Defaults to False, if true, each result of a probability calculation
        is saved along with the parameter. This can be used to plot so
        called "profile traces".

    Returns
    -------
    output : dict
        A dict, which contains a list of (sigma, vals)-tuples for each name.
    trace_dict : dict
        Only if trace is set true. Is a dict, the key is the parameter which
        was fixed.The values are again a dict with the names as keys, but with
        an additional key 'prob'. Each contains an array of the corresponding
        values.

    See also
    --------
    conf_interval2d

    Other Parameters
    ----------------
    maxiter : int
        Maximum of iteration to find an upper limit.
    prob_func : ``None`` or callable
        Function to calculate the probability from the optimized chi-square.
        Default (``None``) uses built-in f_compare (F test).
    verbose: bool
        print extra debuggin information. Default is ``False``.


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

    >>> ci, trace = conf_interval(mini, sigmas=(0.25, 0.5, 0.75, 0.999), trace=True)
    >>> fixed = trace['para1']['para1']
    >>> free = trace['para1']['not_para1']
    >>> prob = trace['para1']['prob']

    This makes it possible to plot the dependence between free and fixed.
    """
    ci = ConfidenceInterval(minimizer, result, p_names, prob_func, sigmas,
                            trace, verbose, maxiter)
    output = ci.calc_all_ci()
    if trace:
        return output, ci.trace_dict
    return output


def map_trace_to_names(trace, params):
    "maps trace to param names"
    out = {}
    allnames = list(params.keys()) + ['prob']
    for name in trace.keys():
        tmp_dict = {}
        tmp = np.array(trace[name])
        for para_name, values in zip(allnames, tmp.T):
            tmp_dict[para_name] = values
        out[name] = tmp_dict
    return out


class ConfidenceInterval(object):
    """
    Class used to calculate the confidence interval.
    """
    def __init__(self, minimizer, result, p_names=None, prob_func=None,
                 sigmas=(0.674, 0.95, 0.997), trace=False, verbose=False,
                 maxiter=50):
        """

        """
        self.verbose = verbose
        self.minimizer = minimizer
        self.result = result
        self.params = result.params
        self.org = copy_vals(self.params)
        self.best_chi = result.chisqr

        if p_names is None:
            p_names = [i for i in self.params if self.params[i].vary]

        self.p_names = p_names
        self.fit_params = [self.params[p] for p in self.p_names]

        # check that there are at least 2 true variables!
        # check that all stderrs are sensible (including not None or NaN)
        nvars = 0
        for par in self.fit_params:
            if par.vary:
                nvars += 1
            try:
                if not (par.vary and par.stderr > 0.0):
                    raise MinimizerException(CONF_ERR_STDERR)
            except TypeError:
                raise MinimizerException(CONF_ERR_STDERR)
        if nvars < 2:
            raise MinimizerException(CONF_ERR_NVARS)

        if prob_func is None or not hasattr(prob_func, '__call__'):
            self.prob_func = f_compare
        if trace:
            self.trace_dict = dict([(i, []) for i in self.p_names])

        self.trace = trace
        self.maxiter = maxiter
        self.min_rel_change = 1e-5

        self.sigmas = list(sigmas)
        self.sigmas.sort()

    def calc_all_ci(self):
        """
        Calculates all cis.
        """
        out = {}

        for p in self.p_names:
            out[p] = (self.calc_ci(p, -1)[::-1] +
                      [(0., self.params[p].value)] +
                      self.calc_ci(p, 1))
        if self.trace:
            self.trace_dict = map_trace_to_names(self.trace_dict,
                                                 self.params)

        return out

    def calc_ci(self, para, direction):
        """
        Calculate the ci for a single parameter for a single direction.
        Direction is either positive or negative 1.
        """

        if isinstance(para, str):
            para = self.params[para]

        #function used to calculate the pro
        calc_prob = lambda val, prob: self.calc_prob(para, val, prob)
        if self.trace:
            x = [i.value for i in self.params.values()]
            self.trace_dict[para.name].append(x + [0])

        para.vary = False
        limit, max_prob = self.find_limit(para, direction)
        start_val = para.value.copy()
        a_limit = start_val.copy()
        ret = []
        orig_warn_settings = np.geterr()
        np.seterr(all='ignore')
        for prob in self.sigmas:
            if prob > max_prob:
                ret.append((prob, direction*np.inf))
                continue

            try:
                val = brentq(calc_prob, a_limit,
                             limit, rtol=.5e-4, args=prob)

            except ValueError:
                self.reset_vals()
                try:
                    val = brentq(calc_prob, start_val,
                                 limit, rtol=.5e-4, args=prob)
                except ValueError:
                    val = np.nan

            a_limit = val
            ret.append((prob, val))

        para.vary = True
        self.reset_vals()
        np.seterr(**orig_warn_settings)
        return ret

    def reset_vals(self):
        restore_vals(self.org, self.params)

    def find_limit(self, para, direction):
        """
        For given para, search a value so that prob(val) > sigmas.
        """
        if self.verbose:
            print('Calculating CI for ' + para.name)
        self.reset_vals()

        #starting steps:
        if para.stderr > 0 and para.stderr < abs(para.value):
            step = para.stderr
        else:
            step = max(abs(para.value) * 0.2, 0.001)
        para.vary = False
        start_val = para.value

        old_prob = 0
        limit = start_val
        i = 0

        while old_prob < max(self.sigmas):
            i = i + 1
            limit += step * direction

            new_prob = self.calc_prob(para, limit)
            rel_change = (new_prob - old_prob) / max(new_prob, old_prob, 1.e-12)
            old_prob = new_prob

            # Check convergence.
            if i > self.maxiter:
                errmsg = "Warning, maxiter={0} reached".format(self.maxiter)
                errmsg += "and prob({0}={1}) = {2} < max(sigmas).".format(para.name, limit, new_prob)
                warn(errmsg)
                break

            if rel_change < self.min_rel_change:
                errmsg = "Warning, rel_change={0} < 0.01 ".format(rel_change)
                errmsg += " at iteration {3} and prob({0}={1}) = {2} < max(sigmas).".format(para.name, limit, new_prob, i)
                warn(errmsg)
                break

        self.reset_vals()

        return limit, new_prob

    def calc_prob(self, para, val, offset=0., restore=False):
        """Returns the probability for given Value."""
        if restore:
            restore_vals(self.org, self.params)
        para.value = val
        save_para = self.params[para.name]
        self.params[para.name] = para
        self.minimizer.prepare_fit(self.params)
        out = self.minimizer.leastsq()
        prob = self.prob_func(out.ndata, out.ndata - out.nfree,
                              out.chisqr, self.best_chi)

        if self.trace:
            x = [i.value for i in out.params.values()]
            self.trace_dict[para.name].append(x + [prob])
        self.params[para.name] = save_para
        return prob - offset

def conf_interval2d(minimizer, result, x_name, y_name, nx=10, ny=10,
                    limits=None, prob_func=None):
    r"""Calculates confidence regions for two fixed parameters.

    The method is explained in *conf_interval*: here we are fixing
    two parameters.

    Parameters
    ----------
    minimizer : Minimizer
        The minimizer to use, holding objective function.
    result : MinimizerResult
        The result of running minimize().
    x_name : string
        The name of the parameter which will be the x direction.
    y_name : string
        The name of the parameter which will be the y direction.
    nx, ny : ints, optional
        Number of points.
    limits : tuple: optional
        Should have the form ((x_upper, x_lower),(y_upper, y_lower)). If not
        given, the default is 5 std-errs in each direction.

    Returns
    -------
    x : (nx)-array
        x-coordinates
    y : (ny)-array
        y-coordinates
    grid : (nx,ny)-array
        grid contains the calculated probabilities.

    Examples
    --------

    >>> mini = Minimizer(some_func, params)
    >>> result = mini.leastsq()
    >>> x, y, gr = conf_interval2d(mini, result, 'para1','para2')
    >>> plt.contour(x,y,gr)

    Other Parameters
    ----------------
    prob_func : ``None`` or callable
        Function to calculate the probability from the optimized chi-square.
        Default (``None``) uses built-in f_compare (F test).
    """
    # used to detect that .leastsq() has run!
    params = result.params

    best_chi = result.chisqr
    org = copy_vals(result.params)

    if prob_func is None or not hasattr(prob_func, '__call__'):
        prob_func = f_compare

    x = params[x_name]
    y = params[y_name]

    if limits is None:
        (x_upper, x_lower) = (x.value + 5 * x.stderr, x.value - 5
                                                      * x.stderr)
        (y_upper, y_lower) = (y.value + 5 * y.stderr, y.value - 5
                                                      * y.stderr)
    elif len(limits) == 2:
        (x_upper, x_lower) = limits[0]
        (y_upper, y_lower) = limits[1]

    x_points = np.linspace(x_lower, x_upper, nx)
    y_points = np.linspace(y_lower, y_upper, ny)
    grid = np.dstack(np.meshgrid(x_points, y_points))

    x.vary = False
    y.vary = False

    def calc_prob(vals, restore=False):
        if restore:
            restore_vals(org, result.params)
        x.value = vals[0]
        y.value = vals[1]
        save_x = result.params[x.name]
        save_y = result.params[y.name]
        result.params[x.name] = x
        result.params[y.name] = y
        minimizer.prepare_fit(params=result.params)
        out = minimizer.leastsq()
        prob = prob_func(out.ndata, out.ndata - out.nfree, out.chisqr,
                         best_chi, nfix=2.)
        result.params[x.name] = save_x
        result.params[y.name] = save_y
        return prob

    out = x_points, y_points, np.apply_along_axis(calc_prob, -1, grid)

    x.vary, y.vary = True, True
    restore_vals(org, result.params)
    result.chisqr = best_chi
    return out
