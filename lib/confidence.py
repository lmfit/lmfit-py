#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains functions to calculate confidence intervals.
"""

import numpy as np
from scipy.stats import f
from scipy.optimize import brentq

# def calc_max_chi(Ndata, Npara, best_chi):
#    fval=f.isf(0.05,Npara, Ndata-Npara)
#    return best_chi*(fval*Npara/float(Ndata-Npara)+1)

def f_compare(Ndata, Nparas, new_chi, best_chi, Nfix=1.):
    """
    Returns the probalitiy for two given parameter sets.
    Nfix is the number of fixed parameters.
    """

    # print new_chi, best_chi, Ndata, Nparas

    Nparas = Nparas + Nfix
    return f.cdf((new_chi / best_chi - 1) * (Ndata - Nparas) / Nfix,
                 Nfix, Ndata - Nparas)


# def log_compare(Ndata,Nparas,new_chi,best_chi,Nfix=1.):
#    pass

def copy_vals(params):
    '''Saves the values of paras and errs in temporay dict'''

    tmp_params = {}
    for para_key in params:
        tmp_params[para_key] = (params[para_key].value,
                                params[para_key].stderr)
    return tmp_params


def restore_vals(tmp_params, params):
    '''Restores the values of params from a temporay dict'''

    for para_key in params:
        params[para_key].value, params[para_key].stderr = \
            tmp_params[para_key]


def p_trace_to_dict(p_tr, params):
    """
    p_tr has following form:
        
        ([[p1, p2,...],[p1, p2,...]],[res_prob1,res_prob2..])
        
    Returns a dict with p-names and prob as keys and lists as their values.
    """

    out = {}
    for name in params.keys():
        out[name] = np.array([l.pop(0) for l in p_tr[0]])
    out['prob'] = np.array(p_tr[1])
    return out


def conf_interval(minimizer, p_names=None, sigmas=(0.674, 0.95, 0.997),
                  trace=False, maxiter=200, verbose=False, prob_func=None):
    r"""Calculates the confidence interval for parameters
    from the given minimizer.
    
    The parameter for which the ci is calculated will be varied, while
    the remaining parameters are reoptimized for minimizing chi-square.
    The resulting chi-square is used  to calculate the probability with 
    a given statistic e.g. F-statistic. This function uses a 1d-rootfinder
    from scipy to find the values resulting in the searched confidence
    region. 

    Parameters
    ----------          
    minimizer : Minimizer
        The minimizer to use, should be already fitted via leastsq.
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
        A dict, which contais a list of (sigma, vals)-tuples for each name.
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
        Function to calculate the probality from the opimized chi-square.
        Default (``None``) uses built-in f_compare (F test).
    
   
    
    Examples
    --------
    
    >>> from lmfit.printfuncs import *
    >>> mini=minimize(some_func, params)
    >>> mini.leastsq()
    True
    >>> report_errors(params)
    ... #report
    >>> ci=conf_interval(mini)
    >>> report_ci(ci)
    ... #report
    
    Now with quantils for the sigmas and using the trace.
    
    >>> ci, trace=conf_interval(mini, sigmas=(0.25,0.5,0.75,0.999),trace=True)
    >>> fixed=trace['para1']['para1']
    >>> free=trace['para1']['not_para1']
    >>> prob=trace['para1']['prob']
    
    This makes it possible to plot the dependence between free and fixed.
    """

    if p_names == None:
        p_names = minimizer.params.keys()
    fit_params = [minimizer.params[p] for p in p_names]
    if prob_func is None or not hasattr(prob_func, '__call__'):
        prob_func = f_compare

    if trace:
        trace_dict = {}

    # copy the best fit values.

    org = copy_vals(minimizer.params)
    best_chi = minimizer.chisqr

    output = {}

    for para in fit_params:
        if trace:
            p_trace = ([], [])
        if verbose:
            print('Calculating CI for ' + para.name)

        restore_vals(org, minimizer.params)

        if para.stderr > 0:
            step = para.stderr
        else:
            step = max(para.value * 0.05, 0.01)

        para.vary = False
        start_val = para.value

        def calc_prob(val, offset=0., restore=False):
            '''Returns the probability for given Value.'''

            if restore:
                restore_vals(org, minimizer.params)
            para.value = val
            minimizer.prepare_fit(para)
            minimizer.leastsq()
            out = minimizer
            prob = prob_func(out.ndata, out.ndata - out.nfree,
                             out.chisqr, best_chi)

            if trace:
                p_trace[0].append([i.value for i in
                                  out.params.values()])
                p_trace[1].append(prob)

            return prob - offset

        def search_limits(direction):
            """
            Searchs for the limits. First it looks for a upper limit and
            then finds the sigma-limits with help of scipy root finder.
            """

            change = 1
            old_prob = 0
            i = 0
            limit = start_val

            # Find a upper limit,

            while change > 0.001 and old_prob < max(sigmas):
                i += 1
                limit += step * direction
                new_prob = calc_prob(limit)
                change = new_prob - old_prob
                old_prob = new_prob
                if i > maxiter:
                    break

            restore_vals(org, minimizer.params)

            # use brentq to find sigmas.

            ret = [(p, brentq(calc_prob, start_val, limit, args=p,
                   rtol=0.001)) for p in sigmas if p < old_prob]
            return ret

        upper_err = search_limits(1)
        restore_vals(org, minimizer.params)
        lower_err = search_limits(-1)

        if trace:
            trace_dict[para.name] = p_trace_to_dict(p_trace,
                    minimizer.params)

        para.vary = True
        output[para.name]= list(lower_err[::-1]) + [(0, start_val)] \
                            + list(upper_err)

    restore_vals(org, minimizer.params)
    if trace:
        return (output, trace_dict)
    return output


def conf_interval2d(minimizer, x_name, y_name, nx=10, ny=10, limits=None,
                    prob_func=None):
    r"""Calculates confidence regions for two fixed parameters.
    
    The method is explained in *conf_interval*: here we are fixing
    two parameters.
    
    Parameters
    ----------
    minimizer : minimizer
        The minimizer to use, should be already fitted via leastsq.
    x_name : string
        The name of the parameter which will be the x direction.
    y_name : string
        The name of the parameter which will be the y direction.
    nx, ny : ints, optional
        Number of points.
    limits : tuple: optional
        Should have the form ((x_upper, x_lower),(y_upper, y_lower)). If not
        given, the default is 5 stderrs in each direction.
        
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
    
    >>> from lmfit.printfuncs import *
    >>> mini=minimize(some_func, params)
    >>> mini.leastsq()
    True
    >>> x,y,gr=conf_interval2d('para1','para2')
    >>> plt.contour(x,y,gr)
 
    Other Parameters
    ----------------
    prob_func : ``None`` or callable 
        Function to calculate the probality from the opimized chi-square.
        Default (``None``) uses built-in f_compare (F test).
    """

    best_chi = minimizer.chisqr
    org = copy_vals(minimizer.params)

    if prob_func is None or not hasattr(prob_func, '__call__'):
        prob_func = f_compare


    x = minimizer.params[x_name]
    y = minimizer.params[y_name]

    if limits == None:
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
            restore_vals(org, minimizer.params)
        x.value = vals[0]
        y.value = vals[1]

        # minimizer.__prepared=False

        minimizer.prepare_fit([x, y])
        minimizer.leastsq()
        out = minimizer

        prob = prob_func(out.ndata, out.ndata - out.nfree, out.chisqr,
                         best_chi, Nfix=2.)
        return prob

    out = x_points, y_points, np.apply_along_axis(calc_prob, -1, grid)

    x.vary, y.vary = True, True
    restore_vals(org, minimizer.params)
    minimizer.chisqr = best_chi
    return out


