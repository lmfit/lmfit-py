"""Adaptive Memory Programming for Global Optimization (AMPGO).

added to lmfit by Renee Otten (2018)

based on the Python implementation of Andrea Gavana
(see: http://infinity77.net/global_optimization/)

Implementation details can be found in this paper:
    http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf
"""

import numpy as np
from scipy.optimize import minimize

SCIPY_LOCAL_SOLVERS = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP']


def ampgo(objfun, x0, args=(), local='L-BFGS-B', local_opts=None, bounds=None,
          maxfunevals=None, totaliter=20, maxiter=5, glbtol=1e-5, eps1=0.02,
          eps2=0.1, tabulistsize=5, tabustrategy='farthest', disp=False):
    """Find the global minimum of a multivariate function using the AMPGO
    (Adaptive Memory Programming for Global Optimization) algorithm.

    Parameters
    ----------
    objfun: callable
        Objective function to be minimized. The function must have the signature:
        objfun(params, *args, **kws)
    x0: numpy.ndarray
         Initial guesses for parameter values.
    args: tuple, optional
         Additional arguments passed to `objfun`.
    local: str, optional
        Name of the local minimization method. Valid options are:

        - `'L-BFGS-B'` (default)
        - `Nelder-Mead'`
        - `'Powell'`
        - `'TNC'`
        - `'SLSQP'`

    local_opts: dict, optional
        Options to pass to the local minimizer.
    bounds: sequence, optional
        List of tuples specifying the lower and upper bound for each
        independent variable [(`xl0`, `xu0`), (`xl1`, `xu1`), ...].
    maxfunevals: int, optional
        Maximum number of function evaluations. If None, the optimization will
        stop after `totaliter` number of iterations.
    totaliter: int, optional
        Maximum number of global iterations.
    maxiter: int, optional
        Maximum number of `Tabu Tunneling` iterations during each global
        iteration.
    glbtol: float, optional
        Tolerance whether or not to accept a solution after a tunneling phase.
    eps1: float, optional
        Constant used to define an aspiration value for the objective function
        during the Tunneling phase.
    eps2: float, optional
        Perturbation factor used to move away from the latest local minimum
        at the start of a Tunneling phase.
    tabulistsize: int, optional
        Size of the (circular) tabu search list.
    tabustrategy: str, optional
        Strategy to use when the size of the tabu list exceeds `tabulistsize`.
        It can be 'oldest' to drop the oldest point from the tabu list or
        'farthest' to drop the element farthest from the last local minimum
        found.
    disp: bool, optional
        Set to True to print convergence messages.

    Returns
    -------
    tuple:
        A tuple of 5 elements, in the following order:
         1. **best_x** (`array_like`): the estimated position of the global
             minimum.
         2. **best_f** (`float`): the value of `objfun` at the minimum.
         3. **evaluations** (`integer`): the number of function evaluations.
         4. **msg** (`string`): a message describes the cause of the termination.
         5. **tunnel_info** (`tuple`): a tuple containing the total number of
             Tunneling phases performed and the successful ones.

    Notes
    -----
    The detailed implementation of AMPGO is described in the paper
    "Adaptive Memory Programming for Constrained Global Optimization" located
    here:

    http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

    """
    if local not in SCIPY_LOCAL_SOLVERS:
        raise Exception('Invalid local solver selected: {}'.format(local))

    x0 = np.atleast_1d(x0)
    n = len(x0)

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    bounds = [b if b is not None else (None, None) for b in bounds]
    _bounds = [(-np.inf if l is None else l, np.inf if u is None else u)
               for l, u in bounds]
    low, up = np.array(_bounds).T

    if maxfunevals is None:
        maxfunevals = np.inf

    if tabulistsize < 1:
        raise Exception('Invalid tabulistsize specified: {:d}. It should be '
                        'an integer greater than zero.'.format(tabulistsize))
    if tabustrategy not in ['oldest', 'farthest']:
        raise Exception('Invalid tabustrategy specified: {:s}. It must be one '
                        'of "oldest" or "farthest".'.format(tabustrategy))

    tabulist = []
    best_f = np.inf
    best_x = x0

    global_iter = 0
    all_tunnel = success_tunnel = 0
    evaluations = 0

    local_tol = min(1e-8, glbtol)

    while 1:

        # minimization to find local minimum, either from initial values or
        # after a successful tunneling loop
        if disp:
            print('\n{0}\nStarting MINIMIZATION Phase {1:d}\n{0}'
                  .format('='*72, global_iter+1))

        options = {'maxiter': max(1, maxfunevals), 'disp': disp}
        if local_opts is not None:
            options.update(local_opts)
        res = minimize(objfun, x0, args=args, method=local, bounds=bounds,
                       tol=local_tol, options=options)
        xf, yf, num_fun = res['x'], res['fun'], res['nfev']
        if isinstance(yf, np.ndarray):
            yf = yf[0]

        maxfunevals -= num_fun
        evaluations += num_fun

        if yf < best_f:
            best_f = yf
            best_x = xf

        if disp:
            print('\n\n ==> Reached local minimum: {:.5g}\n'.format(yf))

        if maxfunevals <= 0:
            if disp:
                print('='*72)
            return (best_x, best_f, evaluations,
                    'Maximum number of function evaluations exceeded',
                    (all_tunnel, success_tunnel))

        # if needed, drop a value from the tabu tunneling list and add the
        # current solution
        tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
        tabulist.append(xf)

        i = improve = 0

        while i < maxiter and improve == 0:

            if disp:
                print('{0}\nStarting TUNNELING Phase ({1:d}-{2:d})\n{0}'
                      .format('='*72, global_iter+1, i+1))

            all_tunnel += 1

            # generate a new starting point away from the current solution
            r = np.random.uniform(-1.0, 1.0, size=(n, ))
            beta = eps2*np.linalg.norm(xf) / np.linalg.norm(r)

            if np.abs(beta) < 1e-8:
                beta = eps2

            x0 = xf + beta*r

            # make sure that the new starting point is within bounds
            x0 = np.where(x0 < low, low, x0)
            x0 = np.where(x0 > up, up, x0)

            # aspired value of the objective function for the tunneling loop
            aspiration = best_f - eps1*(1.0 + np.abs(best_f))

            tunnel_args = tuple([objfun, aspiration, tabulist] + list(args))

            options = {'maxiter': max(1, maxfunevals), 'disp': disp}
            if local_opts is not None:
                options.update(local_opts)

            res = minimize(tunnel, x0, args=tunnel_args, method=local,
                           bounds=bounds, tol=local_tol, options=options)
            xf, yf, num_fun = res['x'], res['fun'], res['nfev']
            if isinstance(yf, np.ndarray):
                yf = yf[0]

            maxfunevals -= num_fun
            evaluations += num_fun

            yf = inverse_tunnel(xf, yf, aspiration, tabulist)

            if yf <= best_f + glbtol:
                oldf = best_f
                best_f = yf
                best_x = xf
                improve = 1
                success_tunnel += 1

                if disp:
                    print('\n\n ==> Successful tunnelling phase. Reached new '
                          'local minimum: {:.5g} < {:.5g}\n'.format(yf, oldf))

            i += 1

            if maxfunevals <= 0:
                return (best_x, best_f, evaluations,
                        'Maximum number of function evaluations exceeded',
                        (all_tunnel, success_tunnel))

            tabulist = drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy)
            tabulist.append(xf)

        if disp:
            print('='*72)

        global_iter += 1
        x0 = xf.copy()

        if global_iter >= totaliter:
            return (best_x, best_f, evaluations,
                    'Maximum number of global iterations exceeded',
                    (all_tunnel, success_tunnel))


def drop_tabu_points(xf, tabulist, tabulistsize, tabustrategy):
    """Drop a point from the tabu search list."""
    if len(tabulist) < tabulistsize:
        return tabulist

    if tabustrategy == 'oldest':
        tabulist.pop(0)
    else:
        distance = np.sqrt(np.sum((tabulist - xf)**2, axis=1))
        index = np.argmax(distance)
        tabulist.pop(index)

    return tabulist


def tunnel(x0, *args):
    """Tunneling objective function.

    This function has a global minimum of zero at any feasible point where
    `f(x) = aspiration`, and minimizing this expression tends to move away
    from all points in `tabulist`.

    """
    objfun, aspiration, tabulist = args[0:3]

    fun_args = ()
    if len(args) > 3:
        fun_args = tuple(args[3:])

    numerator = (objfun(x0, *fun_args) - aspiration)**2
    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator * np.sqrt(np.sum((x0 - tabu)**2))

    ytf = numerator/denominator

    return ytf


def inverse_tunnel(xtf, ytf, aspiration, tabulist):
    """Calculate the function value after a tunneling phase step."""
    denominator = 1.0

    for tabu in tabulist:
        denominator = denominator * np.sqrt(np.sum((xtf - tabu)**2))

    numerator = ytf*denominator
    yf = aspiration + np.sqrt(numerator)

    return yf
