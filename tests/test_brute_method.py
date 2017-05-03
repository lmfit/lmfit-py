from __future__ import print_function
import pickle
import numpy as np
from numpy.testing import (assert_, decorators, assert_raises,
                           assert_almost_equal, assert_equal,
                           assert_allclose)
from scipy import optimize
import lmfit


# use example problem described int he scipy documentation:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

# setup for scipy-brute optimization #
params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)

def f1(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

def f2(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

def f3(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params
    return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

def f(z, *params):
    return f1(z, *params) + f2(z, *params) + f3(z, *params)
# setup for scipy-brute optimization #

# setup for lmfit-brute optimization #
params_lmfit = lmfit.Parameters()
params_lmfit.add_many(
        ('a', 2, False, None, None, None),
        ('b', 3, False, None, None, None),
        ('c', 7, False, None, None, None),
        ('d', 8, False, None, None, None),
        ('e', 9, False, None, None, None),
        ('f', 10, False, None, None, None),
        ('g', 44, False, None, None, None),
        ('h', -1, False, None, None, None),
        ('i', 2, False, None, None, None),
        ('j', 26, False, None, None, None),
        ('k', 1, False, None, None, None),
        ('l', -2, False, None, None, None),
        ('scale', 0.5, False, None, None, None),
        ('x', -4.0, True, -4.0, 4.0, None, None),
        ('y', -2.0, True, -2.0, 2.0, None, None),
    )

def f1_lmfit(p):
    par = p.valuesdict()
    return (par['a'] * par['x']**2 + par['b'] * par['x'] * par['y'] +
            par['c'] * par['y']**2 + par['d']*par['x'] + par['e']*par['y'] +
            par['f'])

def f2_lmfit(p):
    par = p.valuesdict()
    return (-1.0*par['g']*np.exp(-((par['x']-par['h'])**2 +
            (par['y']-par['i'])**2) / par['scale']))

def f3_lmfit(p):
    par = p.valuesdict()
    return (-1.0*par['j']*np.exp(-((par['x']-par['k'])**2 +
            (par['y']-par['l'])**2) / par['scale']))

def f_lmfit(params_lmfit):
    return f1_lmfit(params_lmfit) + f2_lmfit(params_lmfit) + f3_lmfit(params_lmfit)
# setup for lmfit-brute optimization ###


def test_brute_lmfit_vs_scipy():
    # The tests below are to make sure that the implementation of the brute
    # method in lmfit gives identical results to scipy.optimize.brute, when
    # using finite bounds for all varying parameters.

    # TEST 1: using bounds, with (default) Ns=20 and no stepsize specified
    assert(not params_lmfit['x'].brute_step)  # brute_step for x == None
    assert(not params_lmfit['y'].brute_step)  # brute_step for y == None

    rranges = ((-4, 4), (-2, 2))
    resbrute = optimize.brute(f, rranges, args=params, full_output=True, Ns=20,
                              finish=None)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=20)

    assert_equal(resbrute[2], resbrute_lmfit.brute_grid, verbose=True) # grid identical
    assert_equal(resbrute[3], resbrute_lmfit.brute_Jout, verbose=True) # function values on grid identical
    assert_equal(resbrute[0][0], resbrute_lmfit.brute_x0[0], verbose=True) # best fit x value identical
    assert_equal(resbrute[0][0], resbrute_lmfit.params['x'].value, verbose=True) # best fit x value stored correctly
    assert_equal(resbrute[0][1], resbrute_lmfit.brute_x0[1], verbose=True) # best fit y value identical
    assert_equal(resbrute[0][1], resbrute_lmfit.params['y'].value, verbose=True) # best fit y value stored correctly
    assert_equal(resbrute[1], resbrute_lmfit.brute_fval, verbose=True) # best fit function value identical
    assert_equal(resbrute[1], resbrute_lmfit.chisqr, verbose=True) # best fit function value stored correctly

    # TEST 2: using bounds, setting Ns=40 and no stepsize specified
    assert(not params_lmfit['x'].brute_step)  # brute_step for x == None
    assert(not params_lmfit['y'].brute_step)  # brute_step for y == None

    rranges = ((-4, 4), (-2, 2))
    resbrute = optimize.brute(f, rranges, args=params, full_output=True, Ns=40,
                              finish=None)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=40)

    assert_equal(resbrute[2], resbrute_lmfit.brute_grid, verbose=True) # grid identical
    assert_equal(resbrute[3], resbrute_lmfit.brute_Jout, verbose=True) # function values on grid identical
    assert_equal(resbrute[0][0], resbrute_lmfit.params['x'].value, verbose=True) # best fit x value identical
    assert_equal(resbrute[0][1], resbrute_lmfit.params['y'].value, verbose=True) # best fit y value identical
    assert_equal(resbrute[1], resbrute_lmfit.chisqr, verbose=True) # best fit function value identical

    # TEST 3: using bounds and specifing stepsize for both parameters
    params_lmfit['x'].set(brute_step=0.25)
    params_lmfit['y'].set(brute_step=0.25)
    assert_equal(params_lmfit['x'].brute_step, 0.25 ,verbose=True)
    assert_equal(params_lmfit['y'].brute_step, 0.25 ,verbose=True)

    rranges = (slice(-4, 4, 0.25), slice(-2, 2, 0.25))
    resbrute = optimize.brute(f, rranges, args=params, full_output=True, Ns=20,
                              finish=None)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute')

    assert_equal(resbrute[2], resbrute_lmfit.brute_grid, verbose=True) # grid identical
    assert_equal(resbrute[3], resbrute_lmfit.brute_Jout, verbose=True) # function values on grid identical
    assert_equal(resbrute[0][0], resbrute_lmfit.params['x'].value, verbose=True) # best fit x value identical
    assert_equal(resbrute[0][1], resbrute_lmfit.params['y'].value, verbose=True) # best fit y value identical
    assert_equal(resbrute[1], resbrute_lmfit.chisqr, verbose=True) # best fit function value identical

    # TEST 4: using bounds, Ns=10, adn specifing stepsize for parameter 'x'
    params_lmfit['x'].set(brute_step=0.15)
    params_lmfit['y'].set(brute_step=0) # brute_step for y == None
    assert_equal(params_lmfit['x'].brute_step, 0.15 ,verbose=True)
    assert(not params_lmfit['y'].brute_step)

    rranges = (slice(-4, 4, 0.15), (-2, 2))
    resbrute = optimize.brute(f, rranges, args=params, full_output=True, Ns=10,
                              finish=None)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=10, keep='all')

    assert_equal(resbrute[2], resbrute_lmfit.brute_grid, verbose=True) # grid identical
    assert_equal(resbrute[3], resbrute_lmfit.brute_Jout, verbose=True) # function values on grid identical
    assert_equal(resbrute[0][0], resbrute_lmfit.params['x'].value, verbose=True) # best fit x value identical
    assert_equal(resbrute[0][1], resbrute_lmfit.params['y'].value, verbose=True) # best fit y value identical
    assert_equal(resbrute[1], resbrute_lmfit.chisqr, verbose=True) # best fit function value identical


def test_brute():
    # The tests below are to make sure that the implementation of the brute
    # method in lmfit works as intended.

    # restore original settings for paramers 'x' and 'y'
    params_lmfit.add_many(
        ('x', -4.0, True, -4.0, 4.0, None, None),
        ('y', -2.0, True, -2.0, 2.0, None, None))

    # TEST 1: only upper bound and brute_step specified, using default Ns=20
    Ns = 20
    params_lmfit['x'].set(min=-np.inf)
    params_lmfit['x'].set(brute_step=0.25)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute')
    grid_x_expected = np.linspace(params_lmfit['x'].max - Ns*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].max, Ns, False)
    grid_x = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][0])
    assert_almost_equal(grid_x_expected, grid_x, verbose=True)
    grid_y = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][1])
    grid_y_expected = np.linspace(params_lmfit['y'].min, params_lmfit['y'].max, Ns)
    assert_almost_equal(grid_y_expected, grid_y, verbose=True)

    # TEST 2: only lower bound and brute_step specified, using Ns=15
    Ns = 15
    params_lmfit['y'].set(max=np.inf)
    params_lmfit['y'].set(brute_step=0.1)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=15)
    grid_x_expected = np.linspace(params_lmfit['x'].max - Ns*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].max, Ns, False)
    grid_x = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][0])
    assert_almost_equal(grid_x_expected, grid_x, verbose=True)
    grid_y = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][1])
    grid_y_expected = np.linspace(params_lmfit['y'].min, params_lmfit['y'].min + Ns*params_lmfit['y'].brute_step, Ns, False)
    assert_almost_equal(grid_y_expected, grid_y, verbose=True)

    # TEST 3: only value and brute_step specified, using Ns=15
    Ns = 15
    params_lmfit['x'].set(max=np.inf)
    params_lmfit['x'].set(min=-np.inf)
    params_lmfit['x'].set(brute_step=0.1)
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=15)
    grid_x_expected = np.linspace(params_lmfit['x'].value - (Ns//2)*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].value + (Ns//2)*params_lmfit['x'].brute_step, Ns)
    grid_x = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][0])
    assert_almost_equal(grid_x_expected, grid_x, verbose=True)
    grid_y = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][1])
    grid_y_expected = np.linspace(params_lmfit['y'].min, params_lmfit['y'].min + Ns*params_lmfit['y'].brute_step, Ns, False)
    assert_almost_equal(grid_y_expected, grid_y, verbose=True)

    # TEST 3: only value and brute_step specified, using Ns=15
    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute', Ns=15)
    grid_x_expected = np.linspace(params_lmfit['x'].value - (Ns//2)*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].value + (Ns//2)*params_lmfit['x'].brute_step, Ns)
    grid_x = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][0])
    assert_almost_equal(grid_x_expected, grid_x, verbose=True)
    grid_y = np.unique([par.ravel() for par in resbrute_lmfit.brute_grid][1])
    grid_y_expected = np.linspace(params_lmfit['y'].min, params_lmfit['y'].min + Ns*params_lmfit['y'].brute_step, Ns, False)
    assert_almost_equal(grid_y_expected, grid_y, verbose=True)

    # TEST 4: check for correct functioning of keep argument and candidates attribute
    params_lmfit.add_many( # restore original settings for paramers 'x' and 'y'
        ('x', -4.0, True, -4.0, 4.0, None, None),
        ('y', -2.0, True, -2.0, 2.0, None, None))

    fitter = lmfit.Minimizer(f_lmfit, params_lmfit)
    resbrute_lmfit = fitter.minimize(method='brute')
    assert(len(resbrute_lmfit.candidates) == 50) # default number of stored candidates

    resbrute_lmfit = fitter.minimize(method='brute', keep=10)
    assert(len(resbrute_lmfit.candidates) == 10)

    assert(isinstance(resbrute_lmfit.candidates[0].params, lmfit.Parameters))

    # TEST 5: make sure the MinimizerResult can be pickle'd
    pkl = pickle.dumps(resbrute_lmfit)

test_brute_lmfit_vs_scipy()
test_brute()
