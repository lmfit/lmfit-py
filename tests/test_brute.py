"""Tests for the brute force algorithm (aka 'grid-search').

Use example problem described in the SciPy documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

"""
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import optimize

import lmfit


def func_scipy(z, *params):
    x, y = z
    a, b, c, d, e, f, g, h, i, j, k, l, scale = params

    f1 = a * x**2 + b*x*y + c * y**2 + d*x + e*y + f
    f2 = -g*np.exp(-((x-h)**2 + (y-i)**2) / scale)
    f3 = -j*np.exp(-((x-k)**2 + (y-l)**2) / scale)

    return f1 + f2 + f3


params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)


def func_lmfit(p):
    par = p.valuesdict()
    f1 = (par['a'] * par['x']**2 + par['b']*par['x']*par['y'] +
          par['c'] * par['y']**2 + par['d']*par['x'] + par['e']*par['y'] +
          par['f'])
    f2 = (-1.0*par['g']*np.exp(-((par['x']-par['h'])**2 +
                                 (par['y']-par['i'])**2) / par['scale']))
    f3 = (-1.0*par['j']*np.exp(-((par['x']-par['k'])**2 +
                                 (par['y']-par['l'])**2) / par['scale']))

    return f1 + f2 + f3


@pytest.fixture
def params_lmfit():
    """Return lmfit.Parameters class with initial values and bounds."""
    params = lmfit.Parameters()
    params.add_many(
        ('a', 2, False), ('b', 3, False), ('c', 7, False), ('d', 8, False),
        ('e', 9, False), ('f', 10, False), ('g', 44, False), ('h', -1, False),
        ('i', 2, False), ('j', 26, False), ('k', 1, False), ('l', -2, False),
        ('scale', 0.5, False), ('x', -4.0, True, -4.0, 4.0, None, None),
        ('y', -2.0, True, -2.0, 2.0, None, None))
    return params


def test_brute_lmfit_vs_scipy_default(params_lmfit):
    """TEST 1: using finite bounds with Ns=20, keep=50 and brute_step=None."""
    assert params_lmfit['x'].brute_step is None
    assert params_lmfit['y'].brute_step is None

    rranges = ((-4, 4), (-2, 2))
    ret = optimize.brute(func_scipy, rranges, args=params, full_output=True,
                         Ns=20, finish=None)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=20)

    assert out.method == 'brute'
    assert_equal(out.nfev, 20**len(out.var_names))  # Ns * nmb varying params
    assert_equal(len(out.candidates), 50)  # top-50 candidates are stored

    assert_equal(ret[2], out.brute_grid)  # grid identical
    assert_equal(ret[3], out.brute_Jout)  # function values on grid identical

    # best-fit values identical / stored correctly in MinimizerResult
    assert_equal(ret[0][0], out.brute_x0[0])
    assert_equal(ret[0][0], out.params['x'].value)

    assert_equal(ret[0][1], out.brute_x0[1])
    assert_equal(ret[0][1], out.params['y'].value)

    assert_equal(ret[1], out.brute_fval)
    assert_equal(ret[1], out.residual)


def test_brute_lmfit_vs_scipy_Ns(params_lmfit):
    """TEST 2: using finite bounds, with Ns=40 and brute_step=None."""
    rranges = ((-4, 4), (-2, 2))
    ret = optimize.brute(func_scipy, rranges, args=params, full_output=True,
                         Ns=40, finish=None)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=40)

    assert_equal(ret[2], out.brute_grid)  # grid identical
    assert_equal(ret[3], out.brute_Jout)  # function values on grid identical
    assert_equal(out.nfev, 40**len(out.var_names))  # Ns * nmb varying params

    # best-fit values and function value identical
    assert_equal(ret[0][0], out.brute_x0[0])
    assert_equal(ret[0][1], out.brute_x0[1])
    assert_equal(ret[1], out.brute_fval)


def test_brute_lmfit_vs_scipy_stepsize(params_lmfit):
    """TEST 3: using finite bounds and brute_step for both parameters."""
    # set brute_step for parameters and assert whether that worked correctly
    params_lmfit['x'].set(brute_step=0.25)
    params_lmfit['y'].set(brute_step=0.25)
    assert_equal(params_lmfit['x'].brute_step, 0.25)
    assert_equal(params_lmfit['y'].brute_step, 0.25)

    rranges = (slice(-4, 4, 0.25), slice(-2, 2, 0.25))
    ret = optimize.brute(func_scipy, rranges, args=params, full_output=True,
                         Ns=20, finish=None)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute')

    assert_equal(ret[2], out.brute_grid)  # grid identical
    assert_equal(ret[3], out.brute_Jout)  # function values on grid identical

    # best-fit values and function value identical
    assert_equal(ret[0][0], out.brute_x0[0])
    assert_equal(ret[0][1], out.brute_x0[1])
    assert_equal(ret[1], out.brute_fval)

    points_x = np.arange(rranges[0].start, rranges[0].stop, rranges[0].step).size
    points_y = np.arange(rranges[1].start, rranges[1].stop, rranges[1].step).size
    nmb_evals = points_x * points_y
    assert_equal(out.nfev, nmb_evals)


def test_brute_lmfit_vs_scipy_Ns_stepsize(params_lmfit):
    """TEST 4: using finite bounds, using Ns, brute_step for 'x'."""
    # set brute_step for x to 0.15 and reset to None for y and assert result
    params_lmfit['x'].set(brute_step=0.15)
    assert_equal(params_lmfit['x'].brute_step, 0.15)

    rranges = (slice(-4, 4, 0.15), (-2, 2))
    ret = optimize.brute(func_scipy, rranges, args=params, full_output=True,
                         Ns=10, finish=None)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=10)

    assert_equal(ret[2], out.brute_grid)  # grid identical
    assert_equal(ret[3], out.brute_Jout)  # function values on grid identical

    points_x = np.arange(rranges[0].start, rranges[0].stop, rranges[0].step).size
    points_y = 10
    nmb_evals = points_x * points_y
    assert_equal(out.nfev, nmb_evals)

    # best-fit values and function value identical
    assert_equal(ret[0][0], out.brute_x0[0])
    assert_equal(ret[0][1], out.brute_x0[1])
    assert_equal(ret[1], out.brute_fval)


def test_brute_upper_bounds_and_brute_step(params_lmfit):
    """TEST 5: using finite upper bounds, Ns=20, and brute_step specified."""
    Ns = 20
    params_lmfit['x'].set(min=-np.inf)
    params_lmfit['x'].set(brute_step=0.25)

    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=Ns)

    assert_equal(out.params['x'].min, -np.inf)
    assert_equal(out.params['x'].brute_step, 0.25)

    grid_x_expected = np.linspace(params_lmfit['x'].max -
                                  Ns*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].max, Ns, False)
    grid_x = np.unique([par.ravel() for par in out.brute_grid][0])
    assert_allclose(grid_x, grid_x_expected)

    grid_y_expected = np.linspace(params_lmfit['y'].min,
                                  params_lmfit['y'].max, Ns)
    grid_y = np.unique([par.ravel() for par in out.brute_grid][1])
    assert_allclose(grid_y, grid_y_expected)


def test_brute_lower_bounds_and_brute_step(params_lmfit):
    """TEST 6: using finite lower bounds, Ns=15, and brute_step specified."""
    Ns = 15
    params_lmfit['y'].set(max=np.inf)
    params_lmfit['y'].set(brute_step=0.1)

    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=Ns)

    grid_x_expected = np.linspace(params_lmfit['x'].min,
                                  params_lmfit['x'].max, Ns)
    grid_x = np.unique([par.ravel() for par in out.brute_grid][0])
    assert_allclose(grid_x, grid_x_expected)

    grid_y_expected = np.linspace(params_lmfit['y'].min,
                                  params_lmfit['y'].min +
                                  Ns*params_lmfit['y'].brute_step, Ns, False)
    grid_y = np.unique([par.ravel() for par in out.brute_grid][1])
    assert_allclose(grid_y, grid_y_expected)


def test_brute_no_bounds_with_brute_step(params_lmfit):
    """TEST 7: using no bounds, but brute_step specified (Ns=15)."""
    Ns = 15
    params_lmfit['x'].set(min=-np.inf, max=np.inf, brute_step=0.1)
    params_lmfit['y'].set(min=-np.inf, max=np.inf, brute_step=0.2)

    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute', Ns=15)

    grid_x_expected = np.linspace(params_lmfit['x'].value -
                                  (Ns//2)*params_lmfit['x'].brute_step,
                                  params_lmfit['x'].value +
                                  (Ns//2)*params_lmfit['x'].brute_step, Ns)
    grid_x = np.unique([par.ravel() for par in out.brute_grid][0])
    assert_allclose(grid_x, grid_x_expected)

    grid_y_expected = np.linspace(params_lmfit['y'].value -
                                  (Ns//2)*params_lmfit['y'].brute_step,
                                  params_lmfit['y'].value +
                                  (Ns//2)*params_lmfit['y'].brute_step, Ns)
    grid_y = np.unique([par.ravel() for par in out.brute_grid][1])
    assert_allclose(grid_y, grid_y_expected)


def test_brute_no_bounds_no_brute_step(params_lmfit):
    """TEST 8: insufficient information provided."""
    params_lmfit['x'].set(min=-np.inf, max=np.inf)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)

    msg = r'Not enough information provided for the brute force method.'
    with pytest.raises(ValueError, match=msg):
        mini.minimize(method='brute')


def test_brute_one_parameter(params_lmfit):
    """TEST 9: only one varying parameter."""
    params_lmfit['x'].set(vary=False)
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute')
    assert out.candidates[0].score <= out.candidates[1].score
    assert isinstance(out.candidates[0], lmfit.minimizer.Candidate)
    assert isinstance(out.candidates[0].params, lmfit.Parameters)
    assert isinstance(out.candidates[0].score, float)


def test_brute_keep(params_lmfit, capsys):
    """TEST 10: using 'keep' argument and check candidates attribute."""
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute')
    assert_equal(len(out.candidates), 50)  # default

    out_keep_all = mini.minimize(method='brute', keep='all')
    assert_equal(len(out_keep_all.candidates),
                 len(out_keep_all.brute_Jout.ravel()))

    out_keep10 = mini.minimize(method='brute', keep=10)
    assert_equal(len(out_keep10.candidates), 10)

    assert isinstance(out.candidates[0], lmfit.minimizer.Candidate)
    assert isinstance(out.candidates[0].params, lmfit.Parameters)
    assert isinstance(out.candidates[0].score, float)

    with pytest.raises(ValueError, match=r"'candidate_nmb' should be between"):
        out_keep10.show_candidates(25)

    with pytest.raises(ValueError, match=r"'candidate_nmb' should be between"):
        out_keep10.show_candidates(0)

    out_keep10.show_candidates(5)
    captured = capsys.readouterr()
    assert 'Candidate #5' in captured.out

    # for coverage and to make sure the 'all' argument works; no assert...
    out_keep10.show_candidates('all')


def test_brute_pickle(params_lmfit):
    """TEST 11: make sure the MinimizerResult can be pickle'd."""
    mini = lmfit.Minimizer(func_lmfit, params_lmfit)
    out = mini.minimize(method='brute')
    pickle.dumps(out)


def test_nfev_workers(params_lmfit):
    """TEST 12: make sure the nfev is correct for workers != 1."""
    mini = lmfit.Minimizer(func_lmfit, params_lmfit, workers=-1)
    out = mini.minimize(method='brute')
    assert_equal(out.nfev, 20**len(out.var_names))
