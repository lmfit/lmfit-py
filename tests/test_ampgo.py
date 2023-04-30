"""Tests for the AMPGO global minimization algorithm."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import __version__ as scipy_version

import lmfit
from lmfit._ampgo import ampgo, tunnel

# correct result for Alpine02 function
global_optimum = [7.91705268, 4.81584232]
fglob = -6.12950


@pytest.mark.parametrize("tabustrategy", ['farthest', 'oldest'])
def test_ampgo_Alpine02(minimizer_Alpine02, tabustrategy):
    """Test AMPGO algorithm on Alpine02 function."""
    kws = {'tabustrategy': tabustrategy}
    out = minimizer_Alpine02.minimize(method='ampgo', **kws)
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert 'global' in out.ampgo_msg


def test_ampgo_bounds(minimizer_Alpine02):
    """Test AMPGO algorithm with bounds."""
    # change boundaries of parameters
    pars_bounds = lmfit.Parameters()
    pars_bounds.add_many(('x0', 1., True, 5.0, 15.0),
                         ('x1', 1., True, 2.5, 7.5))

    out = minimizer_Alpine02.minimize(params=pars_bounds, method='ampgo')
    assert 5.0 <= out.params['x0'].value <= 15.0
    assert 2.5 <= out.params['x1'].value <= 7.5


def test_ampgo_disp_true(minimizer_Alpine02, capsys):
    """Test AMPGO algorithm with disp is True."""
    # disp to False for L-BFGS-B to avoid too much output...
    kws = {'disp': True, 'local_opts': {'disp': False}}
    minimizer_Alpine02.minimize(method='ampgo', **kws)
    captured = capsys.readouterr()
    assert "Starting MINIMIZATION Phase" in captured.out


def test_ampgo_maxfunevals(minimizer_Alpine02):
    """Test AMPGO algorithm with maxfunevals."""
    # disp to False for L-BFGS-B to avoid too much output...
    kws = {'maxfunevals': 5, 'disp': True, 'local_opts': {'disp': False}}
    out = minimizer_Alpine02.minimize(method='ampgo', **kws)

    assert out.ampgo_msg == 'Maximum number of function evaluations exceeded'


def test_ampgo_local_solver(minimizer_Alpine02):
    """Test AMPGO algorithm with local solver."""
    kws = {'local': 'Nelder-Mead'}

    # bounds in Nelder-Mead are supported since SciPy v1.7.0
    # FIXME: clean this up after we require SciPy >= 1.7.0
    if int(scipy_version.split('.')[1]) < 7:
        msg = r'Method Nelder-Mead cannot handle constraints nor bounds'
        with pytest.warns(RuntimeWarning, match=msg):
            out = minimizer_Alpine02.minimize(method='ampgo', **kws)
    else:
        out = minimizer_Alpine02.minimize(method='ampgo', **kws)

    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert 'ampgo' and 'Nelder-Mead' in out.method
    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert 'global' in out.ampgo_msg


def test_ampgo_invalid_local_solver(minimizer_Alpine02):
    """Test AMPGO algorithm with invalid local solvers."""
    kws = {'local': 'leastsq'}
    with pytest.raises(Exception, match=r'Invalid local solver selected'):
        minimizer_Alpine02.minimize(method='ampgo', **kws)


def test_ampgo_invalid_tabulistsize(minimizer_Alpine02):
    """Test AMPGO algorithm with invalid tabulistsize."""
    kws = {'tabulistsize': 0}
    with pytest.raises(Exception, match=r'Invalid tabulistsize specified'):
        minimizer_Alpine02.minimize(method='ampgo', **kws)


def test_ampgo_invalid_tabustrategy(minimizer_Alpine02):
    """Test AMPGO algorithm with invalid tabustrategy."""
    kws = {'tabustrategy': 'unknown'}
    with pytest.raises(Exception, match=r'Invalid tabustrategy specified'):
        minimizer_Alpine02.minimize(method='ampgo', **kws)


def test_ampgo_local_opts(minimizer_Alpine02):
    """Test AMPGO algorithm, pass local_opts to solver."""
    # use local_opts to pass maxfun to the local optimizer: providing a string
    # whereas an integer is required, this should throw an error.
    kws = {'local_opts': {'maxfun': 'string'}}
    with pytest.raises(TypeError):
        minimizer_Alpine02.minimize(method='ampgo', **kws)

    # for coverage: make sure that both occurrences are reached
    kws = {'local_opts': {'maxfun': 10}, 'maxfunevals': 50}
    minimizer_Alpine02.minimize(method='ampgo', **kws)


def test_ampgo_incorrect_length_for_bounds():
    """Test for ValueError when bounds are given for only some parameters."""
    def func(x):
        return x[0]**2 + 10.0*x[1]

    msg = r'length of x0 != length of bounds'
    with pytest.raises(ValueError, match=msg):
        ampgo(func, x0=[0, 1], bounds=[(-10, 10)])


def test_ampgo_tunnel_more_than_three_arguments():
    """Test AMPGO tunnel function with more than three arguments."""
    def func(x, val):
        return x[0]**2 + val*x[1]

    args = [func, 0.1, np.array([1, 2]), 5.0]
    out = tunnel(np.array([10, 5]), *args)
    assert_allclose(out, 185.386275588)
