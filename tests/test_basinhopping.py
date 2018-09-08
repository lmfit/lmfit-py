import numpy as np
from numpy.testing import assert_allclose

from scipy.optimize import basinhopping
from scipy.version import version as scipy_version

import lmfit


def test_basinhopping():
    """Test basinhopping in lmfit versus scipy."""

    # SciPy
    def func(x):
        return np.cos(14.5*x - 0.3) + (x+0.2) * x

    minimizer_kwargs = {'method': 'L-BFGS-B'}
    x0 = [1.]

    # FIXME - remove after requirement for scipy >= 0.19
    major, minor, micro = np.array(scipy_version.split('.'), dtype='int')
    if major < 1 and minor < 19:
        ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs)
    else:
        ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, seed=7)

    # lmfit
    def residual(params):
        x = params['x'].value
        return np.cos(14.5*x - 0.3) + (x+0.2) * x

    pars = lmfit.Parameters()
    pars.add_many(('x', 1.))
    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    mini = lmfit.Minimizer(residual, pars)
    out = mini.minimize(method='basinhopping', **kws)

    assert_allclose(out.residual, ret.fun)
    assert_allclose(out.params['x'].value, ret.x)


def test_basinhopping_2d():
    """Test basinhopping in lmfit versus scipy."""

    # SciPy
    def func2d(x):
        return np.cos(14.5*x[0] - 0.3) + (x[1]+0.2) * x[1] + (x[0]+0.2) * x[0]

    minimizer_kwargs = {'method': 'L-BFGS-B'}
    x0 = [1.0, 1.0]

    # FIXME - remove after requirement for scipy >= 0.19
    major, minor, micro = np.array(scipy_version.split('.'), dtype='int')
    if major < 1 and minor < 19:
        ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs)
    else:
        ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs, seed=7)

    # lmfit
    def residual_2d(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.cos(14.5*x0 - 0.3) + (x1+0.2) * x1 + (x0+0.2) * x0

    pars = lmfit.Parameters()
    pars.add_many(('x0', 1.), ('x1', 1.))

    mini = lmfit.Minimizer(residual_2d, pars)
    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    out = mini.minimize(method='basinhopping', **kws)

    assert_allclose(out.residual, ret.fun)
    assert_allclose(out.params['x0'].value, ret.x[0])
    assert_allclose(out.params['x1'].value, ret.x[1], rtol=1e-5)


def test_basinhopping_Alpine02():
    """Test basinhopping on Alpine02 function."""

    global_optimum = [7.91705268, 4.81584232]
    fglob = -6.12950

    # SciPy
    def Alpine02(x):
        x0 = x[0]
        x1 = x[1]
        return np.prod(np.sqrt(x0) * np.sin(x0)) * np.prod(np.sqrt(x1) *
                                                           np.sin(x1))

    def basinhopping_accept(f_new, f_old, x_new, x_old):
        """Does the new candidate vector lie inbetween the bounds?

        Returns
        -------
        accept_test : bool
            The candidate vector lies inbetween the bounds
        """
        if np.any(x_new < np.array([0.0, 0.0])):
            return False
        if np.any(x_new > np.array([10.0, 10.0])):
            return False
        return True

    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': [(0.0, 10.0),
                                                         (0.0, 10.0)]}
    x0 = [1.0, 1.0]

    # FIXME - remove after requirement for scipy >= 0.19
    major, minor, micro = np.array(scipy_version.split('.'), dtype='int')
    if major < 1 and minor < 19:
        ret = basinhopping(Alpine02, x0, minimizer_kwargs=minimizer_kwargs,
                           accept_test=basinhopping_accept)
    else:
        ret = basinhopping(Alpine02, x0, minimizer_kwargs=minimizer_kwargs,
                           accept_test=basinhopping_accept, seed=7)

    # lmfit
    def residual_Alpine02(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.prod(np.sqrt(x0) * np.sin(x0)) * np.prod(np.sqrt(x1) *
                                                           np.sin(x1))

    pars = lmfit.Parameters()
    pars.add_many(('x0', 1., True, 0.0, 10.0),
                  ('x1', 1., True, 0.0, 10.0))

    mini = lmfit.Minimizer(residual_Alpine02, pars)
    kws = {'minimizer_kwargs': {'method': 'L-BFGS-B'}, 'seed': 7}
    out = mini.minimize(method='basinhopping', **kws)
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum))
    assert_allclose(max(out_x), max(global_optimum))


if __name__ == '__main__':
    test_basinhopping()
    test_basinhopping_2d()
    test_basinhopping_Alpine02()
