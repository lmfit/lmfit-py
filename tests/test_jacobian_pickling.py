"""Tests for (un)pickling the Minimizer class when providing a Jacobian function."""
from multiprocessing import get_context
from pickle import dumps, loads

import numpy as np

from lmfit import Minimizer, Parameters, minimize


def func(pars, x, data=None):
    a, b, c = pars['a'], pars['b'], pars['c']
    model = a * np.exp(-b*x) + c
    if data is None:
        return model
    return model - data


def dfunc(pars, x, data=None):
    a, b = pars['a'], pars['b']
    v = np.exp(-b*x)
    return np.array([v, -a*x*v, np.ones(len(x))])


def jacfunc(pars, x, data=None):
    a, b = pars['a'], pars['b']
    v = np.exp(-b*x)
    jac = np.ones((len(x), 3), dtype=np.float64)
    jac[:, 0] = v
    jac[:, 1] = -a * x * v
    return jac


def f(var, x):
    return var[0] * np.exp(-var[1]*x) + var[2]


def wrapper(args):
    (
        method,
        x,
        data,
    ) = args
    params = Parameters()
    params.add('a', value=10)
    params.add('b', value=10)
    params.add('c', value=10)

    return minimize(
        func,
        params,
        method=method,
        args=(x,),
        kws={'data': data},
        **(
            {'Dfun': dfunc, 'col_deriv': 1}
            if method == 'leastsq' else
            {'jac': jacfunc}
        ),
    )


def test_jacobian_with_pickle():
    """Test using pickle.dumps/loads."""
    params = Parameters()
    params.add('a', value=10)
    params.add('b', value=10)
    params.add('c', value=10)

    a, b, c = 2.5, 1.3, 0.8
    x = np.linspace(0, 4, 50)
    y = f([a, b, c], x)
    np.random.seed(2021)
    data = y + 0.15*np.random.normal(size=x.size)

    for method in ('leastsq', 'least_squares'):
        if method == 'leastsq':
            kwargs = {'Dfun': dfunc, 'col_deriv': 1}
        else:
            kwargs = {'jac': jacfunc}

        min = Minimizer(
            func,
            params,
            fcn_args=(x,),
            fcn_kws={'data': data},
            **kwargs,
        )

        pickled = dumps(min)
        unpickled = loads(pickled)

        out = unpickled.minimize(method=method)

        assert np.isclose(out.params['a'], 2.5635, atol=1e-4)
        assert np.isclose(out.params['b'], 1.3585, atol=1e-4)
        assert np.isclose(out.params['c'], 0.8241, atol=1e-4)


def test_jacobian_with_forkingpickler():
    """Test using multiprocessing.Pool, which uses a subclass of pickle.Pickler."""
    a, b, c = 2.5, 1.3, 0.8
    x = np.linspace(0, 4, 50)
    y = f([a, b, c], x)
    np.random.seed(2021)
    data = y + 0.15*np.random.normal(size=x.size)

    with get_context(method='spawn').Pool(1) as pool:
        iterator = pool.imap_unordered(
            wrapper,
            (
                (
                    method,
                    x,
                    data,
                )
                for method in ('leastsq', 'least_squares')
            ),
            chunksize=1,
        )

        while True:
            try:
                out = iterator.next(timeout=30)
            except StopIteration:
                break

            assert np.isclose(out.params['a'], 2.5635, atol=1e-4)
            assert np.isclose(out.params['b'], 1.3585, atol=1e-4)
            assert np.isclose(out.params['c'], 0.8241, atol=1e-4)
