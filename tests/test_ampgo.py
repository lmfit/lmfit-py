import numpy as np
from numpy.testing import assert_allclose

import lmfit


def test_ampgo_Alpine02():
    """Test AMPGO algorithm on Alpine02 function."""

    global_optimum = [7.91705268, 4.81584232]
    fglob = -6.12950

    def residual_Alpine02(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.prod(np.sqrt(x0) * np.sin(x0)) * np.prod(np.sqrt(x1) *
                                                           np.sin(x1))

    pars = lmfit.Parameters()
    pars.add_many(('x0', 1., True, 0.0, 10.0),
                  ('x1', 1., True, 0.0, 10.0))

    mini = lmfit.Minimizer(residual_Alpine02, pars)
    out = mini.minimize(method='ampgo')
    out_x = np.array([out.params['x0'].value, out.params['x1'].value])

    assert_allclose(out.residual, fglob, rtol=1e-5)
    assert_allclose(min(out_x), min(global_optimum), rtol=1e-3)
    assert_allclose(max(out_x), max(global_optimum), rtol=1e-3)
    assert('global' in out.ampgo_msg)



def test_ampgo_Alpine02_maxfunevals():
    """Test AMPGO algorithm on Alpine02 function."""

    def residual_Alpine02(params):
        x0 = params['x0'].value
        x1 = params['x1'].value
        return np.prod(np.sqrt(x0) * np.sin(x0)) * np.prod(np.sqrt(x1) *
                                                           np.sin(x1))

    pars = lmfit.Parameters()
    pars.add_many(('x0', 1., True, 0.0, 10.0),
                  ('x1', 1., True, 0.0, 10.0))

    mini = lmfit.Minimizer(residual_Alpine02, pars)
    kws = {'maxfunevals': 50}
    out = mini.minimize(method='ampgo', **kws)
    assert('function' in out.ampgo_msg)


if __name__ == '__main__':
    test_ampgo_Alpine02()
    test_ampgo_Alpine02_maxfunevals()
