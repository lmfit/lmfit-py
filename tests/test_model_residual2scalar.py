import unittest
import warnings
import nose
from numpy.testing import assert_allclose, assert_raises
from numpy.testing.decorators import knownfailureif
import numpy as np

from lmfit import Model, Parameter, models
from lmfit.lineshapes import gaussian

def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03,
                         err_msg='', verbose=True):
    for param_name, value in desired.items():
         assert_allclose(actual[param_name], value, rtol, atol, err_msg, verbose)

def _skip_if_no_pandas():
    try:
        import pandas
    except ImportError:
        raise nose.SkipTest("Skipping tests that require pandas.")



class TestResidual2Scalar(unittest.TestCase):
    def test1(self):
        model = models.GaussianModel(residual2scalar=lambda x: np.sum(np.abs(x)))
        params = model.make_params()
        params["amplitude"].set(100)
        x=np.linspace(-10,10)
        data = model.eval(params, x=x)
        model.fit(data, params, x=x)
