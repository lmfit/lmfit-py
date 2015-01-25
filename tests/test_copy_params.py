"""
Originally added for gh-#56
"""
import numpy as np
from lmfit import Parameters, minimize, report_fit, Minimizer
import unittest
from numpy.testing import assert_, assert_equal

class test_copy_params(unittest.TestCase):
    def setUp(self):

        self.x = np.arange(0, 1, 0.01)
        self.y1 = 1.5 * np.exp(0.9 * self.x) + np.random.normal(scale=0.001, size=len(self.x))
        self.y2 = 2.0 + self.x + 1/2. * self.x**2 +1/3. * self.x**3
        self.y2 = self.y2 + np.random.normal(scale=0.001, size=len(self.x))

    def residual(self, params, x, data):
        a = params['a'].value
        b = params['b'].value

        model = a * np.exp(b * x)
        return (data - model)

    def params_values(self, params):
        return list(params.valuesdict().values())

    def test_copy_params(self):
        # checking output for gh-#56
        # 1. the output.params is a different instance to the supplied params
        # instance
        # 2. the supplied params instance changes its values after the fit.

        params = Parameters()
        params.add('a', value = 2.0)
        params.add('b', value = 2.0)

        # fit to first data set
        out1 = minimize(self.residual, params, args=(self.x, self.y1))

        assert_equal(self.params_values(params),
                     self.params_values(out1.params))
        assert_(not params is out1)

        # fit to second data set
        out2 = minimize(self.residual, params, args=(self.x, self.y2))

        assert_equal(self.params_values(params),
                     self.params_values(out2.params))
        assert_(not params is out2)
        assert_(not out1 is out2)

        adiff = out1.params['a'].value - out2.params['a'].value
        bdiff = out1.params['b'].value - out2.params['b'].value

        assert(abs(adiff) > 1.e-2)
        assert(abs(bdiff) > 1.e-2)

    def test_copy_params_Minimizer(self):
        # check that the params instance supplied to construct the Minimizer
        # object is present as Minimizer.params for the lifetime of the object

        params = Parameters()
        params.add('a', value = 2.0)
        params.add('b', value = 2.0)

        fitter = Minimizer(self.residual, params, fcn_args=(self.x, self.y1))

        #checks that fitter.params is the same instance as params
        assert_equal(self.params_values(params),
                     self.params_values(fitter.params))

        assert_(params is fitter.params)

        fitter.minimize()

        # checks that fitter.params is the same instance as params, after a fit.
        # Checking that the values are the same is almost redundant.
        assert_equal(self.params_values(params),
                     self.params_values(fitter.params))
        assert_(params is fitter.params)
