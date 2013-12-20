import unittest
import warnings
from numpy.testing import assert_allclose
from lmfit.utilfuncs import assert_results_close
import numpy as np

from lmfit import Model, Parameter
from lmfit import specified_models
from lmfit.utilfuncs import gaussian, normalized_gaussian


class TestUserDefiniedModel(unittest.TestCase):
    # mainly aimed at checking that the API does what it says it does
    # and raises the right exceptions or warnings when things are not right

    def setUp(self):
        self.x = np.linspace(-10, 10, num=1000)
        np.random.seed(1)
        self.noise = 0.01*np.random.randn(*self.x.shape)
        self.true_values = lambda: dict(height=7, center=1, sigma=3)
        self.guess = lambda: dict(height=5, center=2, sigma=4)
        # return a fresh copy
        self.model = Model(gaussian, ['x'])
        self.data = gaussian(x=self.x, **self.true_values()) + self.noise

    def test_fit_with_keyword_params(self):
        result = self.model.fit(self.data, x=self.x, **self.guess())
        assert_results_close(result.values, self.true_values())

    def test_fit_with_parameters_obj(self):
        params = self.model.params()
        for param_name, value in self.guess().items():
            params[param_name].value = value
        result = self.model.fit(self.data, params, x=self.x)
        assert_results_close(result.values, self.true_values())

    def test_missing_param_raises_error(self):

        # using keyword argument parameters
        guess_missing_sigma = self.guess()
        del guess_missing_sigma['sigma']
        f = lambda: self.model.fit(self.data, x=self.x, **guess_missing_sigma)
        self.assertRaises(ValueError, f)

        # using Parameters
        params = self.model.params()
        for param_name, value in guess_missing_sigma.items():
            params[param_name].value = value
        f = lambda: self.model.fit(self.data, params, x=self.x)

    def test_extra_param_issues_warning(self):
        # The function accepts extra params, Model will warn but not raise.
        guess = self.guess()
        guess['extra'] = 5

        def flexible_func(x, height, center, sigma, **kwargs):
            return gaussian(x, height, center, sigma)
        
        flexible_model = Model(flexible_func, ['x'])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flexible_model.fit(self.data, x=self.x, **guess)
        self.assertTrue(len(w) == 1)
        self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_missing_independent_variable_raises_error(self):
        f = lambda: self.model.fit(self.data, **self.guess())
        self.assertRaises(KeyError, f)

    def test_bounding(self):
        guess = self.guess()
        guess['center'] = Parameter(value=2, min=1.3)
        true_values = self.true_values()
        true_values['center'] = 1.3  # as close as it's allowed to get
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)

    def test_vary_false(self):
        guess = self.guess()
        guess['center'] = Parameter(value=1.3, vary=False)
        true_values = self.true_values()
        true_values['center'] = 1.3
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)

    def test_result_attributes(self):

        # result.init_values
        result = self.model.fit(self.data, x=self.x, **self.guess())
        assert_results_close(result.values, self.true_values())
        self.assertTrue(result.init_values == self.guess())

        # result.init_params
        params = self.model.params()
        for param_name, value in self.guess().items():
            params[param_name].value = value
        self.assertTrue(result.init_params == params)

        # result.best_fit
        assert_allclose(result.best_fit, self.data, atol=self.noise.max())

        # result.init_fit
        init_fit = self.model.func(x=self.x, **self.guess())
        assert_allclose(result.init_fit, init_fit)

        # result.model
        self.assertTrue(result.model is self.model)

    # testing model addition...

    def test_user_defined_gaussian_plus_constant(self):
        data = self.data + 5
        model = self.model + specified_models.Constant()
        guess = self.guess()
        guess['c'] = 10
        true_values = self.true_values()
        true_values['c'] = 5
        result = model.fit(data, x=self.x, **guess)
        assert_results_close(result.values, true_values)

    def test_sum_of_two_gaussians(self):

        # two user-defined gaussians
        model1 = self.model
        f2 = lambda x, height_, center_, sigma_: gaussian(
            x, height_, center_, sigma_)
        model2 = Model(f2, ['x'])
        values1 = self.true_values()
        values2 = self.true_values()
        values2['sigma'] = 1.5
        values2['height'] = 4
        data = gaussian(x=self.x, **values1)
        data += gaussian(x=self.x, **values2)
        model = self.model + model2
        values2 = {k + '_': v for k, v in values2.items()}
        guess = {'sigma': Parameter(value=2, min=0), 'center': 1,
                 'height': 1,
                 'sigma_': Parameter(value=1, min=0), 'center_': 1,
                 'height_': 1}

        true_values = dict(list(values1.items()) + list(values2.items()))
        result = model.fit(data, x=self.x, **guess)
        assert_results_close(result.values, true_values)

        # user-defined models with common parameter names
        # cannot be added, and should raise
        f = lambda: model1 + model1
        self.assertRaises(NameError, f)

        # two predefined_gaussians, using suffix to differentiate
        model1 = specified_models.Gaussian(['x'])
        model2 = specified_models.Gaussian(['x'], suffix='_')
        model = model1 + model2
        true_values = {'center': values1['center'],
                       'height': values1['height'],
                       'sigma': values1['sigma'],
                       'center_': values2['center_'],
                       'height_': values2['height_'],
                       'sigma_': values2['sigma_']}
        guess = {'sigma': 2, 'center': 1, 'height': 1,
                 'sigma_': 1, 'center_': 1, 'height_': 1}
        result = model.fit(data, x=self.x, **guess)
        assert_results_close(result.values, true_values)

        # without suffix, the names collide and Model should raise
        model1 = specified_models.Gaussian(['x'])
        model2 = specified_models.Gaussian(['x'])
        f = lambda: model1 + model2
        self.assertRaises(NameError, f)


class CommonTests(object):
    # to be subclassed for testing predefined models

    def setUp(self):
        self.x = np.linspace(1, 10, num=1000)
        noise = 0.0001*np.random.randn(*self.x.shape)
        # Some Models need args (e.g., polynomial order), and others don't.
        try:
            args = self.args
        except AttributeError:
            self.model_instance = self.model(['x'])
            func = self.model_instance.func

        else:
            self.model_instance = self.model(*args, independent_vars=['x'])
            func = self.model_instance.func
        self.data = func(x=self.x, **self.true_values()) + noise

    def test_fit(self):
        model = self.model_instance
        result = model.fit(self.data, x=self.x, **self.guess())
        assert_results_close(result.values, self.true_values())


class TestNormalizedGaussian(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(center=0, sigma=1.5)
        self.guess = lambda: dict(center=1, sigma=2)
        self.model = specified_models.NormalizedGaussian
        super(TestNormalizedGaussian, self).setUp()


class TestLinear(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(slope=5, intercept=2)
        self.guess = lambda: dict(slope=10, intercept=6)
        self.model = specified_models.Linear
        super(TestLinear, self).setUp()


class TestParabolic(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(a=5, b=2, c=8)
        self.guess = lambda: dict(a=1, b=6, c=3)
        self.model = specified_models.Parabolic
        super(TestParabolic, self).setUp()


class TestPolynomialOrder2(CommonTests, unittest.TestCase):
   # class Polynomial constructed with order=2

    def setUp(self):
        self.true_values = lambda: dict(c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c1=1, c2=6, c0=3)
        self.model = specified_models.Polynomial
        self.args = (2,)
        super(TestPolynomialOrder2, self).setUp()


class TestPolynomialOrder3(CommonTests, unittest.TestCase):
   # class Polynomial constructed with order=3

    def setUp(self):
        self.true_values = lambda: dict(c3=2, c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c3=1, c1=1, c2=6, c0=3)
        self.model = specified_models.Polynomial
        self.args = (3,)
        super(TestPolynomialOrder3, self).setUp()


class TestConstant(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(c=5)
        self.guess = lambda: dict(c=2)
        self.model = specified_models.Constant
        super(TestConstant, self).setUp()


class TestPowerlaw(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(coefficient=5, exponent=3)
        self.guess = lambda: dict(coefficient=2, exponent=8)
        self.model = specified_models.PowerLaw
        super(TestPowerlaw, self).setUp()


class TestExponential(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(amplitude=5, decay=3)
        self.guess = lambda: dict(amplitude=2, decay=8)
        self.model = specified_models.Exponential
        super(TestExponential, self).setUp()
