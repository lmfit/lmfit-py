import unittest
import warnings
from lmfit.utilfuncs import assert_results_close
import numpy as np

from lmfit import Model, Parameter
from lmfit import specified_models

def gaussian(x, amp, cen, sd):
    # N.B. not defined the same way in lmfit.utilfuncs
    return amp*np.exp(-(x-cen)**2/(2*sd**2))

class TestUserDefiniedModel(unittest.TestCase):
    # mainly aimed at checking that the API does what it says it does
    # and raises the right exceptions or warnings when things are not right

    def setUp(self):
        self.x = np.linspace(-10, 10, num=1000)
        self.noise = 0.01*np.random.randn(*self.x.shape)
        self.true_values = lambda: dict(amp=7, cen=1, sd=3)
        self.guess = lambda: dict(amp=5, cen=2, sd=4)  # return a fresh copy
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
        guess_missing_sd = self.guess()
        del guess_missing_sd['sd']
        f = lambda: self.model.fit(self.data, x=self.x, **guess_missing_sd)
        self.assertRaises(ValueError, f)

        # using Parameters
        params = self.model.params()
        for param_name, value in guess_missing_sd.iteritems():
            params[param_name].value = value
        f = lambda: self.model.fit(self.data, params, x=self.x)

    def test_extra_param_issues_warning(self):
        # The function accepts extra params, Model will warn but not raise.
        guess = self.guess()
        guess['extra'] = 5
        def flexible_func(x, amp, cen, sd, **kwargs):
            return gaussian(x, amp, cen, sd)
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
        guess['cen'] = Parameter(value=2, min=1.3)
        true_values = self.true_values()
        true_values['cen'] = 1.3  # as close as it's allowed to get
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)

    def test_vary_false(self):
        guess = self.guess()
        guess['cen'] = Parameter(value=1.3, vary=False)
        true_values = self.true_values()
        true_values['cen'] = 1.3
        result = self.model.fit(self.data, x=self.x, **guess)
        assert_results_close(result.values, true_values, rtol=0.05)

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
        f2 = lambda x, amp_, cen_, sd_: gaussian(x, amp_, cen_, sd_)
        model2 = Model(f2, ['x'])
        values1 = self.true_values()
        values2 = self.true_values()
        values2['sd'] = 1.5 
        values2['amp'] = 4
        data = gaussian(x=self.x, **values1)
        data += gaussian(x=self.x, **values2)
        model = self.model + model2
        values2 = {k + '_': v for k, v in values2.items()}
        guess = {'sd': Parameter(value=2, min=0), 'cen': 1, 'amp': 1, 
                 'sd_': Parameter(value=1, min=0), 'cen_': 1, 'amp_': 1}

        true_values = dict(values1.items() + values2.items())
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
        true_values = {'center': values1['cen'],
                       'height': values1['amp'],
                       'sd': values1['sd'],
                       'center_': values2['cen_'],
                       'height_': values2['amp_'],
                       'sd_': values2['sd_']}
        guess = {'sd': 2, 'center': 1, 'height': 1, 
                 'sd_': 1, 'center_': 1, 'height_': 1}
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
    	self.true_values = lambda: dict(center=0, sd=1.5)
    	self.guess = lambda: dict(center=1, sd=2)
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
