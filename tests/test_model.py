import unittest
import warnings
import nose
from numpy.testing import assert_allclose
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


class CommonTests(object):
    # to be subclassed for testing predefined models

    def setUp(self):
        np.random.seed(1)
        self.noise = 0.0001*np.random.randn(*self.x.shape)
        # Some Models need args (e.g., polynomial order), and others don't.
        try:
            args = self.args
        except AttributeError:
            self.model = self.model_constructor()
            self.model_drop = self.model_constructor(missing='drop')
            self.model_raise = self.model_constructor(missing='raise')
            self.model_explicit_var = self.model_constructor(['x'])
            func = self.model.func
        else:
            self.model = self.model_constructor(*args)
            self.model_drop = self.model_constructor(*args, missing='drop')
            self.model_raise = self.model_constructor(*args, missing='raise')
            self.model_explicit_var = self.model_constructor(
                *args, independent_vars=['x'])
            func = self.model.func
        self.data = func(x=self.x, **self.true_values()) + self.noise

    @property
    def x(self):
        return np.linspace(1, 10, num=1000)

    def test_fit(self):
        model = self.model

        # Pass Parameters object.
        params = model.make_params(**self.guess())
        result = model.fit(self.data, params, x=self.x)
        assert_results_close(result.values, self.true_values())

        # Pass inidividual Parameter objects as kwargs.
        kwargs = {name: p for name, p in params.items()}
        result = self.model.fit(self.data, x=self.x, **kwargs)
        assert_results_close(result.values, self.true_values())

        # Pass guess values (not Parameter objects) as kwargs.
        kwargs = {name: p.value for name, p in params.items()}
        result = self.model.fit(self.data, x=self.x, **kwargs)
        assert_results_close(result.values, self.true_values())

    def test_explicit_independent_vars(self):
        self.check_skip_independent_vars()
        model = self.model_explicit_var
        pars = model.make_params(**self.guess())
        result = model.fit(self.data, pars, x=self.x)
        assert_results_close(result.values, self.true_values())

    def test_fit_with_weights(self):
        model = self.model

        # fit without weights
        params = model.make_params(**self.guess())
        out1 = model.fit(self.data, params, x=self.x)

        # fit with weights
        weights = 1.0/(0.5 + self.x**2)
        out2 = model.fit(self.data, params, weights=weights, x=self.x)

        max_diff = 0.0
        for parname, val1 in out1.values.items():
            val2 = out2.values[parname]
            if max_diff < abs(val1-val2):
                max_diff = abs(val1-val2)
        assert(max_diff > 1.e-8)

    def test_result_attributes(self):
        pars = self.model.make_params(**self.guess())
        result = self.model.fit(self.data, pars, x=self.x)

        # result.init_values
        assert_results_close(result.values, self.true_values())
        self.assertEqual(result.init_values, self.guess())

        # result.init_params
        params = self.model.make_params()
        for param_name, value in self.guess().items():
            params[param_name].value = value
        self.assertEqual(result.init_params, params)

        # result.best_fit
        assert_allclose(result.best_fit, self.data, atol=self.noise.max())

        # result.init_fit
        init_fit = self.model.func(x=self.x, **self.guess())
        assert_allclose(result.init_fit, init_fit)

        # result.model
        self.assertTrue(result.model is self.model)

    def test_result_eval(self):
        # Check eval() output against init_fit and best_fit.
        pars = self.model.make_params(**self.guess())
        result = self.model.fit(self.data, pars, x=self.x)

        assert_allclose(result.eval(x=self.x, **result.values),
                        result.best_fit)
        assert_allclose(result.eval(x=self.x, **result.init_values),
                        result.init_fit)

    def test_result_eval_custom_x(self):
        self.check_skip_independent_vars()
        pars = self.model.make_params(**self.guess())
        result = self.model.fit(self.data, pars, x=self.x)

        # Check that the independent variable is respected.
        short_eval = result.eval(x=np.array([0, 1, 2]), **result.values)
        self.assertEqual(len(short_eval), 3)

    def test_data_alignment(self):
        _skip_if_no_pandas()
        from pandas import Series

        # Align data and indep var of different lengths using pandas index.
        data = Series(self.data.copy()).iloc[10:-10]
        x = Series(self.x.copy())

        model = self.model
        params = model.make_params(**self.guess())
        result = model.fit(data, params, x=x)
        result = model.fit(data, params, x=x)
        assert_results_close(result.values, self.true_values())

        # Skip over missing (NaN) values, aligning via pandas index.
        data.iloc[500:510] = np.nan
        result = self.model_drop.fit(data, params, x=x)
        assert_results_close(result.values, self.true_values())

        # Raise if any NaN values are present.
        raises = lambda: self.model_raise.fit(data, params, x=x)
        self.assertRaises(ValueError, raises)

    def check_skip_independent_vars(self):
        # to be overridden for models that do not accept indep vars
        pass

    def test_aic(self):
        model = self.model

        # Pass Parameters object.
        params = model.make_params(**self.guess())
        result = model.fit(self.data, params, x=self.x)
        aic = result.aic
        self.assertTrue(aic < 0) # aic must be negative

        # Pass extra unused Parameter.
        params.add("unused_param", value=1.0, vary=True)
        result = model.fit(self.data, params, x=self.x)
        aic_extra = result.aic
        self.assertTrue(aic_extra < 0)   # aic must be negative
        self.assertTrue(aic < aic_extra) # the extra param should lower the aic


    def test_bic(self):
        model = self.model

        # Pass Parameters object.
        params = model.make_params(**self.guess())
        result = model.fit(self.data, params, x=self.x)
        bic = result.bic
        self.assertTrue(bic < 0) # aic must be negative

        # Compare to AIC
        aic = result.aic
        self.assertTrue(aic < bic) # aic should be lower than bic

        # Pass extra unused Parameter.
        params.add("unused_param", value=1.0, vary=True)
        result = model.fit(self.data, params, x=self.x)
        bic_extra = result.bic
        self.assertTrue(bic_extra < 0)   # bic must be negative
        self.assertTrue(bic < bic_extra) # the extra param should lower the bic




class TestUserDefiniedModel(CommonTests, unittest.TestCase):
    # mainly aimed at checking that the API does what it says it does
    # and raises the right exceptions or warnings when things are not right

    def setUp(self):
        self.true_values = lambda: dict(amplitude=7.1, center=1.1, sigma=2.40)
        self.guess = lambda: dict(amplitude=5, center=2, sigma=4)
        # return a fresh copy
        self.model_constructor = (
            lambda *args, **kwargs: Model(gaussian, *args, **kwargs))
        super(TestUserDefiniedModel, self).setUp()

    @property
    def x(self):
        return np.linspace(-10, 10, num=1000)

    def test_lists_become_arrays(self):
        # smoke test
        self.model.fit([1, 2, 3], x=[1, 2, 3], **self.guess())
        self.model.fit([1, 2, None, 3], x=[1, 2, 3, 4], **self.guess())

    def test_missing_param_raises_error(self):

        # using keyword argument parameters
        guess_missing_sigma = self.guess()
        del guess_missing_sigma['sigma']
        #f = lambda: self.model.fit(self.data, x=self.x, **guess_missing_sigma)
        #self.assertRaises(ValueError, f)

        # using Parameters
        params = self.model.make_params()
        for param_name, value in guess_missing_sigma.items():
            params[param_name].value = value
        f = lambda: self.model.fit(self.data, params, x=self.x)

    def test_extra_param_issues_warning(self):
        # The function accepts extra params, Model will warn but not raise.
        def flexible_func(x, amplitude, center, sigma, **kwargs):
            return gaussian(x, amplitude, center, sigma)

        flexible_model = Model(flexible_func)
        pars = flexible_model.make_params(**self.guess())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flexible_model.fit(self.data, pars, x=self.x, extra=5)
        self.assertTrue(len(w) == 1)
        self.assertTrue(issubclass(w[-1].category, UserWarning))

    def test_missing_independent_variable_raises_error(self):
        pars = self.model.make_params(**self.guess())
        f = lambda: self.model.fit(self.data, pars)
        self.assertRaises(KeyError, f)

    def test_bounding(self):
        true_values = self.true_values()
        true_values['center'] = 1.3  # as close as it's allowed to get
        pars = self.model.make_params(**self.guess())
        pars['center'].set(value=2, min=1.3)
        result = self.model.fit(self.data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.05)

    def test_vary_false(self):
        true_values = self.true_values()
        true_values['center'] = 1.3
        pars = self.model.make_params(**self.guess())
        pars['center'].set(value=1.3, vary=False)
        result = self.model.fit(self.data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.05)

    # testing model addition...

    def test_user_defined_gaussian_plus_constant(self):
        data = self.data + 5.0
        model = self.model + models.ConstantModel()
        guess = self.guess()
        pars = model.make_params(c= 10.1, **guess)
        true_values = self.true_values()
        true_values['c'] = 5.0

        result = model.fit(data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.01, atol=0.01)

    def test_model_with_prefix(self):
        # model with prefix of 'a' and 'b'
        mod = models.GaussianModel(prefix='a')
        vals = {'center': 2.45, 'sigma':0.8, 'amplitude':3.15}
        data = gaussian(x=self.x, **vals) + self.noise/3.0
        pars = mod.guess(data, x=self.x)
        self.assertTrue('aamplitude' in pars)
        self.assertTrue('asigma' in pars)
        out = mod.fit(data, pars, x=self.x)
        self.assertTrue(out.params['aamplitude'].value > 2.0)
        self.assertTrue(out.params['acenter'].value > 2.0)
        self.assertTrue(out.params['acenter'].value < 3.0)

        mod = models.GaussianModel(prefix='b')
        data = gaussian(x=self.x, **vals) + self.noise/3.0
        pars = mod.guess(data, x=self.x)
        self.assertTrue('bamplitude' in pars)
        self.assertTrue('bsigma' in pars)

    def test_change_prefix(self):
        mod = models.GaussianModel(prefix='b')
        mod.prefix = 'c'
        params = mod.make_params()
        names = params.keys()
        all_begin_with_c = all([n.startswith('c') for n in names])
        self.assertTrue(all_begin_with_c)

    def test_sum_of_two_gaussians(self):
        # two user-defined gaussians
        model1 = self.model
        f2 = lambda x, amp, cen, sig: gaussian(x, amplitude=amp, center=cen, sigma=sig)
        model2 = Model(f2)
        values1 = self.true_values()
        values2 = {'cen': 2.45, 'sig':0.8, 'amp':3.15}

        data  = gaussian(x=self.x, **values1) + f2(x=self.x, **values2) + self.noise/3.0
        model = self.model + model2
        pars = model.make_params()
        pars['sigma'].set(value=2, min=0)
        pars['center'].set(value=1, min=0.2, max=1.8)
        pars['amplitude'].set(value=3, min=0)
        pars['sig'].set(value=1, min=0)
        pars['cen'].set(value=2.4, min=2, max=3.5)
        pars['amp'].set(value=1, min=0)

        true_values = dict(list(values1.items()) + list(values2.items()))
        result = model.fit(data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.01, atol=0.01)

        # user-defined models with common parameter names
        # cannot be added, and should raise
        f = lambda: model1 + model1
        self.assertRaises(NameError, f)

        # two predefined_gaussians, using suffix to differentiate
        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')
        model = model1 + model2
        true_values = {'g1_center': values1['center'],
                       'g1_amplitude': values1['amplitude'],
                       'g1_sigma': values1['sigma'],
                       'g2_center': values2['cen'],
                       'g2_amplitude': values2['amp'],
                       'g2_sigma': values2['sig']}
        pars = model.make_params()
        pars['g1_sigma'].set(2)
        pars['g1_center'].set(1)
        pars['g1_amplitude'].set(3)
        pars['g2_sigma'].set(1)
        pars['g2_center'].set(2.4)
        pars['g2_amplitude'].set(1)

        result = model.fit(data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.01, atol=0.01)

        # without suffix, the names collide and Model should raise
        model1 = models.GaussianModel()
        model2 = models.GaussianModel()
        f = lambda: model1 + model2
        self.assertRaises(NameError, f)

    def test_sum_composite_models(self):
        # test components of composite model created adding composite model
        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')
        model3 = models.GaussianModel(prefix='g3_')
        model4 = models.GaussianModel(prefix='g4_')

        model_total1 = (model1 + model2) + model3
        for mod in [model1, model2, model3]:
            self.assertTrue(mod in model_total1.components)

        model_total2 = model1 + (model2 + model3)
        for mod in [model1, model2, model3]:
            self.assertTrue(mod in model_total2.components)

        model_total3 = (model1 + model2) + (model3 + model4)
        for mod in [model1, model2, model3, model4]:
            self.assertTrue(mod in model_total3.components)

    def test_composite_has_bestvalues(self):
        # test that a composite model has non-empty best_values
        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')

        mod  = model1 + model2
        pars = mod.make_params()

        values1 = dict(amplitude=7.10, center=1.1, sigma=2.40)
        values2 = dict(amplitude=12.2, center=2.5, sigma=0.5)
        data  = gaussian(x=self.x, **values1) + gaussian(x=self.x, **values2) + 0.1*self.noise

        pars['g1_sigma'].set(2)
        pars['g1_center'].set(1, max=1.5)
        pars['g1_amplitude'].set(3)
        pars['g2_sigma'].set(1)
        pars['g2_center'].set(2.6, min=2.0)
        pars['g2_amplitude'].set(1)

        result = mod.fit(data, params=pars, x=self.x)

        self.assertTrue(len(result.best_values) == 6)

        self.assertTrue(abs(result.params['g1_amplitude'].value -  7.1) < 0.5)
        self.assertTrue(abs(result.params['g2_amplitude'].value - 12.2) < 0.5)
        self.assertTrue(abs(result.params['g1_center'].value    -  1.1) < 0.2)
        self.assertTrue(abs(result.params['g2_center'].value    -  2.5) < 0.2)


    def test_hints_in_composite_models(self):
        # test propagation of hints from base models to composite model
        def func(x, amplitude):
            pass

        m1 = Model(func, prefix='p1_')
        m2 = Model(func, prefix='p2_')

        m1.set_param_hint('amplitude', value=1)
        m2.set_param_hint('amplitude', value=2)

        mx = (m1 + m2)
        params = mx.make_params()
        param_values = {name: p.value for name, p in params.items()}
        self.assertEqual(param_values['p1_amplitude'], 1)
        self.assertEqual(param_values['p2_amplitude'], 2)


class TestLinear(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(slope=5, intercept=2)
        self.guess = lambda: dict(slope=10, intercept=6)
        self.model_constructor = models.LinearModel
        super(TestLinear, self).setUp()


class TestParabolic(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(a=5, b=2, c=8)
        self.guess = lambda: dict(a=1, b=6, c=3)
        self.model_constructor = models.ParabolicModel
        super(TestParabolic, self).setUp()


class TestPolynomialOrder2(CommonTests, unittest.TestCase):
   # class Polynomial constructed with order=2

    def setUp(self):
        self.true_values = lambda: dict(c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c1=1, c2=6, c0=3)
        self.model_constructor = models.PolynomialModel
        self.args = (2,)
        super(TestPolynomialOrder2, self).setUp()


class TestPolynomialOrder3(CommonTests, unittest.TestCase):
   # class Polynomial constructed with order=3

    def setUp(self):
        self.true_values = lambda: dict(c3=2, c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c3=1, c1=1, c2=6, c0=3)
        self.model_constructor = models.PolynomialModel
        self.args = (3,)
        super(TestPolynomialOrder3, self).setUp()


class TestConstant(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(c=5)
        self.guess = lambda: dict(c=2)
        self.model_constructor = models.ConstantModel
        super(TestConstant, self).setUp()

    def check_skip_independent_vars(self):
        raise nose.SkipTest("ConstantModel has not independent_vars.")

class TestPowerlaw(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(amplitude=5, exponent=3)
        self.guess = lambda: dict(amplitude=2, exponent=8)
        self.model_constructor = models.PowerLawModel
        super(TestPowerlaw, self).setUp()


class TestExponential(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(amplitude=5, decay=3)
        self.guess = lambda: dict(amplitude=2, decay=8)
        self.model_constructor = models.ExponentialModel
        super(TestExponential, self).setUp()
