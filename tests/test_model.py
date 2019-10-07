import functools
import sys
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from lmfit import Model, models
from lmfit.lineshapes import gaussian
from lmfit.models import PseudoVoigtModel


def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03, err_msg='',
                         verbose=True):
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol, atol, err_msg,
                        verbose)


def _skip_if_no_pandas():
    try:
        import pandas  # noqa: F401
    except ImportError:
        raise pytest.skip("Skipping tests that require pandas.")


def firstarg_ndarray(func):
    """a simple wrapper used for testing that wrapped
    functions can be model functions"""
    @functools.wraps(func)
    def wrapper(x, *args, **kws):
        x = np.asarray(x)
        return func(x, *args, **kws)
    return wrapper


@firstarg_ndarray
def linear_func(x, a, b):
    "test wrapped model function"
    return a*x+b


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
            self.model_omit = self.model_constructor(nan_policy='omit')
            self.model_raise = self.model_constructor(nan_policy='raise')
            self.model_explicit_var = self.model_constructor(['x'])
            func = self.model.func
        else:
            self.model = self.model_constructor(*args)
            self.model_omit = self.model_constructor(*args, nan_policy='omit')
            self.model_raise = self.model_constructor(*args, nan_policy='raise')
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
        assert max_diff > 1.e-8

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
        if hasattr(short_eval, '__len__'):
            self.assertEqual(len(short_eval), 3)

    def test_result_report(self):
        pars = self.model.make_params(**self.guess())
        result = self.model.fit(self.data, pars, x=self.x)
        report = result.fit_report()
        assert "[[Model]]" in report
        assert "[[Variables]]" in report
        assert "[[Fit Statistics]]" in report
        assert " # function evals   =" in report
        assert " Akaike " in report
        assert " chi-square " in report

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
        result = self.model_omit.fit(data, params, x=x)
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
        self.assertTrue(aic < 0)  # aic must be negative

        # Pass extra unused Parameter.
        params.add("unused_param", value=1.0, vary=True)
        result = model.fit(self.data, params, x=self.x)
        aic_extra = result.aic
        self.assertTrue(aic_extra < 0)  # aic must be negative
        self.assertTrue(aic < aic_extra)  # extra param should lower the aic

    def test_bic(self):
        model = self.model

        # Pass Parameters object.
        params = model.make_params(**self.guess())
        result = model.fit(self.data, params, x=self.x)
        bic = result.bic
        self.assertTrue(bic < 0)  # aic must be negative

        # Compare to AIC
        aic = result.aic
        self.assertTrue(aic < bic)  # aic should be lower than bic

        # Pass extra unused Parameter.
        params.add("unused_param", value=1.0, vary=True)
        result = model.fit(self.data, params, x=self.x)
        bic_extra = result.bic
        self.assertTrue(bic_extra < 0)  # bic must be negative
        self.assertTrue(bic < bic_extra)  # extra param should lower the bic


class TestUserDefiniedModel(CommonTests, unittest.TestCase):
    # mainly aimed at checking that the API does what it says it does
    # and raises the right exceptions or warnings when things are not right
    import six
    if six.PY2:
        from six import assertRaisesRegex

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
        pytest.raises(ValueError,
                      self.model.fit,
                      [1, 2, None, 3],
                      x=[1, 2, 3, 4],
                      **self.guess())

    def test_missing_param_raises_error(self):
        # using keyword argument parameters
        guess_missing_sigma = self.guess()
        del guess_missing_sigma['sigma']
        # f = lambda: self.model.fit(self.data, x=self.x, **guess_missing_sigma)
        # self.assertRaises(ValueError, f)

        # using Parameters
        params = self.model.make_params()
        for param_name, value in guess_missing_sigma.items():
            params[param_name].value = value
        self.model.fit(self.data, params, x=self.x)

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
        pars = model.make_params(c=10.1, **guess)
        true_values = self.true_values()
        true_values['c'] = 5.0

        result = model.fit(data, pars, x=self.x)
        assert_results_close(result.values, true_values, rtol=0.01, atol=0.01)

    def test_model_with_prefix(self):
        # model with prefix of 'a' and 'b'
        mod = models.GaussianModel(prefix='a')
        vals = {'center': 2.45, 'sigma': 0.8, 'amplitude': 3.15}
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
        "should pass!"
        mod = models.GaussianModel(prefix='b')
        set_prefix_failed = None
        try:
            mod.prefix = 'c'
            set_prefix_failed = False
        except AttributeError:
            set_prefix_failed = True
        except:  # noqa: E722
            set_prefix_failed = None
        self.assertFalse(set_prefix_failed)

        new_expr = mod.param_hints['fwhm']['expr']
        self.assertTrue('csigma' in new_expr)
        self.assertFalse('bsigma' in new_expr)

    def test_model_name(self):
        # test setting the name for built-in models
        mod = models.GaussianModel(name='user_name')
        self.assertEqual(mod.name, "Model(user_name)")

    def test_sum_of_two_gaussians(self):
        # two user-defined gaussians
        model1 = self.model
        f2 = lambda x, amp, cen, sig: gaussian(x, amplitude=amp, center=cen,
                                               sigma=sig)
        model2 = Model(f2)
        values1 = self.true_values()
        values2 = {'cen': 2.45, 'sig': 0.8, 'amp': 3.15}

        data = (gaussian(x=self.x, **values1) + f2(x=self.x, **values2) +
                self.noise/3.0)
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

    def test_eval_components(self):
        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')
        model3 = models.ConstantModel(prefix='bkg_')
        mod = model1 + model2 + model3
        pars = mod.make_params()

        values1 = dict(amplitude=7.10, center=1.1, sigma=2.40)
        values2 = dict(amplitude=12.2, center=2.5, sigma=0.5)
        data = (1.01 + gaussian(x=self.x, **values1) +
                gaussian(x=self.x, **values2) + 0.05*self.noise)

        pars['g1_sigma'].set(2)
        pars['g1_center'].set(1, max=1.5)
        pars['g1_amplitude'].set(3)
        pars['g2_sigma'].set(1)
        pars['g2_center'].set(2.6, min=2.0)
        pars['g2_amplitude'].set(1)
        pars['bkg_c'].set(1.88)

        result = mod.fit(data, params=pars, x=self.x)

        self.assertTrue(abs(result.params['g1_amplitude'].value - 7.1) < 1.5)
        self.assertTrue(abs(result.params['g2_amplitude'].value - 12.2) < 1.5)
        self.assertTrue(abs(result.params['g1_center'].value - 1.1) < 0.2)
        self.assertTrue(abs(result.params['g2_center'].value - 2.5) < 0.2)
        self.assertTrue(abs(result.params['bkg_c'].value - 1.0) < 0.25)

        comps = mod.eval_components(x=self.x)
        assert 'bkg_' in comps

    def test_composite_has_bestvalues(self):
        # test that a composite model has non-empty best_values
        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')

        mod = model1 + model2
        pars = mod.make_params()

        values1 = dict(amplitude=7.10, center=1.1, sigma=2.40)
        values2 = dict(amplitude=12.2, center=2.5, sigma=0.5)
        data = (gaussian(x=self.x, **values1) + gaussian(x=self.x, **values2)
                + 0.1*self.noise)

        pars['g1_sigma'].set(value=2)
        pars['g1_center'].set(value=1, max=1.5)
        pars['g1_amplitude'].set(value=3)
        pars['g2_sigma'].set(value=1)
        pars['g2_center'].set(value=2.6, min=2.0)
        pars['g2_amplitude'].set(value=1)

        result = mod.fit(data, params=pars, x=self.x)

        self.assertTrue(len(result.best_values) == 6)

        self.assertTrue(abs(result.params['g1_amplitude'].value - 7.1) < 0.5)
        self.assertTrue(abs(result.params['g2_amplitude'].value - 12.2) < 0.5)
        self.assertTrue(abs(result.params['g1_center'].value - 1.1) < 0.2)
        self.assertTrue(abs(result.params['g2_center'].value - 2.5) < 0.2)

        for name, par in pars.items():
            assert len(repr(par)) > 5

    def test_composite_plotting(self):
        # test that a composite model has non-empty best_values
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use('Agg')

        model1 = models.GaussianModel(prefix='g1_')
        model2 = models.GaussianModel(prefix='g2_')

        mod = model1 + model2
        pars = mod.make_params()

        values1 = dict(amplitude=7.10, center=1.1, sigma=2.40)
        values2 = dict(amplitude=12.2, center=2.5, sigma=0.5)
        data = (gaussian(x=self.x, **values1) + gaussian(x=self.x, **values2)
                + 0.1*self.noise)

        pars['g1_sigma'].set(2)
        pars['g1_center'].set(1, max=1.5)
        pars['g1_amplitude'].set(3)
        pars['g2_sigma'].set(1)
        pars['g2_center'].set(2.6, min=2.0)
        pars['g2_amplitude'].set(1)

        result = mod.fit(data, params=pars, x=self.x)
        fig, ax = result.plot(show_init=True)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.GridSpec)

        comps = result.eval_components(x=self.x)
        assert len(comps) == 2
        assert 'g1_' in comps

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

    def test_hints_for_peakmodels(self):
        # test that height/fwhm do not cause asteval errors.

        x = np.linspace(-10, 10, 101)
        y = np.sin(x / 3) + x/100.

        m1 = models.LinearModel(prefix='m1_')
        params = m1.guess(y, x=x)

        m2 = models.GaussianModel(prefix='m2_')
        params.update(m2.make_params())

        m = m1 + m2  # noqa: F841

        param_values = {name: p.value for name, p in params.items()}
        self.assertTrue(param_values['m1_intercept'] < -0.0)
        self.assertEqual(param_values['m2_amplitude'], 1)

    def test_weird_param_hints(self):
        # tests Github Issue 312, a very weird way to access param_hints
        def func(x, amp):
            return amp*x

        m = Model(func)
        models = {}
        for i in range(2):
            m.set_param_hint('amp', value=1)
            m.set_param_hint('amp', value=25)

            models[i] = Model(func, prefix='mod%i_' % i)
            models[i].param_hints['amp'] = m.param_hints['amp']

        self.assertEqual(models[0].param_hints['amp'],
                         models[1].param_hints['amp'])

    def test_param_hint_explicit_value(self):
        # tests Github Issue 384
        pmod = PseudoVoigtModel()
        params = pmod.make_params(sigma=2, fraction=0.77)
        assert_allclose(params['fraction'].value, 0.77, rtol=0.01)

    def test_composite_model_with_expr_constrains(self):
        """Smoke test for composite model fitting with expr constraints."""
        y = [0, 0, 4, 2, 1, 8, 21, 21, 23, 35, 50, 54, 46, 70, 77, 87, 98,
             113, 148, 136, 185, 195, 194, 168, 170, 139, 155, 115, 132, 109,
             102, 85, 69, 81, 82, 80, 71, 64, 79, 88, 111, 97, 97, 73, 72, 62,
             41, 30, 13, 3, 9, 7, 0, 0, 0]
        x = np.arange(-0.2, 1.2, 0.025)[:-1] + 0.5*0.025

        def gauss(x, sigma, mu, A):
            return A*np.exp(-(x-mu)**2/(2*sigma**2))

        # Initial values
        p1_mu = 0.2
        p1_sigma = 0.1
        p2_sigma = 0.1

        peak1 = Model(gauss, prefix='p1_')
        peak2 = Model(gauss, prefix='p2_')
        model = peak1 + peak2

        model.set_param_hint('p1_mu', value=p1_mu, min=-1, max=2)
        model.set_param_hint('p1_sigma', value=p1_sigma, min=0.01, max=0.2)
        model.set_param_hint('p2_sigma', value=p2_sigma, min=0.01, max=0.2)
        model.set_param_hint('p1_A', value=100, min=0.01)
        model.set_param_hint('p2_A', value=50, min=0.01)

        # Constrains the distance between peaks to be > 0
        model.set_param_hint('pos_delta', value=0.3, min=0)
        model.set_param_hint('p2_mu', min=-1, expr='p1_mu + pos_delta')

        # Test fitting
        result = model.fit(y, x=x)
        self.assertTrue(result.params['pos_delta'].value > 0)

    def test_model_nan_policy(self):
        """Tests for nan_policy with NaN values in the input data."""
        x = np.linspace(0, 10, 201)
        np.random.seed(0)
        y = gaussian(x, 10.0, 6.15, 0.8)
        y += gaussian(x, 8.0, 6.35, 1.1)
        y += gaussian(x, 0.25, 6.00, 7.5)
        y += np.random.normal(size=len(x), scale=0.5)

        # with NaN values in the input data
        y[55] = y[91] = np.nan
        mod = PseudoVoigtModel()
        params = mod.make_params(amplitude=20, center=5.5,
                                 sigma=1, fraction=0.25)
        params['fraction'].vary = False

        # with raise, should get a ValueError
        result = lambda: mod.fit(y, params, x=x, nan_policy='raise')
        msg = ('NaN values detected in your input data or the output of your '
               'objective/model function - fitting algorithms cannot handle this!')
        self.assertRaisesRegex(ValueError, msg, result)

        # with propagate, should get no error, but bad results
        result = mod.fit(y, params, x=x, nan_policy='propagate')
        self.assertTrue(result.success)
        self.assertTrue(np.isnan(result.chisqr))
        self.assertTrue(np.isnan(result.aic))
        self.assertFalse(result.errorbars)
        self.assertTrue(result.params['amplitude'].stderr is None)
        self.assertTrue(abs(result.params['amplitude'].value - 20.0) < 0.001)

        # with omit, should get good results
        result = mod.fit(y, params, x=x, nan_policy='omit')
        self.assertTrue(result.success)
        self.assertTrue(result.chisqr > 2.0)
        self.assertTrue(result.aic < -100)
        self.assertTrue(result.errorbars)
        self.assertTrue(result.params['amplitude'].stderr > 0.1)
        self.assertTrue(abs(result.params['amplitude'].value - 20.0) < 5.0)
        self.assertTrue(abs(result.params['center'].value - 6.0) < 0.5)

        # with 'wrong_argument', should get a ValueError
        err_msg = r"nan_policy must be 'propagate', 'omit', or 'raise'."
        with pytest.raises(ValueError, match=err_msg):
            mod.fit(y, params, x=x, nan_policy='wrong_argument')

    def test_model_nan_policy_NaNs_by_model(self):
        """Test for nan_policy with NaN values generated by the model function."""
        def double_exp(x, a1, t1, a2, t2):
            return a1*np.exp(-x/t1) + a2*np.exp(-(x-0.1) / t2)

        model = Model(double_exp)

        truths = (3.0, 2.0, -5.0, 10.0)
        x = np.linspace(1, 10, 250)
        np.random.seed(0)
        y = double_exp(x, *truths) + 0.1*np.random.randn(x.size)

        p = model.make_params(a1=4, t1=3, a2=4, t2=3)
        result = lambda: model.fit(data=y, params=p, x=x, method='Nelder',
                                   nan_policy='raise')

        msg = 'The model function generated NaN values and the fit aborted!'
        self.assertRaisesRegex(ValueError, msg, result)

    @pytest.mark.skipif(sys.version_info.major == 2,
                        reason="cannot use wrapped functions with Python 2")
    def test_wrapped_model_func(self):
        x = np.linspace(-1, 1, 51)
        y = 2.0*x + 3 + 0.0003 * x*x
        y += np.random.normal(size=len(x), scale=0.025)
        mod = Model(linear_func)
        pars = mod.make_params(a=1.5, b=2.5)

        tmp = mod.eval(pars, x=x)

        self.assertTrue(tmp.max() > 3)
        self.assertTrue(tmp.min() > -20)

        result = mod.fit(y, pars, x=x)
        self.assertTrue(result.chisqr < 0.05)
        self.assertTrue(result.aic < -350)
        self.assertTrue(result.errorbars)

        self.assertTrue(abs(result.params['a'].value - 2.0) < 0.05)
        self.assertTrue(abs(result.params['b'].value - 3.0) < 0.41)


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
        raise pytest.skip("ConstantModel has not independent_vars.")


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


class TestComplexConstant(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(re=5, im=5)
        self.guess = lambda: dict(re=2, im=2)
        self.model_constructor = models.ComplexConstantModel
        super(TestComplexConstant, self).setUp()


class TestExpression(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(off_c=0.25, amp_c=1.0, x0=2.0)
        self.guess = lambda: dict(off_c=0.20, amp_c=1.5, x0=2.5)
        self.expression = "off_c + amp_c * exp(-x/x0)"
        self.model_constructor = (
            lambda *args, **kwargs: models.ExpressionModel(self.expression, *args, **kwargs))
        super(TestExpression, self).setUp()

    def test_composite_with_expression(self):
        expression_model = models.ExpressionModel("exp(-x/x0)", name='exp')
        amp_model = models.ConstantModel(prefix='amp_')
        off_model = models.ConstantModel(prefix='off_', name="off")

        comp_model = off_model + amp_model * expression_model

        x = self.x
        true_values = self.true_values()
        data = comp_model.eval(x=x, **true_values) + self.noise
        # data = 0.25 + 1 * np.exp(-x / 2.)

        params = comp_model.make_params(**self.guess())

        result = comp_model.fit(data, x=x, params=params)
        assert_results_close(result.values, true_values, rtol=0.01, atol=0.01)

        data_components = comp_model.eval_components(x=x)
        self.assertIn('exp', data_components)
#
