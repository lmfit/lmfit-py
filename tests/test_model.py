"""Tests for the Model, CompositeModel, and ModelResult classes."""

import functools
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import __version__ as scipy_version

import lmfit
from lmfit import Model, Parameters, models
from lmfit.lineshapes import gaussian, lorentzian
from lmfit.model import get_reducer, propagate_err
from lmfit.models import GaussianModel, PseudoVoigtModel


@pytest.fixture()
def gmodel():
    """Return a Gaussian model."""
    return Model(gaussian)


def test_get_reducer_invalid_option():
    """Tests for ValueError when using an unsupported option."""
    option = 'unknown'
    msg = r'Invalid option'
    with pytest.raises(ValueError, match=msg):
        get_reducer(option)


test_data_get_reducer = [('real', [1.0, 1.0, 2.0, 2.0]),
                         ('imag', [0.0, 10.0, 0.0, 20.0]),
                         ('abs', [1.0, 10.04987562, 2.0, 20.09975124]),
                         ('angle', [0.0, 1.47112767, 0.0, 1.471127670])]


@pytest.mark.parametrize('option, expected_array', test_data_get_reducer)
def test_get_reducer(option, expected_array):
    """Tests for ValueError when using an unsupported option."""
    complex_array = np.array([1.0, 1.0+10j, 2.0, 2.0+20j], dtype='complex')
    func = get_reducer(option)
    real_array = func(complex_array)

    assert np.all(np.isreal(real_array))
    assert_allclose(real_array, expected_array)

    # nothing should happen to an array that only contains real data
    assert_allclose(func(real_array), real_array)


def test_propagate_err_invalid_option():
    """Tests for ValueError when using an unsupported option."""
    z = np.array([0, 1, 2, 3, 4, 5])
    dz = np.random.normal(size=z.size, scale=0.1)
    option = 'unknown'
    msg = r'Invalid option'
    with pytest.raises(ValueError, match=msg):
        propagate_err(z, dz, option)


def test_propagate_err_unequal_shape_z_dz():
    """Tests for ValueError when using unequal arrays for z and dz."""
    z = np.array([0, 1, 2, 3, 4, 5])
    dz = np.random.normal(size=z.size-1, scale=0.1)
    msg = r'shape of z:'
    with pytest.raises(ValueError, match=msg):
        propagate_err(z, dz, option='abs')


@pytest.mark.parametrize('option', ['real', 'imag', 'abs', 'angle'])
def test_propagate_err(option):
    """Tests for ValueError when using an unsupported option."""
    np.random.seed(2020)
    z = np.array([1.0, 1.0+10j, 2.0, 2.0+20j], dtype='complex')
    dz = np.random.normal(z.size, scale=0.1)*z

    # if `z` is real, assume that `dz` is also real and return it as-is
    err = propagate_err(np.real(z), np.real(dz), option)
    assert_allclose(err, np.real(dz))

    # if `z` is complex, but `dz` is real apply the err to both real/imag
    err_complex_real = propagate_err(z, np.real(dz), option)
    assert np.all(np.isreal(err_complex_real))
    dz_used = np.real(dz)+1j*np.real(dz)
    if option == 'real':
        assert_allclose(err_complex_real, np.real(dz_used))
    elif option == 'imag':
        assert_allclose(err_complex_real, np.imag(dz_used))
    elif option == 'abs':
        assert_allclose(err_complex_real,
                        [3.823115, 3.823115, 7.646231, 7.646231],
                        rtol=1.0e-5)
    elif option == 'angle':
        assert_allclose(err_complex_real,
                        [3.823115, 0.380414, 3.823115, 0.380414],
                        rtol=1.0e-5)

    # both `z` and `dz` are complex
    err_complex_complex = propagate_err(z, dz, option)
    assert np.all(np.isreal(err_complex_complex))
    if option == 'real':
        assert_allclose(err_complex_complex, np.real(dz))
    elif option == 'imag':
        assert_allclose(err_complex_complex, np.imag(dz))
    elif option == 'abs':
        assert_allclose(err_complex_complex,
                        [3.823115, 38.043322, 7.646231, 76.086645],
                        rtol=1.0e-5)
    elif option == 'angle':
        assert_allclose(err_complex_complex, [0., 0.535317, 0., 0.535317],
                        rtol=1.0e-5)


def test_initialize_Model_class_default_arguments(gmodel):
    """Test for Model class initialized with default arguments."""
    assert gmodel.prefix == ''
    assert gmodel._param_root_names == ['amplitude', 'center', 'sigma']
    assert gmodel.param_names == ['amplitude', 'center', 'sigma']
    assert gmodel.independent_vars == ['x']
    assert gmodel.nan_policy == 'raise'
    assert gmodel.name == 'Model(gaussian)'
    assert gmodel.opts == {}
    assert gmodel.def_vals == {'amplitude': 1.0, 'center': 0.0, 'sigma': 1.0}


def test_initialize_Model_class_independent_vars():
    """Test for Model class initialized with independent_vars."""
    model = Model(gaussian, independent_vars=['amplitude'])
    assert model._param_root_names == ['x', 'center', 'sigma']
    assert model.param_names == ['x', 'center', 'sigma']
    assert model.independent_vars == ['amplitude']


def test_initialize_Model_class_param_names():
    """Test for Model class initialized with param_names."""
    model = Model(gaussian, param_names=['amplitude'])

    assert model._param_root_names == ['amplitude']
    assert model.param_names == ['amplitude']


@pytest.mark.parametrize("policy", ['raise', 'omit', 'propagate'])
def test_initialize_Model_class_nan_policy(policy):
    """Test for Model class initialized with nan_policy."""
    model = Model(gaussian, nan_policy=policy)

    assert model.nan_policy == policy


def test_initialize_Model_class_prefix():
    """Test for Model class initialized with prefix."""
    model = Model(gaussian, prefix='test_')

    assert model.prefix == 'test_'
    assert model._param_root_names == ['amplitude', 'center', 'sigma']
    assert model.param_names == ['test_amplitude', 'test_center', 'test_sigma']
    assert model.name == "Model(gaussian, prefix='test_')"

    model = Model(gaussian, prefix=None)

    assert model.prefix == ''


def test_initialize_Model_name():
    """Test for Model class initialized with name."""
    model = Model(gaussian, name='test_function')

    assert model.name == 'Model(test_function)'


def test_initialize_Model_kws():
    """Test for Model class initialized with **kws."""
    kws = {'amplitude': 10.0}
    model = Model(gaussian,
                  independent_vars=['x', 'amplitude'], **kws)

    assert model._param_root_names == ['center', 'sigma']
    assert model.param_names == ['center', 'sigma']
    assert model.independent_vars == ['x', 'amplitude']
    assert model.opts == kws


test_reprstring_data = [(False, 'Model(gaussian)'),
                        (True, "Model(gaussian, amplitude='10.0')")]


@pytest.mark.parametrize("option, expected", test_reprstring_data)
def test_Model_reprstring(option, expected):
    """Test for Model class function _reprstring."""
    kws = {'amplitude': 10.0}
    model = Model(gaussian,
                  independent_vars=['x', 'amplitude'], **kws)

    assert model._reprstring(option) == expected


def test_Model_get_state(gmodel):
    """Test for Model class function _get_state."""
    out = gmodel._get_state()

    assert isinstance(out, tuple)
    assert out[1] == out[2] is None
    assert (out[0][1] is not None) == lmfit.jsonutils.HAS_DILL

    assert out[0][0] == 'gaussian'
    assert out[0][2:] == ('gaussian', '', ['x'],
                          ['amplitude', 'center', 'sigma'], {}, 'raise', {})


def test_Model_set_state(gmodel):
    """Test for Model class function _set_state.

    This function is just calling `_buildmodel`, which will be tested
    below together with the use of `funcdefs`.

    """
    out = gmodel._get_state()

    new_model = Model(lorentzian)
    new_model = new_model._set_state(out)

    assert new_model.prefix == gmodel.prefix
    assert new_model._param_root_names == gmodel._param_root_names
    assert new_model.param_names == gmodel.param_names
    assert new_model.independent_vars == gmodel.independent_vars
    assert new_model.nan_policy == gmodel.nan_policy
    assert new_model.name == gmodel.name
    assert new_model.opts == gmodel.opts


def test_Model_dumps_loads(gmodel):
    """Test for Model class functions dumps and loads.

    These function are used when saving/loading the Model class and will be
    tested more thoroughly in test_model_saveload.py.

    """
    model_json = gmodel.dumps()
    _ = gmodel.loads(model_json)


def test_Model_getter_setter_name(gmodel):
    """Test for Model class getter/setter functions for name."""
    assert gmodel.name == 'Model(gaussian)'

    gmodel.name = 'test_gaussian'
    assert gmodel.name == 'Model(test_gaussian)'


def test_Model_getter_setter_prefix(gmodel):
    """Test for Model class getter/setter functions for prefix."""
    assert gmodel.prefix == ''
    assert gmodel.param_names == ['amplitude', 'center', 'sigma']

    gmodel.prefix = 'g1_'
    assert gmodel.prefix == 'g1_'
    assert gmodel.param_names == ['g1_amplitude', 'g1_center', 'g1_sigma']

    gmodel.prefix = ''
    assert gmodel.prefix == ''
    assert gmodel.param_names == ['amplitude', 'center', 'sigma']


def test_Model_getter_param_names(gmodel):
    """Test for Model class getter function for param_names."""
    assert gmodel.param_names == ['amplitude', 'center', 'sigma']


def test_Model__repr__(gmodel):
    """Test for Model class __repr__ method."""
    assert gmodel.__repr__() == '<lmfit.Model: Model(gaussian)>'


def test_Model_copy(gmodel):
    """Test for Model class copy method."""
    msg = 'Model.copy does not work. Make a new Model'
    with pytest.raises(NotImplementedError, match=msg):
        gmodel.copy()


def test__parse_params_func_None():
    """Test for _parse_params function with func=None."""
    mod = Model(None)

    assert mod._prefix == ''
    assert mod.func is None
    assert mod._func_allargs == []
    assert mod._func_haskeywords is False
    assert mod.independent_vars == []


def test__parse_params_asteval_functions():
    """Test for _parse_params function with asteval functions."""
    # TODO: cannot find a use-case for this....
    pass


def test__parse_params_inspect_signature():
    """Test for _parse_params function using inspect.signature."""
    # 1. function with a positional argument
    def func_var_positional(a, *b):
        pass

    with pytest.raises(ValueError, match=r"varargs '\*b' is not supported"):
        Model(func_var_positional)

    # 2. function with a keyword argument
    def func_keyword(a, b, **c):
        pass

    mod = Model(func_keyword)
    assert mod._func_allargs == ['a', 'b']
    assert mod._func_haskeywords is True
    assert mod.independent_vars == ['a']
    assert mod.def_vals == {}

    # 3. function with keyword argument only
    def func_keyword_only(**b):
        pass

    mod = Model(func_keyword_only)
    assert mod._func_allargs == []
    assert mod._func_haskeywords is True
    assert mod.independent_vars == []
    assert mod._param_root_names is None

    # 4. function with default value
    def func_default_value(a, b, c=10):
        pass

    mod = Model(func_default_value)
    assert mod._func_allargs == ['a', 'b', 'c']
    assert mod._func_haskeywords is False
    assert mod.independent_vars == ['a']

    assert isinstance(mod.def_vals, dict)
    assert_allclose(mod.def_vals['c'], 10)


def test_make_params_withprefixs():
    # tests Github Issue #893
    gmod1 = GaussianModel(prefix='p1_')
    gmod2 = GaussianModel(prefix='p2_')

    model = gmod1 + gmod2

    pars_1a = gmod1.make_params(p1_amplitude=10, p1_center=600, p1_sigma=3)
    pars_1b = gmod1.make_params(amplitude=10, center=600, sigma=3)

    pars_2a = gmod2.make_params(p2_amplitude=30, p2_center=730, p2_sigma=4)
    pars_2b = gmod2.make_params(amplitude=30, center=730, sigma=4)

    pars_a = Parameters()
    pars_a.update(pars_1a)
    pars_a.update(pars_2a)

    pars_b = Parameters()
    pars_b.update(pars_1b)
    pars_b.update(pars_2b)

    pars_c = model.make_params()

    for pname in ('p1_amplitude', 'p1_center', 'p1_sigma',
                  'p2_amplitude', 'p2_center', 'p2_sigma'):
        assert pname in pars_a
        assert pname in pars_b
        assert pname in pars_c


def test__parse_params_forbidden_variable_names():
    """Tests for _parse_params function using invalid variable names."""

    def func_invalid_var(data, a):
        pass

    def func_invalid_par(a, weights):
        pass

    msg = r"Invalid independent variable name \('data'\) for function func_invalid_var"
    with pytest.raises(ValueError, match=msg):
        Model(func_invalid_var)

    msg = r"Invalid parameter name \('weights'\) for function func_invalid_par"
    with pytest.raises(ValueError, match=msg):
        Model(func_invalid_par)


@pytest.mark.parametrize('input_dtype', (np.int16, np.int32, np.float32,
                                         np.complex64, np.complex128, 'list',
                                         'tuple', 'pandas-real',
                                         'pandas-complex'))
def test_coercion_of_input_data(peakdata, input_dtype):
    """Test for coercion of 'data' and 'independent_vars'.

    'data' and `independent_vars` should be coerced to 'float64' or 'complex128'

    unless told not be coerced by setting ``coerce_farray=False``.

    # - dtype for 'indepdendent_vars' is only changed when the input is a list,
    #    tuple, numpy.ndarray, or pandas.Series

    """
    x, y = peakdata

    def gaussian_lists(x, amplitude=1.0, center=0.0, sigma=1.0):
        xarr = np.array(x, dtype=np.float64)
        return ((amplitude/(max(1.e-15, np.sqrt(2*np.pi)*sigma)))
                * np.exp(-(xarr-center)**2 / max(1.e-15, (2*sigma**2))))

    for coerce_farray in True, False:
        if (input_dtype in ('pandas-real', 'pandas-complex')
           and not lmfit.minimizer.HAS_PANDAS):
            return

        if not coerce_farray and input_dtype in ('list', 'tuple'):
            model = lmfit.Model(gaussian_lists)
        else:
            model = lmfit.Model(gaussian)

        pars = model.make_params(amplitude=5, center=10, sigma=2)

        if input_dtype == 'pandas-real':
            result = model.fit(lmfit.model.Series(y, dtype=np.float32), pars,
                               x=lmfit.model.Series(x, dtype=np.float32),
                               coerce_farray=coerce_farray)

            expected_dtype = np.float64 if coerce_farray else np.float32

        elif input_dtype == 'pandas-complex':
            result = model.fit(lmfit.model.Series(y, dtype=np.complex64), pars,
                               x=lmfit.model.Series(x, dtype=np.complex64),
                               coerce_farray=coerce_farray)
            expected_dtype = np.complex128 if coerce_farray else np.complex64

        elif input_dtype == 'list':
            result = model.fit(y.tolist(), pars, x=x.tolist(),
                               coerce_farray=coerce_farray)
            expected_dtype = np.float64 if coerce_farray else list

        elif input_dtype == 'tuple':
            result = model.fit(tuple(y), pars, x=tuple(x),
                               coerce_farray=coerce_farray)
            expected_dtype = np.float64 if coerce_farray else tuple

        else:
            result = model.fit(np.asarray(y, dtype=input_dtype), pars,
                               x=np.asarray(x, dtype=input_dtype),
                               coerce_farray=coerce_farray)
            expected_dtype = np.float64
            if input_dtype in (np.complex64, np.complex128):
                expected_dtype = np.complex128
            expected_dtype = expected_dtype if coerce_farray else input_dtype

        if not coerce_farray and input_dtype in ('list', 'tuple'):
            assert isinstance(result.userkws['x'], (list, tuple))
            assert isinstance(result.userargs[0], (list, tuple))
        else:
            assert result.userkws['x'].dtype == expected_dtype
            assert result.userargs[0].dtype == expected_dtype


def test_figure_default_title(peakdata):
    """Test default figure title."""
    pytest.importorskip('matplotlib')

    x, y = peakdata
    pvmodel = PseudoVoigtModel()
    params = pvmodel.guess(y, x=x)
    result = pvmodel.fit(y, params, x=x)

    ax = result.plot_fit()
    assert ax.axes.get_title() == 'Model(pvoigt)'

    ax = result.plot_residuals()
    assert ax.axes.get_title() == 'Model(pvoigt)'

    fig = result.plot()
    assert fig.axes[0].get_title() == 'Model(pvoigt)'  # default model.name
    assert fig.axes[1].get_title() == ''  # no title for fit subplot


def test_figure_title_using_title_keyword_argument(peakdata):
    """Test setting figure title using title keyword argument."""
    pytest.importorskip('matplotlib')

    x, y = peakdata
    pvmodel = PseudoVoigtModel()
    params = pvmodel.guess(y, x=x)
    result = pvmodel.fit(y, params, x=x)

    ax = result.plot_fit(title='test')
    assert ax.axes.get_title() == 'test'

    ax = result.plot_residuals(title='test')
    assert ax.axes.get_title() == 'test'

    fig = result.plot(title='test')
    assert fig.axes[0].get_title() == 'test'
    assert fig.axes[1].get_title() == ''  # no title for fit subplot


def test_figure_title_using_title_to_ax_kws(peakdata):
    """Test setting figure title by supplying ax_{fit,res}_kws."""
    pytest.importorskip('matplotlib')

    x, y = peakdata
    pvmodel = PseudoVoigtModel()
    params = pvmodel.guess(y, x=x)
    result = pvmodel.fit(y, params, x=x)

    ax = result.plot_fit(ax_kws={'title': 'ax_kws'})
    assert ax.axes.get_title() == 'ax_kws'

    ax = result.plot_residuals(ax_kws={'title': 'ax_kws'})
    assert ax.axes.get_title() == 'ax_kws'

    fig = result.plot(ax_res_kws={'title': 'ax_res_kws'})
    assert fig.axes[0].get_title() == 'ax_res_kws'
    assert fig.axes[1].get_title() == ''

    fig = result.plot(ax_fit_kws={'title': 'ax_fit_kws'})
    assert fig.axes[0].get_title() == 'Model(pvoigt)'  # default model.name
    assert fig.axes[1].get_title() == ''  # no title for fit subplot


def test_priority_setting_figure_title(peakdata):
    """Test for setting figure title: title keyword argument has priority."""
    pytest.importorskip('matplotlib')

    x, y = peakdata
    pvmodel = PseudoVoigtModel()
    params = pvmodel.guess(y, x=x)
    result = pvmodel.fit(y, params, x=x)

    ax = result.plot_fit(ax_kws={'title': 'ax_kws'}, title='test')
    assert ax.axes.get_title() == 'test'

    ax = result.plot_residuals(ax_kws={'title': 'ax_kws'}, title='test')
    assert ax.axes.get_title() == 'test'

    fig = result.plot(ax_res_kws={'title': 'ax_res_kws'}, title='test')
    assert fig.axes[0].get_title() == 'test'
    assert fig.axes[1].get_title() == ''

    fig = result.plot(ax_fit_kws={'title': 'ax_fit_kws'}, title='test')
    assert fig.axes[0].get_title() == 'test'
    assert fig.axes[1].get_title() == ''


def test_eval_with_kwargs():
    # Check eval() with both params and kwargs, even when there are
    # constraints
    x = np.linspace(0, 30, 301)
    np.random.seed(13)
    y1 = (gaussian(x, amplitude=10, center=12.0, sigma=2.5) +
          gaussian(x, amplitude=20, center=19.0, sigma=2.5))

    y2 = (gaussian(x, amplitude=10, center=12.0, sigma=1.5) +
          gaussian(x, amplitude=20, center=19.0, sigma=2.5))

    model = Model(gaussian, prefix='g1_') + Model(gaussian, prefix='g2_')
    params = model.make_params(g1_amplitude=10, g1_center=12.0, g1_sigma=1,
                               g2_amplitude=20, g2_center=19.0,
                               g2_sigma={'expr': 'g1_sigma'})

    r1 = model.eval(params, g1_sigma=2.5, x=x)
    assert_allclose(r1, y1, atol=1.e-3)

    assert params['g2_sigma'].value == 1
    assert params['g1_sigma'].value == 1

    params['g1_sigma'].value = 1.5
    params['g2_sigma'].expr = None
    params['g2_sigma'].value = 2.5

    r2 = model.eval(params, x=x)
    assert_allclose(r2, y2, atol=1.e-3)


def test_guess_requires_x():
    """Test to make sure that ``guess()`` method requires the argument ``x``.

    The ``guess`` method needs ``x`` values (i.e., the independent variable)
    to estimate initial parameters, but this was not a required argument.
    See GH #747.

    """
    mod = lmfit.model.Model(gaussian)

    msg = r"guess\(\) missing 2 required positional arguments: 'data' and 'x'"
    with pytest.raises(TypeError, match=msg):
        mod.guess()


# Below is the content of the original test_model.py file. These tests still
# need to be checked and possibly updated to the pytest-style. They work fine
# though so leave them in for now.

def assert_results_close(actual, desired, rtol=1e-03, atol=1e-03, err_msg='',
                         verbose=True):
    for param_name, value in desired.items():
        assert_allclose(actual[param_name], value, rtol, atol, err_msg,
                        verbose)


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


class CommonTests:
    # to be subclassed for testing predefined models

    def setUp(self):
        np.random.seed(1)
        self.noise = 0.0001*np.random.randn(self.x.size)
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

        # Pass individual Parameter objects as kwargs.
        kwargs = dict(params.items())
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
        pytest.importorskip('pandas')

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
    def setUp(self):
        self.true_values = lambda: dict(amplitude=7.1, center=1.1, sigma=2.40)
        self.guess = lambda: dict(amplitude=5, center=2, sigma=4)
        # return a fresh copy
        self.model_constructor = (
            lambda *args, **kwargs: Model(gaussian, *args, **kwargs))
        super().setUp()

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
        except Exception:
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

        for _, par in pars.items():
            assert len(repr(par)) > 5

    @pytest.mark.skipif(not lmfit.model._HAS_MATPLOTLIB,
                        reason="requires matplotlib.pyplot")
    def test_composite_plotting(self):
        # test that a composite model has non-empty best_values
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
        fig = result.plot(show_init=True)

        assert isinstance(fig, matplotlib.figure.Figure)

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

        _m = m1 + m2  # noqa: F841

        param_values = {name: p.value for name, p in params.items()}
        assert_almost_equal(param_values['m1_intercept'], 0.)
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

            models[i] = Model(func, prefix=f'mod{i}')
            models[i].param_hints['amp'] = m.param_hints['amp']

        self.assertEqual(models[0].param_hints['amp'],
                         models[1].param_hints['amp'])

    def test_param_hint_explicit_value(self):
        # tests Github Issue 384
        pmod = PseudoVoigtModel()
        params = pmod.make_params(sigma=2, fraction=0.77)
        assert_allclose(params['fraction'].value, 0.77, rtol=0.01)

    def test_symmetric_boundss(self):
        # tests Github Issue 700
        np.random.seed(0)

        x = np.linspace(0, 20, 51)
        y = gaussian(x, amplitude=8.0, center=13, sigma=2.5)
        y += np.random.normal(size=len(x), scale=0.1)

        mod = Model(gaussian)
        params = mod.make_params(sigma=2.2, center=10, amplitude=10)
        # carefully selected to have inexact floating-point representation
        params['sigma'].min = 2.2 - 0.95
        params['sigma'].max = 2.2 + 0.95

        result = mod.fit(y, params, x=x)
        print(result.fit_report())
        self.assertTrue(result.params['sigma'].value > 2.3)
        self.assertTrue(result.params['sigma'].value < 2.7)
        self.assertTrue(result.params['sigma'].stderr is not None)
        self.assertTrue(result.params['amplitude'].stderr is not None)
        self.assertTrue(result.params['sigma'].stderr > 0.02)
        self.assertTrue(result.params['sigma'].stderr < 0.50)

    def test_unprefixed_name_collisions(self):
        # tests Github Issue 710
        np.random.seed(0)
        x = np.linspace(0, 20, 201)
        y = 6 + x * 0.55 + gaussian(x, 4.5, 8.5, 2.1) + np.random.normal(size=len(x), scale=0.03)

        def myline(x, a, b):
            return a + b * x

        def mygauss(x, a, b, c):
            return gaussian(x, a, b, c)

        mod = Model(myline, prefix='line_') + Model(mygauss, prefix='peak_')
        pars = mod.make_params(line_a=5, line_b=1, peak_a=10, peak_b=10, peak_c=5)
        pars.add('a', expr='line_a + peak_a')

        result = mod.fit(y, pars, x=x)
        self.assertTrue(result.params['peak_a'].value > 4)
        self.assertTrue(result.params['peak_a'].value < 5)
        self.assertTrue(result.params['peak_b'].value > 8)
        self.assertTrue(result.params['peak_b'].value < 9)
        self.assertTrue(result.params['peak_c'].value > 1.5)
        self.assertTrue(result.params['peak_c'].value < 2.5)
        self.assertTrue(result.params['line_a'].value > 5.5)
        self.assertTrue(result.params['line_a'].value < 6.5)
        self.assertTrue(result.params['line_b'].value > 0.25)
        self.assertTrue(result.params['line_b'].value < 0.75)
        self.assertTrue(result.params['a'].value > 10)
        self.assertTrue(result.params['a'].value < 11)

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

        # for SciPy v1.10+ this results in an AbortFitException, even with
        # `max_nfev=100000`:
        #   lmfit.minimizer.AbortFitException: fit aborted: too many function
        #   evaluations xxxxx
        if int(scipy_version.split('.')[1]) < 10:
            self.assertTrue(np.isnan(result.chisqr))
            self.assertTrue(np.isnan(result.aic))
            self.assertFalse(result.errorbars)
            self.assertTrue(result.params['amplitude'].stderr is None)
            self.assertTrue(abs(result.params['amplitude'].value - 20.0) < 0.001)
        else:
            pass

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

    def test_different_independent_vars_composite_modeld(self):
        """Regression test for different independent variables in CompositeModel.

        See: https://github.com/lmfit/lmfit-py/discussions/787

        """
        def two_independent_vars(y, z, a):
            return a * y + z

        BackgroundModel = Model(two_independent_vars,
                                independent_vars=["y", "z"], prefix="yz_")
        PeakModel = Model(gaussian, independent_vars=["x"], prefix="x_")
        CompModel = BackgroundModel + PeakModel
        assert CompModel.independent_vars == ['x', 'y', 'z']


class TestLinear(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(slope=5, intercept=2)
        self.guess = lambda: dict(slope=10, intercept=6)
        self.model_constructor = models.LinearModel
        super().setUp()


class TestParabolic(CommonTests, unittest.TestCase):

    def setUp(self):
        self.true_values = lambda: dict(a=5, b=2, c=8)
        self.guess = lambda: dict(a=1, b=6, c=3)
        self.model_constructor = models.ParabolicModel
        super().setUp()


class TestPolynomialOrder2(CommonTests, unittest.TestCase):
    # class Polynomial constructed with order=2
    def setUp(self):
        self.true_values = lambda: dict(c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c1=1, c2=6, c0=3)
        self.model_constructor = models.PolynomialModel
        self.args = (2,)
        super().setUp()


class TestPolynomialOrder3(CommonTests, unittest.TestCase):
    # class Polynomial constructed with order=3
    def setUp(self):
        self.true_values = lambda: dict(c3=2, c2=5, c1=2, c0=8)
        self.guess = lambda: dict(c3=1, c1=1, c2=6, c0=3)
        self.model_constructor = models.PolynomialModel
        self.args = (3,)
        super().setUp()


class TestConstant(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(c=5)
        self.guess = lambda: dict(c=2)
        self.model_constructor = models.ConstantModel
        super().setUp()

    def check_skip_independent_vars(self):
        raise pytest.skip("ConstantModel has not independent_vars.")


class TestPowerlaw(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(amplitude=5, exponent=3)
        self.guess = lambda: dict(amplitude=2, exponent=8)
        self.model_constructor = models.PowerLawModel
        super().setUp()


class TestExponential(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(amplitude=5, decay=3)
        self.guess = lambda: dict(amplitude=2, decay=8)
        self.model_constructor = models.ExponentialModel
        super().setUp()


class TestComplexConstant(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(re=5, im=5)
        self.guess = lambda: dict(re=2, im=2)
        self.model_constructor = models.ComplexConstantModel
        super().setUp()


class TestExpression(CommonTests, unittest.TestCase):
    def setUp(self):
        self.true_values = lambda: dict(off_c=0.25, amp_c=1.0, x0=2.0)
        self.guess = lambda: dict(off_c=0.20, amp_c=1.5, x0=2.5)
        self.expression = "off_c + amp_c * exp(-x/x0)"
        self.model_constructor = (
            lambda *args, **kwargs: models.ExpressionModel(self.expression, *args, **kwargs))
        super().setUp()

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


def test_make_params_valuetypes():
    mod = lmfit.models.SineModel()

    pars = mod.make_params(amplitude=1, frequency=1, shift=-0.2)

    pars = mod.make_params(amplitude={'value': 0.9, 'min': 0},
                           frequency=1.03,
                           shift={'value': -0.2, 'vary': False})

    val_i32 = np.arange(10, dtype=np.int32)
    val_i64 = np.arange(10, dtype=np.int64)
    # np.longdouble equals to np.float128 on Linux and macOS, np.float64 on Windows
    val_ld = np.arange(10, dtype=np.longdouble)/3.0
    val_c128 = np.arange(10, dtype=np.complex128)/3.0

    pars = mod.make_params(amplitude=val_i64[2],
                           frequency=val_i32[3],
                           shift=-val_ld[4])

    pars = mod.make_params(amplitude=val_c128[2],
                           frequency=val_i32[3],
                           shift=-val_ld[4])

    assert pars is not None
    with pytest.raises(ValueError):
        pars = mod.make_params(amplitude='a string', frequency=2, shift=7)

    with pytest.raises(TypeError):
        pars = mod.make_params(amplitude={'v': 3}, frequency=2, shift=7)

    with pytest.raises(TypeError):
        pars = mod.make_params(amplitude={}, frequency=2, shift=7)


def test_complex_model_eval_uncertainty():
    """Github #900"""
    def cmplx(f, omega, areal, aimag, off, sigma):
        return (areal*np.cos(f*omega + off) + 1j*aimag*np.sin(f*omega + off))*np.exp(-f/sigma)

    f = np.linspace(0, 10, 501)
    dat = cmplx(f, 4, 10, 5, 0.2, 4.5) + (0.1 + 0.2j)*np.random.normal(scale=0.25, size=len(f))
    mod = Model(cmplx)
    params = mod.make_params(omega=5, areal=5, aimag=5,
                             off={'value': 0.5, 'min': -2, 'max': 2},
                             sigma={'value': 3, 'min': 1.e-5, 'max': 1000})

    result = mod.fit(dat, params=params, f=f)
    dfit = result.eval_uncertainty()
    assert len(dfit) == len(f)
    assert dfit.dtype == 'complex128'


def test_compositemodel_returning_list():
    """Github #875"""
    def lin1(x, k):
        return [k*x1 for x1 in x]

    def lin2(x, k):
        return [k*x1 for x1 in x]

    y = np.linspace(0, 100, 100)
    x = np.linspace(0, 100, 100)

    Model1 = Model(lin1, independent_vars=["x"], prefix="m1_")
    Model2 = Model(lin2, independent_vars=["x"], prefix="m2_")
    ModelSum = Model1 + Model2
    pars = Parameters()
    pars.add('m1_k', value=0.5)
    pars.add('m2_k', value=0.5)
    result = ModelSum.fit(y, pars, x=x)
    assert len(result.best_fit) == len(x)
