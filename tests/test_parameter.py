"""Tests for the Parameter class."""

from math import trunc
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

import lmfit


@pytest.fixture
def parameters():
    """Initialize a Parameters class for tests."""
    pars = lmfit.Parameters()
    pars.add(lmfit.Parameter(name='a', value=10.0, vary=True, min=-100.0,
                             max=100.0, expr=None, brute_step=5.0,
                             user_data=1))
    pars.add(lmfit.Parameter(name='b', value=0.0, vary=True, min=-250.0,
                             max=250.0, expr="2.0*a", brute_step=25.0,
                             user_data=2.5))
    exp_attr_values_A = ('a', 10.0, True, -100.0, 100.0, None, 5.0, 1)
    exp_attr_values_B = ('b', 20.0, False, -250.0, 250.0, "2.0*a", 25.0, 2.5)
    assert_parameter_attributes(pars['a'], exp_attr_values_A)
    assert_parameter_attributes(pars['b'], exp_attr_values_B)
    return pars, exp_attr_values_A, exp_attr_values_B


@pytest.fixture
def parameter():
    """Initialize parameter for tests."""
    param = lmfit.Parameter(name='a', value=10.0, vary=True, min=-100.0,
                            max=100.0, expr=None, brute_step=5.0, user_data=1)
    expected_attribute_values = ('a', 10.0, True, -100.0, 100.0, None, 5.0, 1)
    assert_parameter_attributes(param, expected_attribute_values)
    return param, expected_attribute_values


def assert_parameter_attributes(par, expected):
    """Assert that parameter attributes have the expected values."""
    par_attr_values = (par.name, par._val, par.vary, par.min, par.max,
                       par._expr, par.brute_step, par.user_data)
    assert par_attr_values == expected


in_out = [(lmfit.Parameter(name='a'),  # set name
           ('a', -np.inf, True, -np.inf, np.inf, None, None, None)),
          (lmfit.Parameter(name='a', value=10.0),  # set value
           ('a', 10.0, True, -np.inf, np.inf, None, None, None)),
          (lmfit.Parameter(name='a', vary=False),  # fix parameter, set vary to False
           ('a', -np.inf, False, -np.inf, np.inf, None, None, None)),
          (lmfit.Parameter(name='a', min=-10.0),  # set lower bound, value reset to min
           ('a', -10.0, True, -10.0, np.inf, None, None, None)),
          (lmfit.Parameter(name='a', value=-5.0, min=-10.0),  # set lower bound
           ('a', -5.0, True, -10.0, np.inf, None, None, None)),
          (lmfit.Parameter(name='a', max=10.0),  # set upper bound
           ('a', -np.inf, True, -np.inf, 10.0, None, None, None)),
          (lmfit.Parameter(name='a', value=25.0, max=10.0),  # set upper bound, value reset
           ('a', 10.0, True, -np.inf, 10.0, None, None, None)),
          (lmfit.Parameter(name='a', expr="2.0*10.0"),  # set expression, vary becomes False
           ('a', -np.inf, True, -np.inf, np.inf, '2.0*10.0', None, None)),
          (lmfit.Parameter(name='a', brute_step=0.1),  # set brute_step
           ('a', -np.inf, True, -np.inf, np.inf, None, 0.1, None)),
          (lmfit.Parameter(name='a', user_data={'b': {}}),  # set user_data
           ('a', -np.inf, True, -np.inf, np.inf, None, None, {'b': {}}))]


@pytest.mark.parametrize('par, attr_values', in_out)
def test_initialize_Parameter(par, attr_values):
    """Test the initialization of the Parameter class."""
    assert_parameter_attributes(par, attr_values)

    # check for other default attributes
    for attribute in ['_expr', '_expr_ast', '_expr_eval', '_expr_deps',
                      '_delay_asteval', 'stderr', 'correl', 'from_internal',
                      '_val']:
        assert hasattr(par, attribute)


def test_Parameter_no_name():
    """Test for Parameter name, now required positional argument."""
    msg = r"missing 1 required positional argument: 'name'"
    with pytest.raises(TypeError, match=msg):
        lmfit.Parameter()


def test_init_bounds():
    """Tests to make sure that initial bounds are consistent.

    Only for specific cases not tested above with the initializations of the
    Parameter class.

    """
    # test 1: min > max; should swap min and max
    par = lmfit.Parameter(name='a', value=0.0, min=10.0, max=-10.0)
    assert par.min == -10.0
    assert par.max == 10.0

    # test 2: min == max; should raise a ValueError
    msg = r"Parameter 'a' has min == max"
    with pytest.raises(ValueError, match=msg):
        par = lmfit.Parameter(name='a', value=0.0, min=10.0, max=10.0)

    # FIXME: ideally this should be impossible to happen ever....
    # perhaps we should add a  setter method for MIN and MAX as well?
    # test 3: max or min is equal to None
    par.min = None
    par._init_bounds()
    assert par.min == -np.inf

    par.max = None
    par._init_bounds()
    assert par.max == np.inf


def test_parameter_set_value(parameter):
    """Test the Parameter.set() function with value."""
    par, initial_attribute_values = parameter

    par.set(value=None)  # nothing should change
    assert_parameter_attributes(par, initial_attribute_values)

    par.set(value=5.0)
    changed_attribute_values = ('a', 5.0, True, -100.0, 100.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)

    # check if set value works with new bounds, see issue#636
    par.set(value=500.0, min=400, max=600)
    changed_attribute_values2 = ('a', 500.0, True, 400.0, 600.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values2)


def test_parameter_set_vary(parameter):
    """Test the Parameter.set() function with vary."""
    par, initial_attribute_values = parameter

    par.set(vary=None)  # nothing should change
    assert_parameter_attributes(par, initial_attribute_values)

    par.set(vary=False)
    changed_attribute_values = ('a', 10.0, False, -100.0, 100.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)


def test_parameter_set_min(parameter):
    """Test the Parameter.set() function with min."""
    par, initial_attribute_values = parameter

    par.set(min=None)  # nothing should change
    assert_parameter_attributes(par, initial_attribute_values)

    par.set(min=-50.0)
    changed_attribute_values = ('a', 10.0, True, -50.0, 100.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)


def test_parameter_set_max(parameter):
    """Test the Parameter.set() function with max."""
    par, initial_attribute_values = parameter

    par.set(max=None)  # nothing should change
    assert_parameter_attributes(par, initial_attribute_values)

    par.set(max=50.0)
    changed_attribute_values = ('a', 10.0, True, -100.0, 50.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)


def test_parameter_set_expr(parameter):
    """Test the Parameter.set() function with expr.

    Of note, this only tests for setting/removal of the expression; nothing
    else gets evaluated here... More specific tests that require a Parameters
    class can be found below.

    """
    par, _ = parameter

    par.set(expr='2.0*50.0')  # setting an expression, vary --> False
    changed_attribute_values = ('a', 10.0, False, -100.0, 100.0, '2.0*50.0',
                                5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)

    par.set(expr=None)  # nothing should change
    assert_parameter_attributes(par, changed_attribute_values)

    par.set(expr='')  # should remove the expression
    changed_attribute_values = ('a', 10.0, False, -100.0, 100.0, None, 5.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)


def test_parameters_set_value_with_expr(parameters):
    """Test the Parameter.set() function with value in presence of expr."""
    pars, _, _ = parameters

    pars['a'].set(value=5.0)
    pars.update_constraints()  # update constraints/expressions
    changed_attr_values_A = ('a', 5.0, True, -100.0, 100.0, None, 5.0, 1)
    changed_attr_values_B = ('b', 10.0, False, -250.0, 250.0, "2.0*a", 25.0, 2.5)
    assert_parameter_attributes(pars['a'], changed_attr_values_A)
    assert_parameter_attributes(pars['b'], changed_attr_values_B)

    # with expression present, setting a value works and will leave vary=False
    pars['b'].set(value=1.0)
    pars.update_constraints()  # update constraints/expressions
    changed_attr_values_A = ('a', 5.0, True, -100.0, 100.0, None, 5.0, 1)
    changed_attr_values_B = ('b', 1.0, False, -250.0, 250.0, None, 25.0, 2.5)
    assert_parameter_attributes(pars['a'], changed_attr_values_A)
    assert_parameter_attributes(pars['b'], changed_attr_values_B)


def test_parameters_set_vary_with_expr(parameters):
    """Test the Parameter.set() function with vary in presence of expr."""
    pars, init_attr_values_A, _ = parameters

    pars['b'].set(vary=True)  # expression should get cleared
    pars.update_constraints()  # update constraints/expressions
    changed_attr_values_B = ('b', 20.0, True, -250.0, 250.0, None, 25.0, 2.5)
    assert_parameter_attributes(pars['a'], init_attr_values_A)
    assert_parameter_attributes(pars['b'], changed_attr_values_B)


def test_parameters_set_expr(parameters):
    """Test the Parameter.set() function with expr."""
    pars, init_attr_values_A, init_attr_values_B = parameters

    pars['b'].set(expr=None)  # nothing should change
    pars.update_constraints()  # update constraints/expressions
    assert_parameter_attributes(pars['a'], init_attr_values_A)
    assert_parameter_attributes(pars['b'], init_attr_values_B)

    pars['b'].set(expr='')  # expression should get cleared, vary still False
    pars.update_constraints()  # update constraints/expressions
    changed_attr_values_B = ('b', 20.0, False, -250.0, 250.0, None, 25.0, 2.5)
    assert_parameter_attributes(pars['a'], init_attr_values_A)
    assert_parameter_attributes(pars['b'], changed_attr_values_B)

    pars['a'].set(expr="b/4.0")  # expression should be set, vary --> False
    pars.update_constraints()
    changed_attr_values_A = ('a', 5.0, False, -100.0, 100.0, "b/4.0", 5.0, 1)
    changed_attr_values_B = ('b', 20.0, False, -250.0, 250.0, None, 25.0, 2.5)
    assert_parameter_attributes(pars['a'], changed_attr_values_A)
    assert_parameter_attributes(pars['b'], changed_attr_values_B)


def test_parameter_set_brute_step(parameter):
    """Test the Parameter.set() function with brute_step."""
    par, initial_attribute_values = parameter

    par.set(brute_step=None)  # nothing should change
    assert_parameter_attributes(par, initial_attribute_values)

    par.set(brute_step=0.0)  # brute_step set to None
    changed_attribute_values = ('a', 10.0, True, -100.0, 100.0, None, None, 1)
    assert_parameter_attributes(par, changed_attribute_values)

    par.set(brute_step=1.0)
    changed_attribute_values = ('a', 10.0, True, -100.0, 100.0, None, 1.0, 1)
    assert_parameter_attributes(par, changed_attribute_values)


def test_getstate(parameter):
    """Test for the __getstate__ method."""
    par, _ = parameter
    assert par.__getstate__() == ('a', 10.0, True, None, -100.0, 100.0, 5.0,
                                  None, None, 10, 1)


def test_setstate(parameter):
    """Test for the __setstate__ method."""
    par, initial_attribute_values = parameter
    state = par.__getstate__()

    par_new = lmfit.Parameter('new')
    attributes_new = ('new', -np.inf, True, -np.inf, np.inf, None, None, None)
    assert_parameter_attributes(par_new, attributes_new)

    par_new.__setstate__(state)
    assert_parameter_attributes(par_new, initial_attribute_values)


def test_parameter_pickle_(parameter):
    """Test that we can pickle a Parameter."""
    par, _ = parameter
    pkl = pickle.dumps(par)
    loaded_par = pickle.loads(pkl)

    assert loaded_par == par


def test_repr():
    """Tests for the __repr__ method."""
    par = lmfit.Parameter(name='test', value=10.0, min=0.0, max=20.0)
    assert par.__repr__() == "<Parameter 'test', value=10.0, bounds=[0.0:20.0]>"

    par = lmfit.Parameter(name='test', value=10.0, vary=False)
    assert par.__repr__() == "<Parameter 'test', value=10.0 (fixed), bounds=[-inf:inf]>"

    par.set(vary=True)
    par.stderr = 0.1
    assert par.__repr__() == "<Parameter 'test', value=10.0 +/- 0.1, bounds=[-inf:inf]>"

    par = lmfit.Parameter(name='test', expr='10.0*2.5')
    assert par.__repr__() == "<Parameter 'test', value=-inf, bounds=[-inf:inf], expr='10.0*2.5'>"

    par = lmfit.Parameter(name='test', brute_step=0.1)
    assert par.__repr__() == "<Parameter 'test', value=-inf, bounds=[-inf:inf], brute_step=0.1>"


def test_setup_bounds_and_scale_gradient_methods():
    """Tests for the setup_bounds and scale_gradient methods.

    Make use of the MINUIT-style transformation to obtain the the Parameter
    values and scaling factor for the gradient.
    See: https://lmfit.github.io/lmfit-py/bounds.html

    """
    # situation 1: no bounds
    par_no_bounds = lmfit.Parameter('no_bounds', value=10.0)
    assert_allclose(par_no_bounds.setup_bounds(), 10.0)
    assert_allclose(par_no_bounds.scale_gradient(par_no_bounds.value), 1.0)

    # situation 2: no bounds, min/max set to None after creating the parameter
    # TODO: ideally this should never happen; perhaps use a setter here
    par_no_bounds = lmfit.Parameter('no_bounds', value=10.0)
    par_no_bounds.min = None
    par_no_bounds.max = None
    assert_allclose(par_no_bounds.setup_bounds(), 10.0)
    assert_allclose(par_no_bounds.scale_gradient(par_no_bounds.value), 1.0)

    # situation 3: upper bound
    par_upper_bound = lmfit.Parameter('upper_bound', value=10.0, max=25.0)
    assert_allclose(par_upper_bound.setup_bounds(), 15.968719422671311)
    assert_allclose(par_upper_bound.scale_gradient(par_upper_bound.value),
                    -0.99503719, rtol=1.e-6)

    # situation 4: lower bound
    par_lower_bound = lmfit.Parameter('upper_bound', value=10.0, min=-25.0)
    assert_allclose(par_lower_bound.setup_bounds(), 35.98610843)
    assert_allclose(par_lower_bound.scale_gradient(par_lower_bound.value),
                    0.995037, rtol=1.e-6)

    # situation 5: both lower and upper bounds
    par_both_bounds = lmfit.Parameter('both_bounds', value=10.0, min=-25.0,
                                      max=25.0)
    assert_allclose(par_both_bounds.setup_bounds(), 0.4115168460674879)
    assert_allclose(par_both_bounds.scale_gradient(par_both_bounds.value),
                    -20.976788, rtol=1.e-6)


def test_value_setter(parameter):
    """Tests for the value setter."""
    par, initial_attribute_values = parameter
    assert_parameter_attributes(par, initial_attribute_values)

    par.value = 200.0  # above maximum
    assert_allclose(par.value, 100.0)

    par.value = -200.0  # below minimum
    assert_allclose(par.value, -100.0)

    del par._expr_eval
    par.value = 10.0
    assert_allclose(par.value, 10.0)
    assert hasattr(par, '_expr_eval')


# Tests for magic methods of the Parameter class
def test__array__(parameter):
    """Test the __array__ magic method."""
    par, _ = parameter
    assert np.array(par) == np.array(10.0)


def test__str__(parameter):
    """Test the __str__ magic method."""
    par, _ = parameter
    assert str(par) == "<Parameter 'a', value=10.0, bounds=[-100.0:100.0], brute_step=5.0>"


def test__abs__(parameter):
    """Test the __abs__ magic method."""
    par, _ = parameter
    assert_allclose(abs(par), 10.0)
    par.set(value=-10.0)
    assert_allclose(abs(par), 10.0)


def test__neg__(parameter):
    """Test the __neg__ magic method."""
    par, _ = parameter
    assert_allclose(-par, -10.0)
    par.set(value=-10.0)
    assert_allclose(-par, 10.0)


def test__pos__(parameter):
    """Test the __pos__ magic method."""
    par, _ = parameter
    assert_allclose(+par, 10.0)
    par.set(value=-10.0)
    assert_allclose(+par, -10.0)


def test__bool__(parameter):
    """Test the __bool__ magic method."""
    par, _ = parameter
    assert bool(par)


def test__int__(parameter):
    """Test the __int__ magic method."""
    par, _ = parameter
    assert isinstance(int(par), int)
    assert_allclose(int(par), 10)


def test__float__(parameter):
    """Test the __float__ magic method."""
    par, _ = parameter
    par.set(value=5)
    assert isinstance(float(par), float)
    assert_allclose(float(par), 5.0)


def test__trunc__(parameter):
    """Test the __trunc__ magic method."""
    par, _ = parameter
    par.set(value=10.5)
    assert isinstance(trunc(par), int)
    assert_allclose(trunc(par), 10)


def test__add__(parameter):
    """Test the __add__ magic method."""
    par, _ = parameter
    assert_allclose(par + 5.25, 15.25)


def test__sub__(parameter):
    """Test the __sub__ magic method."""
    par, _ = parameter
    assert_allclose(par - 5.25, 4.75)


def test__truediv__(parameter):
    """Test the __truediv__ magic method."""
    par, _ = parameter
    assert_allclose(par / 1.25, 8.0)


def test__floordiv__(parameter):
    """Test the __floordiv__ magic method."""
    par, _ = parameter
    par.set(value=5)
    assert_allclose(par // 2, 2)


def test__divmod__(parameter):
    """Test the __divmod__ magic method."""
    par, _ = parameter
    assert_allclose(divmod(par, 3), (3, 1))


def test__mod__(parameter):
    """Test the __mod__ magic method."""
    par, _ = parameter
    assert_allclose(par % 2, 0)
    assert_allclose(par % 3, 1)


def test__mul__(parameter):
    """Test the __mul__ magic method."""
    par, _ = parameter
    assert_allclose(par * 2.5, 25.0)
    assert_allclose(par * -0.1, -1.0)


def test__pow__(parameter):
    """Test the __pow__ magic method."""
    par, _ = parameter
    assert_allclose(par ** 0.5, 3.16227766)
    assert_allclose(par ** 4, 1e4)


def test__gt__(parameter):
    """Test the __gt__ magic method."""
    par, _ = parameter
    assert par < 11
    assert not par < 10


def test__ge__(parameter):
    """Test the __ge__ magic method."""
    par, _ = parameter
    assert par <= 11
    assert par <= 10
    assert not par <= 9


def test__le__(parameter):
    """Test the __le__ magic method."""
    par, _ = parameter
    assert par >= 9
    assert par >= 10
    assert not par >= 11


def test__lt__(parameter):
    """Test the __lt__ magic method."""
    par, _ = parameter
    assert par > 9
    assert not par > 10


def test__eq__(parameter):
    """Test the __eq__ magic method."""
    par, _ = parameter
    assert par == 10
    assert not par == 9


def test__ne__(parameter):
    """Test the __ne__ magic method."""
    par, _ = parameter
    assert par != 9
    assert not par != 10


def test__radd__(parameter):
    """Test the __radd__ magic method."""
    par, _ = parameter
    assert_allclose(5.25 + par, 15.25)


def test__rtruediv__(parameter):
    """Test the __rtruediv__ magic method."""
    par, _ = parameter
    assert_allclose(1.25 / par, 0.125)


def test__rdivmod__(parameter):
    """Test the __rdivmod__ magic method."""
    par, _ = parameter
    assert_allclose(divmod(3, par), (0, 3))


def test__rfloordiv__(parameter):
    """Test the __rfloordiv__ magic method."""
    par, _ = parameter
    assert_allclose(2 // par, 0)
    assert_allclose(20 // par, 2)


def test__rmod__(parameter):
    """Test the __rmod__ magic method."""
    par, _ = parameter
    assert_allclose(2 % par, 2)
    assert_allclose(25 % par, 5)


def test__rmul__(parameter):
    """Test the __rmul__ magic method."""
    par, _ = parameter
    assert_allclose(2.5 * par, 25.0)
    assert_allclose(-0.1 * par, -1.0)


def test__rpow__(parameter):
    """Test the __rpow__ magic method."""
    par, _ = parameter
    assert_allclose(0.5 ** par, 0.0009765625)
    assert_allclose(4 ** par, 1048576)


def test__rsub__(parameter):
    """Test the __rsub__ magic method."""
    par, _ = parameter
    assert_allclose(5.25 - par, -4.75)
