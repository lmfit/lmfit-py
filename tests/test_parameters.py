"""Tests for the Parameters class."""

from copy import copy, deepcopy
import os
import pickle

import numpy as np
from numpy.testing import assert_allclose
import pytest

import lmfit
from lmfit.models import VoigtModel


@pytest.fixture
def parameters():
    """Initialize a Parameters class for tests."""
    pars = lmfit.Parameters()
    pars.add(lmfit.Parameter(name='a', value=10.0, vary=True, min=-100.0,
                             max=100.0, expr=None, brute_step=5.0,
                             user_data=1))
    pars.add(lmfit.Parameter(name='b', value=0.0, vary=True, min=-250.0,
                             max=250.0, expr="2.0*a", brute_step=25.0,
                             user_data={'test': 123}))
    exp_attr_values_A = ('a', 10.0, True, -100.0, 100.0, None, 5.0, 1)
    exp_attr_values_B = ('b', 20.0, False, -250.0, 250.0, "2.0*a", 25.0, {'test': 123})
    assert_parameter_attributes(pars['a'], exp_attr_values_A)
    assert_parameter_attributes(pars['b'], exp_attr_values_B)
    return pars, exp_attr_values_A, exp_attr_values_B


def assert_parameter_attributes(par, expected):
    """Assert that parameter attributes have the expected values."""
    par_attr_values = (par.name, par._val, par.vary, par.min, par.max,
                       par._expr, par.brute_step, par.user_data)
    assert par_attr_values == expected


def test_check_ast_errors():
    """Assert that an exception is raised upon AST errors."""
    pars = lmfit.Parameters()

    msg = r"at expr='<_?ast.Module object at"
    with pytest.raises(NameError, match=msg):
        pars.add('par1', expr='2.0*par2')


def test_parameters_init_with_usersyms():
    """Test for initialization of the Parameters class with usersyms."""
    pars = lmfit.Parameters(usersyms={'test': np.sin})
    assert 'test' in pars._asteval.symtable


def test_parameters_copy(parameters):
    """Tests for copying a Parameters class; all use the __deepcopy__ method."""
    pars, exp_attr_values_A, exp_attr_values_B = parameters

    copy_pars = copy(pars)
    pars_copy = pars.copy()
    pars__copy__ = pars.__copy__()

    # modifying the original parameters should not modify the copies
    pars['a'].set(value=100)
    pars['b'].user_data['test'] = 456

    for copied in [copy_pars, pars_copy, pars__copy__]:
        assert isinstance(copied, lmfit.Parameters)
        assert copied != pars
        assert copied._asteval is not None
        assert copied._asteval.symtable is not None
        assert_parameter_attributes(copied['a'], exp_attr_values_A)
        assert_parameter_attributes(copied['b'], exp_attr_values_B)


def test_parameters_deepcopy(parameters):
    """Tests for deepcopy of a Parameters class."""
    pars, _, _ = parameters

    deepcopy_pars = deepcopy(pars)
    assert isinstance(deepcopy_pars, lmfit.Parameters)
    assert deepcopy_pars == pars

    # check that we can add a symbol to the interpreter
    pars['b'].expr = 'sin(1)'
    pars['b'].value = 10
    assert_allclose(pars['b'].value, np.sin(1))
    assert_allclose(pars._asteval.symtable['b'], np.sin(1))

    # check that the symbols in the interpreter are still the same after
    # deepcopying
    pars, exp_attr_values_A, exp_attr_values_B = parameters
    deepcopy_pars = deepcopy(pars)

    unique_symbols_pars = pars._asteval.user_defined_symbols()
    unique_symbols_copied = deepcopy_pars._asteval.user_defined_symbols()
    assert unique_symbols_copied == unique_symbols_pars

    for unique_symbol in unique_symbols_copied:
        if pars._asteval.symtable[unique_symbol] is not np.nan:
            assert (pars._asteval.symtable[unique_symbol] ==
                    deepcopy_pars._asteval.symtable[unique_symbol])


def test_parameters_deepcopy_subclass():
    """Test that a subclass of parameters is preserved when performing a deepcopy"""
    class ParametersSubclass(lmfit.Parameters):
        pass

    parameters = ParametersSubclass()
    assert isinstance(parameters, ParametersSubclass)

    parameterscopy = deepcopy(parameters)
    assert isinstance(parameterscopy, ParametersSubclass)


def test_parameters_update(parameters):
    """Tests for updating a Parameters class."""
    pars, exp_attr_values_A, exp_attr_values_B = parameters

    msg = r"'test' is not a Parameters object"
    with pytest.raises(ValueError, match=msg):
        pars.update('test')

    pars2 = lmfit.Parameters()
    pars2.add(lmfit.Parameter(name='c', value=7.0, vary=True, min=-70.0,
                              max=70.0, expr=None, brute_step=0.7,
                              user_data=7))
    exp_attr_values_C = ('c', 7.0, True, -70.0, 70.0, None, 0.7, 7)

    pars_updated = pars.update(pars2)

    assert_parameter_attributes(pars_updated['a'], exp_attr_values_A)
    assert_parameter_attributes(pars_updated['b'], exp_attr_values_B)
    assert_parameter_attributes(pars_updated['c'], exp_attr_values_C)


def test_parameters__setitem__(parameters):
    """Tests for __setitem__ method of a Parameters class."""
    pars, _, exp_attr_values_B = parameters

    msg = r"'10' is not a valid Parameters name"
    with pytest.raises(KeyError, match=msg):
        pars.__setitem__('10', None)

    msg = r"'not_a_parameter' is not a Parameter"
    with pytest.raises(ValueError, match=msg):
        pars.__setitem__('a', 'not_a_parameter')

    par = lmfit.Parameter('b', value=10, min=-25.0, brute_step=1)
    pars.__setitem__('b', par)

    exp_attr_values_B = ('b', 10, True, -25.0, np.inf, None, 1, None)
    assert_parameter_attributes(pars['b'], exp_attr_values_B)


def test_parameters__add__(parameters):
    """Test the __add__ magic method."""
    pars, exp_attr_values_A, exp_attr_values_B = parameters

    msg = r"'other' is not a Parameters object"
    with pytest.raises(ValueError, match=msg):
        _ = pars + 'other'

    pars2 = lmfit.Parameters()
    pars2.add_many(('c', 1., True, None, None, None),
                   ('d', 2., True, None, None, None))
    exp_attr_values_C = ('c', 1, True, -np.inf, np.inf, None, None, None)
    exp_attr_values_D = ('d', 2, True, -np.inf, np.inf, None, None, None)

    pars_added = pars + pars2

    assert_parameter_attributes(pars_added['a'], exp_attr_values_A)
    assert_parameter_attributes(pars_added['b'], exp_attr_values_B)
    assert_parameter_attributes(pars_added['c'], exp_attr_values_C)
    assert_parameter_attributes(pars_added['d'], exp_attr_values_D)


def test_parameters__iadd__(parameters):
    """Test the __iadd__ magic method."""
    pars, exp_attr_values_A, exp_attr_values_B = parameters

    msg = r"'other' is not a Parameters object"
    with pytest.raises(ValueError, match=msg):
        pars += 'other'

    pars2 = lmfit.Parameters()
    pars2.add_many(('c', 1., True, None, None, None),
                   ('d', 2., True, None, None, None))
    exp_attr_values_C = ('c', 1, True, -np.inf, np.inf, None, None, None)
    exp_attr_values_D = ('d', 2, True, -np.inf, np.inf, None, None, None)

    pars += pars2

    assert_parameter_attributes(pars['a'], exp_attr_values_A)
    assert_parameter_attributes(pars['b'], exp_attr_values_B)
    assert_parameter_attributes(pars['c'], exp_attr_values_C)
    assert_parameter_attributes(pars['d'], exp_attr_values_D)


def test_parameters_add_with_symtable():
    """Regression test for GitHub Issue 607."""
    pars1 = lmfit.Parameters()
    pars1.add('a', value=1.0)

    def half(x):
        return 0.5*x

    pars2 = lmfit.Parameters(usersyms={"half": half})
    pars2.add("b", value=3.0)
    pars2.add("c", expr="half(b)")

    params = pars1 + pars2
    assert_allclose(params['c'].value, 1.5)

    params = pars2 + pars1
    assert_allclose(params['c'].value, 1.5)

    params = deepcopy(pars1)
    params.update(pars2)
    assert_allclose(params['c'].value, 1.5)

    pars1 += pars2
    assert_allclose(params['c'].value, 1.5)


def test_parameters__array__(parameters):
    """Test the __array__ magic method."""
    pars, _, _ = parameters

    assert_allclose(np.array(pars), np.array([10.0, 20.0]))


def test_parameters__reduce__(parameters):
    """Test the __reduce__ magic method."""
    pars, _, _ = parameters
    reduced = pars.__reduce__()

    assert isinstance(reduced[2], dict)
    assert 'unique_symbols' in reduced[2].keys()
    assert reduced[2]['unique_symbols']['b'] == 20
    assert 'params' in reduced[2].keys()
    assert isinstance(reduced[2]['params'][0], lmfit.Parameter)


def test_parameters__setstate__(parameters):
    """Test the __setstate__ magic method."""
    pars, exp_attr_values_A, exp_attr_values_B = parameters
    reduced = pars.__reduce__()

    pars_setstate = lmfit.Parameters()
    pars_setstate.__setstate__(reduced[2])

    assert isinstance(pars_setstate, lmfit.Parameters)
    assert_parameter_attributes(pars_setstate['a'], exp_attr_values_A)
    assert_parameter_attributes(pars_setstate['b'], exp_attr_values_B)


def test_pickle_parameters():
    """Test that we can pickle a Parameters object."""
    p = lmfit.Parameters()
    p.add('a', 10, True, 0, 100)
    p.add('b', 10, True, 0, 100, 'a * sin(1)')
    p.update_constraints()
    p._asteval.symtable['abc'] = '2 * 3.142'

    pkl = pickle.dumps(p, -1)
    q = pickle.loads(pkl)

    q.update_constraints()
    assert p == q
    assert p is not q

    # now test if the asteval machinery survived
    assert q._asteval.symtable['abc'] == '2 * 3.142'

    # check that unpickling of Parameters is not affected by expr that
    # refer to Parameter that are added later on. In the following
    # example var_0.expr refers to var_1, which is a Parameter later
    # on in the Parameters dictionary.
    p = lmfit.Parameters()
    p.add('var_0', value=1)
    p.add('var_1', value=2)
    p['var_0'].expr = 'var_1'
    pkl = pickle.dumps(p)
    q = pickle.loads(pkl)


def test_parameters_eval(parameters):
    """Test the eval method."""
    pars, _, _ = parameters
    evaluated = pars.eval('10.0*a+b')
    assert_allclose(evaluated, 120)

    # check that eval() works with usersyms and parameter values
    def myfun(x):
        return 2.0 * x

    pars2 = lmfit.Parameters(usersyms={"myfun": myfun})
    pars2.add('a', value=4.0)
    pars2.add('b', value=3.0)
    assert_allclose(pars2.eval('myfun(2.0) * a'), 16)
    assert_allclose(pars2.eval('b / myfun(3.0)'), 0.5)


def test_parameters_pretty_repr(parameters):
    """Test the pretty_repr method."""
    pars, _, _ = parameters
    output = pars.pretty_repr()
    output_oneline = pars.pretty_repr(oneline=True)

    split_output = output.split('\n')
    assert len(split_output) == 5
    assert 'Parameters' in split_output[0]
    assert "Parameter 'a'" in split_output[1]
    assert "Parameter 'b'" in split_output[2]

    oneliner = ("Parameters([('a', <Parameter 'a', value=10.0, "
                "bounds=[-100.0:100.0], brute_step=5.0>), ('b', <Parameter "
                "'b', value=20.0, bounds=[-250.0:250.0], expr='2.0*a', "
                "brute_step=25.0>)])")
    assert output_oneline == oneliner


def test_parameters_pretty_print(parameters, capsys):
    """Test the pretty_print method."""
    pars, _, _ = parameters

    # oneliner
    pars.pretty_print(oneline=True)
    captured = capsys.readouterr()
    oneliner = ("Parameters([('a', <Parameter 'a', value=10.0, "
                "bounds=[-100.0:100.0], brute_step=5.0>), ('b', <Parameter "
                "'b', value=20.0, bounds=[-250.0:250.0], expr='2.0*a', "
                "brute_step=25.0>)])")
    assert oneliner in captured.out

    # default
    pars.pretty_print()
    captured = capsys.readouterr()
    captured_split = captured.out.split('\n')
    assert len(captured_split) == 4
    header = ('Name     Value      Min      Max   Stderr     Vary     '
              'Expr Brute_Step')
    assert captured_split[0] == header

    # specify columnwidth
    pars.pretty_print(colwidth=12)
    captured = capsys.readouterr()
    captured_split = captured.out.split('\n')
    header = ('Name         Value          Min          Max       Stderr     '
              '    Vary         Expr   Brute_Step')
    assert captured_split[0] == header

    # specify columns
    pars['a'].stderr = 0.01
    pars.pretty_print(columns=['value', 'min', 'max', 'stderr'])
    captured = capsys.readouterr()
    captured_split = captured.out.split('\n')
    assert captured_split[0] == 'Name     Value      Min      Max   Stderr'
    assert captured_split[1] == 'a        10     -100      100     0.01'
    assert captured_split[2] == 'b        20     -250      250     None'

    # specify fmt
    pars.pretty_print(fmt='e', columns=['value', 'min', 'max'])
    captured = capsys.readouterr()
    captured_split = captured.out.split('\n')
    assert captured_split[0] == 'Name     Value      Min      Max'
    assert captured_split[1] == 'a  1.0000e+01 -1.0000e+02 1.0000e+02'
    assert captured_split[2] == 'b  2.0000e+01 -2.5000e+02 2.5000e+02'

    # specify precision
    pars.pretty_print(precision=2, fmt='e', columns=['value', 'min', 'max'])
    captured = capsys.readouterr()
    captured_split = captured.out.split('\n')
    assert captured_split[0] == 'Name     Value      Min      Max'
    assert captured_split[1] == 'a  1.00e+01 -1.00e+02 1.00e+02'
    assert captured_split[2] == 'b  2.00e+01 -2.50e+02 2.50e+02'


def test_parameters__repr_html_(parameters):
    """Test _repr_html method to generate HTML table for Parameters class."""
    pars, _, _ = parameters
    repr_html = pars._repr_html_()

    assert isinstance(repr_html, str)
    assert '<table class="jp-toc-ignore"><caption>Parameters</caption>' in repr_html


def test_parameters_add():
    """Tests for adding a Parameter to the Parameters class."""
    pars = lmfit.Parameters()
    pars_from_par = lmfit.Parameters()

    pars.add('a')
    exp_attr_values_A = ('a', -np.inf, True, -np.inf, np.inf, None, None, None)
    assert_parameter_attributes(pars['a'], exp_attr_values_A)

    pars_from_par.add(lmfit.Parameter('a'))
    assert pars_from_par == pars

    pars.add('b', value=1, vary=False, min=-5.0, max=5.0, brute_step=0.1)
    exp_attr_values_B = ('b', 1.0, False, -5.0, 5.0, None, 0.1, None)
    assert_parameter_attributes(pars['b'], exp_attr_values_B)

    pars_from_par.add(lmfit.Parameter('b', value=1, vary=False, min=-5.0,
                                      max=5.0, brute_step=0.1))
    assert pars_from_par == pars


def test_add_params_expr_outoforder():
    """Regression test for GitHub Issue 560."""
    params1 = lmfit.Parameters()
    params1.add("a", value=1.0)

    params2 = lmfit.Parameters()
    params2.add("b", value=1.0)
    params2.add("c", value=2.0)
    params2['b'].expr = 'c/2'

    params = params1 + params2
    assert 'b' in params
    assert_allclose(params['b'].value, 1.0)


def test_parameters_add_many():
    """Tests for add_many method."""
    a = lmfit.Parameter('a', 1)
    b = lmfit.Parameter('b', 2)

    par = lmfit.Parameters()
    par.add_many(a, b)

    par_with_tuples = lmfit.Parameters()
    par_with_tuples.add_many(('a', 1), ('b', 2))

    assert list(par.keys()) == ['a', 'b']
    assert par == par_with_tuples


def test_parameters_valuesdict(parameters):
    """Test for valuesdict method."""
    pars, _, _ = parameters
    vals_dict = pars.valuesdict()

    assert isinstance(vals_dict, dict)
    assert_allclose(vals_dict['a'], pars['a'].value)
    assert_allclose(vals_dict['b'], pars['b'].value)


def test_dumps_loads_parameters(parameters):
    """Test for dumps and loads methods for a Parameters class."""
    pars, _, _ = parameters

    dumps = pars.dumps()
    assert isinstance(dumps, str)
    newpars = lmfit.Parameters().loads(dumps)
    assert newpars == pars

    newpars['a'].value = 100.0
    assert_allclose(newpars['b'].value, 200.0)


def test_dump_load_parameters(parameters):
    """Test for dump and load methods for a Parameters class."""
    pars, _, _ = parameters

    with open('parameters.sav', 'w') as outfile:
        pars.dump(outfile)

    with open('parameters.sav') as infile:
        newpars = pars.load(infile)

    assert newpars == pars
    newpars['a'].value = 100.0
    assert_allclose(newpars['b'].value, 200.0)


def test_dumps_loads_parameters_usersyms():
    """Test for dumps/loads methods for a Parameters class with usersyms."""
    def half(x):
        return 0.5*x

    pars = lmfit.Parameters(usersyms={"half": half, 'my_func': np.sqrt})
    pars.add(lmfit.Parameter(name='a', value=9.0, min=-100.0, max=100.0))
    pars.add(lmfit.Parameter(name='b', value=100.0, min=-250.0, max=250.0))
    pars.add("c", expr="half(b) + my_func(a)")

    dumps = pars.dumps()
    assert isinstance(dumps, str)
    assert '"half": {' in dumps
    assert '"my_func": {' in dumps

    newpars = lmfit.Parameters().loads(dumps)
    assert 'half' in newpars._asteval.symtable
    assert 'my_func' in newpars._asteval.symtable
    assert_allclose(newpars['a'].value, 9.0)
    assert_allclose(newpars['b'].value, 100.0)

    # within the py.test environment the encoding of the function 'half' does
    # not work correctly as it is changed from <function half at 0x?????????>"
    # to "<function test_dumps_loads_parameters_usersyms.<locals>.half at 0x?????????>
    # This result in the "importer" to be set to None and the final "decode4js"
    # does not do the correct thing.
    #
    # Of note, this is only an issue within the py.test framework and it DOES
    # work correctly in a normal Python interpreter. Also, it isn't an issue
    # when DILL is used, so in that case the two asserts below will pass.
    if lmfit.jsonutils.HAS_DILL:
        assert newpars == pars
        assert_allclose(newpars['c'].value, 53.0)


def test_parameters_expr_and_constraints():
    """Regression tests for GitHub Issue #265. Test that parameters are re-
    evaluated if they have bounds and expr.

    """
    p = lmfit.Parameters()
    p.add(lmfit.Parameter('a', 10, True))
    p.add(lmfit.Parameter('b', 10, True, 0, 20))

    assert_allclose(p['b'].min, 0)
    assert_allclose(p['b'].max, 20)

    p['a'].expr = '2 * b'
    assert_allclose(p['a'].value, 20)

    p['b'].value = 15
    assert_allclose(p['b'].value, 15)
    assert_allclose(p['a'].value, 30)

    p['b'].value = 30
    assert_allclose(p['b'].value, 20)
    assert_allclose(p['a'].value, 40)


def test_parameters_usersyms():
    """Test for passing usersyms to Parameters()."""
    def myfun(x):
        return x**3

    params = lmfit.Parameters(usersyms={"myfun": myfun})
    params.add("a", value=2.3)
    params.add("b", expr="myfun(a)")

    np.random.seed(2020)
    xx = np.linspace(0, 1, 10)
    yy = 3 * xx + np.random.normal(scale=0.002, size=xx.size)

    model = lmfit.Model(lambda x, a: a * x)
    result = model.fit(yy, params=params, x=xx)
    assert_allclose(result.params['a'].value, 3.0, rtol=1e-3)
    assert (result.nfev > 3 and result.nfev < 300)


def test_parameters_expr_with_bounds():
    """Test Parameters using an expression with bounds, without value."""
    pars = lmfit.Parameters()
    pars.add('c1', value=0.2)
    pars.add('c2', value=0.2)
    pars.add('c3', value=0.2)
    pars.add('csum', value=0.8)

    # this should not raise TypeError:
    pars.add('c4', expr='csum-c1-c2-c3', min=0, max=1)
    assert_allclose(pars['c4'].value, 0.2)


def test_invalid_expr_exceptions():
    """Regression test for GitHub Issue #486: check that an exception is
    raised for invalid expressions.

    """
    p1 = lmfit.Parameters()
    p1.add('t', 2.0, min=0.0, max=5.0)
    p1.add('x', 10.0)

    with pytest.raises(SyntaxError):
        p1.add('y', expr='x*t + sqrt(t)/')
    assert len(p1['y']._expr_eval.error) > 0

    p1.add('y', expr='x*t + sqrt(t)/3.0')
    p1['y'].set(expr='x*3.0 + t**2')
    assert 'x*3' in p1['y'].expr
    assert len(p1['y']._expr_eval.error) == 0

    with pytest.raises(SyntaxError):
        p1['y'].set(expr='t+')
    assert len(p1['y']._expr_eval.error) > 0
    assert_allclose(p1['y'].value, 34.0)


def test_create_params():
    """Tests for create_params() function."""
    pars1 = lmfit.create_params(a=8, b=9,
                                c=dict(value=3, min=0, max=10),
                                d=dict(expr='a+b/c'),
                                e=dict(value=10000, brute_step=4))

    assert pars1['a'].value == 8
    assert pars1['b'].value == 9
    assert pars1['c'].value == 3
    assert pars1['c'].min == 0
    assert pars1['c'].max == 10
    assert pars1['d'].expr == 'a+b/c'
    assert pars1['d'].value == 11
    assert pars1['e'].value == 10000
    assert pars1['e'].brute_step == 4


def test_unset_constrained_param():
    """test 'unsetting' a constrained parameter by
    just setting `param.vary = True`

    """
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), '..',
                                   'examples', 'test_peak.dat'))
    x = data[:, 0]
    y = data[:, 1]

    # initial fit
    mod = VoigtModel()
    params = mod.guess(y, x=x)
    out1 = mod.fit(y, params, x=x)

    assert out1.nvarys == 3
    assert out1.chisqr < 20.0

    # now just gamma to vary
    params['gamma'].vary = True
    out2 = mod.fit(y, params, x=x)

    assert out2.nvarys == 4
    assert out2.chisqr < out1.chisqr
    assert out2.rsquared > out1.rsquared
    assert out2.params['gamma'].correl['sigma'] < -0.6
