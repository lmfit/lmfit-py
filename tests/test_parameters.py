from __future__ import print_function

from copy import copy, deepcopy
import pickle
import unittest

import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_equal

from lmfit import Model, Parameter, Parameters
from lmfit.printfuncs import params_html_table


class TestParameters(unittest.TestCase):

    def setUp(self):
        self.params = Parameters()
        self.params.add_many(('a', 1., True, None, None, None),
                             ('b', 2., True, None, None, None),
                             ('c', 3., True, None, None, '2. * a'))

    def test_expr_was_evaluated(self):
        self.params.update_constraints()
        assert_almost_equal(self.params['c'].value,
                            2 * self.params['a'].value)

    def test_copy(self):
        # check simple Parameters.copy() does not fail
        # on non-trivial Parameters
        p1 = Parameters()
        p1.add('t', 2.0, min=0.0, max=5.0)
        p1.add('x', 10.0)
        p1.add('y', expr='x*t + sqrt(t)/3.0')

        p2 = p1.copy()
        assert isinstance(p2, Parameters)
        assert 't' in p2
        assert 'y' in p2
        assert p2['t'].max < 6.0
        assert np.isinf(p2['x'].max) and p2['x'].max > 0
        assert np.isinf(p2['x'].min) and p2['x'].min < 0
        assert 'sqrt(t)' in p2['y'].expr
        assert p2._asteval is not None
        assert p2._asteval.symtable is not None
        assert (p2['y'].value > 20) and (p2['y'].value < 21)

    def test_copy_function(self):
        # check copy(Parameters) does not fail
        p1 = Parameters()
        p1.add('t', 2.0, min=0.0, max=5.0)
        p1.add('x', 10.0)
        p1.add('y', expr='x*t + sqrt(t)/3.0')

        p2 = copy(p1)
        assert isinstance(p2, Parameters)

        # change the 'x' value in the original
        p1['x'].value = 4.0

        assert p2['x'].value > 9.8
        assert p2['x'].value < 10.2
        assert np.isinf(p2['x'].max) and p2['x'].max > 0

        assert 't' in p2
        assert 'y' in p2
        assert p2['t'].max < 6.0

        assert np.isinf(p2['x'].min) and p2['x'].min < 0
        assert 'sqrt(t)' in p2['y'].expr
        assert p2._asteval is not None
        assert p2._asteval.symtable is not None
        assert (p2['y'].value > 20) and (p2['y'].value < 21)

        assert p1['y'].value < 10

    def test_deepcopy(self):
        # check that a simple copy works
        b = deepcopy(self.params)
        assert_(self.params == b)

        # check that we can add a symbol to the interpreter
        self.params['b'].expr = 'sin(1)'
        self.params['b'].value = 10
        assert_almost_equal(self.params['b'].value, np.sin(1))
        assert_almost_equal(self.params._asteval.symtable['b'], np.sin(1))

        # check that the symbols in the interpreter are still the same after
        # deepcopying
        b = deepcopy(self.params)

        unique_symbols_params = self.params._asteval.user_defined_symbols()
        unique_symbols_b = self.params._asteval.user_defined_symbols()
        assert_(unique_symbols_b == unique_symbols_params)
        for unique_symbol in unique_symbols_b:
            if self.params._asteval.symtable[unique_symbol] is np.nan:
                continue

            assert_(self.params._asteval.symtable[unique_symbol]
                    ==
                    b._asteval.symtable[unique_symbol])

    def test_add_many_params(self):
        # test that we can add many parameters, but only parameters are added.
        a = Parameter('a', 1)
        b = Parameter('b', 2)

        p = Parameters()
        p.add_many(a, b)

        assert_(list(p.keys()) == ['a', 'b'])

    def test_expr_and_constraints_GH265(self):
        # test that parameters are reevaluated if they have bounds and expr
        # see GH265
        p = Parameters()

        p['a'] = Parameter('a', 10, True)
        p['b'] = Parameter('b', 10, True, 0, 20)

        assert_equal(p['b'].min, 0)
        assert_equal(p['b'].max, 20)

        p['a'].expr = '2 * b'
        assert_almost_equal(p['a'].value, 20)

        p['b'].value = 15
        assert_almost_equal(p['b'].value, 15)
        assert_almost_equal(p['a'].value, 30)

        p['b'].value = 30
        assert_almost_equal(p['b'].value, 20)
        assert_almost_equal(p['a'].value, 40)

    def test_pickle_parameter(self):
        # test that we can pickle a Parameter
        p = Parameter('a', 10, True, 0, 1)
        pkl = pickle.dumps(p)

        q = pickle.loads(pkl)

        assert_(p == q)

    def test_pickle_parameters(self):
        # test that we can pickle a Parameters object
        p = Parameters()
        p.add('a', 10, True, 0, 100)
        p.add('b', 10, True, 0, 100, 'a * sin(1)')
        p.update_constraints()
        p._asteval.symtable['abc'] = '2 * 3.142'

        pkl = pickle.dumps(p, -1)
        q = pickle.loads(pkl)

        q.update_constraints()
        assert_(p == q)
        assert_(p is not q)

        # now test if the asteval machinery survived
        assert_(q._asteval.symtable['abc'] == '2 * 3.142')

        # check that unpickling of Parameters is not affected by expr that
        # refer to Parameter that are added later on. In the following
        # example var_0.expr refers to var_1, which is a Parameter later
        # on in the Parameters OrderedDict.
        p = Parameters()
        p.add('var_0', value=1)
        p.add('var_1', value=2)
        p['var_0'].expr = 'var_1'
        pkl = pickle.dumps(p)
        q = pickle.loads(pkl)

    def test_params_usersyms(self):
        # test passing usersymes to Parameters()
        def myfun(x):
            return x**3

        params = Parameters(usersyms={"myfun": myfun})
        params.add("a", value=2.3)
        params.add("b", expr="myfun(a)")

        xx = np.linspace(0, 1, 10)
        yy = 3 * xx + np.random.normal(scale=0.002, size=len(xx))

        model = Model(lambda x, a: a * x)
        result = model.fit(yy, params=params, x=xx)
        assert_(np.isclose(result.params['a'].value, 3.0, rtol=0.025))
        assert_(result.nfev > 3)
        assert_(result.nfev < 300)

    def test_set_symtable(self):
        # test that we use Parameter.set(value=XXX) and have
        # that new value be used in constraint expressions
        pars = Parameters()
        pars.add('x', value=1.0)
        pars.add('y', expr='x + 1')

        assert_(np.isclose(pars['y'].value, 2.0))
        pars['x'].set(value=3.0)
        assert_(np.isclose(pars['y'].value, 4.0))

    def test_dumps_loads_parameters(self):
        # test that we can dumps() and then loads() a Parameters
        pars = Parameters()
        pars.add('x', value=1.0)
        pars.add('y', value=2.0)
        pars['x'].expr = 'y / 2.0'

        dumps = pars.dumps()

        newpars = Parameters().loads(dumps)
        newpars['y'].value = 100.0
        assert_(np.isclose(newpars['x'].value, 50.0))

    def test_isclose(self):
        assert_(np.isclose(1., 1+1e-5, atol=1e-4, rtol=0))
        assert_(not np.isclose(1., 1+1e-5, atol=1e-6, rtol=0))
        assert_(np.isclose(1e10, 1.00001e10, rtol=1e-5, atol=1e-8))
        assert_(not np.isclose(0, np.inf))
        assert_(not np.isclose(-np.inf, np.inf))
        assert_(np.isclose(np.inf, np.inf))
        assert_(not np.isclose(np.nan, np.nan))

    def test_expr_with_bounds(self):
        "test an expression with bounds, without value"
        pars = Parameters()
        pars.add('c1', value=0.2)
        pars.add('c2', value=0.2)
        pars.add('c3', value=0.2)
        pars.add('csum', value=0.8)
        # this should not raise TypeError:
        pars.add('c4', expr='csum-c1-c2-c3', min=0, max=1)
        assert_(np.isclose(pars['c4'].value, 0.2))

    def test_invalid_expr_exceptions(self):
        "test if an exception is raised for invalid expressions (GH486)"""
        p1 = Parameters()
        p1.add('t', 2.0, min=0.0, max=5.0)
        p1.add('x', 10.0)
        with self.assertRaises(SyntaxError):
            p1.add('y', expr='x*t + sqrt(t)/')
        assert len(p1['y']._expr_eval.error) > 0
        p1.add('y', expr='x*t + sqrt(t)/3.0')
        p1['y'].set(expr='x*3.0 + t**2')
        assert 'x*3' in p1['y'].expr
        assert len(p1['y']._expr_eval.error) == 0
        with self.assertRaises(SyntaxError):
            p1['y'].set(expr='t+')
        assert len(p1['y']._expr_eval.error) > 0
        assert_almost_equal(p1['y'].value, 34.0)

    def test_eval(self):
        # check that eval() works with usersyms and parameter values
        def myfun(x):
            return 2.0 * x
        p = Parameters(usersyms={"myfun": myfun})
        p.add("a", value=4.0)
        p.add("b", value=3.0)
        assert_almost_equal(p.eval("myfun(2.0) * a"), 16)
        assert_almost_equal(p.eval("b / myfun(3.0)"), 0.5)

    def test_params_html_table(self):
        p1 = Parameters()
        p1.add('t', 2.0, min=0.0, max=5.0)
        p1.add('x', 0.0, )

        html = params_html_table(p1)
        self.assertIsInstance(html, str)

    def test_add_params_expr_outoforder(self):
        params1 = Parameters()
        params1.add("a", value=1.0)

        params2 = Parameters()
        params2.add("b", value=1.0)
        params2.add("c", value=2.0)
        params2['b'].expr = 'c/2'

        params = params1 + params2
        assert 'b' in params
        assert_almost_equal(params['b'].value, 1.0)

    def test_params_prints(self):
        params = Parameters()
        params.add("a", value=1.0, vary=True)
        params.add("b", value=8.5, min=0, vary=True)
        params.add("c", expr='a + sqrt(b)')

        repr_full = params.pretty_repr()
        repr_one = params.pretty_repr(oneline=True)

        out = []
        for key, val in params.items():
            out.append("%s: %s" % (key, repr(val)))
        out = '\n'.join(out)

        assert repr_full.count('\n') > 4
        assert repr_one.count('\n') < 2
        assert len(repr_full) > 150
        assert len(repr_one) > 150
        assert len(out) > 150

    def test_add_with_symtable(self):
        pars1 = Parameters()
        pars1.add("a", value=1.0, vary=True)

        def half(x):
            return 0.5*x

        pars2 = Parameters(usersyms={"half": half})
        pars2.add("b", value=3.0)
        pars2.add("c", expr="half(b)")

        params = pars1 + pars2
        assert_almost_equal(params['c'].value, 1.5)

        params = pars2 + pars1
        assert_almost_equal(params['c'].value, 1.5)

        params = deepcopy(pars1)
        params.update(pars2)
        assert_almost_equal(params['c'].value, 1.5)
