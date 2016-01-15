from __future__ import print_function
from lmfit import Parameters, Parameter
from lmfit.parameter import isclose
from numpy.testing import assert_, assert_almost_equal, assert_equal
import unittest
from copy import deepcopy
import numpy as np
import pickle


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
        assert_(not p is q)

        # now test if the asteval machinery survived
        assert_(q._asteval.symtable['abc'] == '2 * 3.142')

    def test_isclose(self):
        assert_(isclose(1., 1+1e-5, atol=1e-4, rtol=0))
        assert_(not isclose(1., 1+1e-5, atol=1e-6, rtol=0))
        assert_(isclose(1e10, 1.00001e10, rtol=1e-5, atol=1e-8))
        assert_(not isclose(0, np.inf))
        assert_(not isclose(-np.inf, np.inf))
        assert_(isclose(np.inf, np.inf))
        assert_(not isclose(np.nan, np.nan))


if __name__ == '__main__':
    unittest.main()
