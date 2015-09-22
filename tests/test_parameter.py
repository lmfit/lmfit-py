from __future__ import print_function
from lmfit import Parameters, Parameter
import pickle
import numpy as np
from numpy.testing import assert_
import unittest

class TestFitter(unittest.TestCase):

    def setUp(self):
        pass

    def test_pickle_parameter(self):
        # test that we can pickle a Parameter
        p = Parameter('a', 10, True, 0, 1)
        pkl = pickle.dumps(p)

        q = pickle.loads(pkl)

        assert_(p == q)

    def test_pickle_parameters(self):
        # test that we can pickle a Parameters object
        p = Parameters()
        p.add('a', 10, True, 0, 1)
        p.add('b', 10, True, 0, 100, 'a * 2')

        pkl = pickle.dumps(p, -1)
        q = pickle.loads(pkl)

        assert_(p == q)
        assert_(not p is q)

        # now test if the asteval machinery survived
        p['a'].value = 5
        assert_(p != q)
        q['a'].value = 5
        assert_(p == q)


if __name__ == '__main__':
    unittest.main()