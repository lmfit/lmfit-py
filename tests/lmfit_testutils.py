from numpy.testing import assert_allclose

from lmfit import Parameter


def assert_paramval(param, val, tol=1.e-3):
    """assert that a named parameter's value is close to expected value"""

    assert(isinstance(param, Parameter))
    pval = param.value

    assert_allclose([pval], [val], rtol=tol, atol=tol,
                    err_msg='', verbose=True)


def assert_paramattr(param, attr, val):
    """assert that a named parameter's value is a value"""
    assert(isinstance(param, Parameter))
    assert(hasattr(param, attr))
    assert(getattr(param, attr) == val)


def assert_between(val, minval, maxval):
    """assert that a value is between minval and maxval"""
    assert(val >= minval)
    assert(val <= maxval)


def assert_param_between(param, minval, maxval):
    """assert that a named parameter's value is
    between minval and maxval"""

    assert(isinstance(param, Parameter))
    assert_between(param.value, minval, maxval)
