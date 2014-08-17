from lmfit import Parameter
from numpy.testing import assert_allclose

def assert_paramval(param, val, tol=1.e-3):
    """assert that a named parameter's value is close to expected value"""

    assert(isinstance(param, Parameter))
    pval = param.value

    assert_allclose([pval], [val], rtol=tol, atol=tol,
                    err_msg='',verbose=True)

def assert_paramattr(param, attr, val):
    """assert that a named parameter's value is a value"""
    assert(isinstance(param, Parameter))
    assert(hasattr(param, attr))
    assert(getattr(param, attr) == val)

