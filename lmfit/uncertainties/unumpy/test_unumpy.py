"""
Tests of the code in uncertainties/unumpy/__init__.py.

These tests can be run through the Nose testing framework.

(c) 2010-2013 by Eric O. LEBIGOT (EOL).
"""

from __future__ import division

# 3rd-party modules:
try:
    import numpy
except ImportError:
    import sys
    sys.exit()  # There is no reason to test the interface to NumPy

# Local modules:
import uncertainties
from uncertainties import ufloat, unumpy, test_uncertainties
from uncertainties.unumpy import core
from uncertainties.test_uncertainties import _numbers_close, matrices_close
from uncertainties import __author__

def test_numpy():

    """
    Interaction with NumPy, including matrix inversion and correlated_values.
    """

    arr = numpy.array(range(3))
    num = ufloat((3.14, 0.01))

    # NumPy arrays can be multiplied by Variable objects,
    # whatever the order of the operands:
    prod1 = arr*num
    prod2 = num*arr
    # Additional check:
    assert (prod1 == prod2).all()

    # Operations with arrays work (they are first handled by NumPy,
    # then by this module):
    prod1*prod2  # This should be calculable
    assert not (prod1-prod2).any()  # All elements must be 0

    # Comparisons work too:

    # Usual behavior:
    assert len(arr[arr > 1.5]) == 1
    # Comparisons with Variable objects:
    assert len(arr[arr > ufloat((1.5, 0.1))]) == 1

    assert len(prod1[prod1 < prod1*prod2]) == 2

    # The following can be calculated (special NumPy abs() function):
    numpy.abs(arr + ufloat((-1, 0.1)))

    # The following does not completely work, because NumPy does not
    # implement numpy.exp on an array of general objects, apparently:
    assert numpy.exp(arr).all()  # All elements > 0
    # Equivalent with an array of AffineScalarFunc objects:
    try:
        numpy.exp(arr + ufloat((0, 0)))
    except AttributeError:
        pass  # ! This is usual (but could be avoided)
    else:
        raise Exception("numpy.exp unexpectedly worked")

def test_matrix():
    "Matrices of numbers with uncertainties"
    # Matrix inversion:

    # Matrix with a mix of Variable objects and regular
    # Python numbers:

    m = unumpy.matrix([[ufloat((10, 1)), -3.1],
                       [0, ufloat((3, 0))]])
    m_nominal_values = unumpy.nominal_values(m)

    # Test of the nominal_value attribute:
    assert numpy.all(m_nominal_values == m.nominal_values)

    assert type(m[0, 0]) == uncertainties.Variable

    # Test of scalar multiplication, both sides:
    3*m
    m*3

def _derivatives_close(x, y):
    """
    Returns True iff the AffineScalarFunc objects x and y have
    derivatives that are close to each other (they must depend
    on the same variables).
    """

    # x and y must depend on the same variables:
    if set(x.derivatives) != set(y.derivatives):
        return False  # Not the same variables

    return all(_numbers_close(x.derivatives[var], y.derivatives[var])
               for var in x.derivatives)

def test_inverse():
    "Tests of the matrix inverse"

    m = unumpy.matrix([[ufloat((10, 1)), -3.1],
                       [0, ufloat((3, 0))]])
    m_nominal_values = unumpy.nominal_values(m)

    # "Regular" inverse matrix, when uncertainties are not taken
    # into account:
    m_no_uncert_inv = m_nominal_values.I

    # The matrix inversion should not yield numbers with uncertainties:
    assert m_no_uncert_inv.dtype == numpy.dtype(float)

    # Inverse with uncertainties:
    m_inv_uncert = m.I  # AffineScalarFunc elements
    # The inverse contains uncertainties: it must support custom
    # operations on matrices with uncertainties:
    assert isinstance(m_inv_uncert, unumpy.matrix)
    assert type(m_inv_uncert[0, 0]) == uncertainties.AffineScalarFunc

    # Checks of the numerical values: the diagonal elements of the
    # inverse should be the inverses of the diagonal elements of
    # m (because we started with a triangular matrix):
    assert _numbers_close(1/m_nominal_values[0, 0],
                          m_inv_uncert[0, 0].nominal_value), "Wrong value"

    assert _numbers_close(1/m_nominal_values[1, 1],
                          m_inv_uncert[1, 1].nominal_value), "Wrong value"


    ####################

    # Checks of the covariances between elements:
    x = ufloat((10, 1))
    m = unumpy.matrix([[x, x],
                       [0, 3+2*x]])

    m_inverse = m.I

    # Check of the properties of the inverse:
    m_double_inverse = m_inverse.I
    # The initial matrix should be recovered, including its
    # derivatives, which define covariances:
    assert _numbers_close(m_double_inverse[0, 0].nominal_value,
                          m[0, 0].nominal_value)
    assert _numbers_close(m_double_inverse[0, 0].std_dev(),
                          m[0, 0].std_dev())

    assert matrices_close(m_double_inverse, m)

    # Partial test:
    assert _derivatives_close(m_double_inverse[0, 0], m[0, 0])
    assert _derivatives_close(m_double_inverse[1, 1], m[1, 1])

    ####################

    # Tests of covariances during the inversion:

    # There are correlations if both the next two derivatives are
    # not zero:
    assert m_inverse[0, 0].derivatives[x]
    assert m_inverse[0, 1].derivatives[x]

    # Correlations between m and m_inverse should create a perfect
    # inversion:
    assert matrices_close(m * m_inverse,  numpy.eye(m.shape[0]))

def test_pseudo_inverse():
    "Tests of the pseudo-inverse"

    # Numerical version of the pseudo-inverse:
    pinv_num = core.wrap_array_func(numpy.linalg.pinv)

    ##########
    # Full rank rectangular matrix:
    m = unumpy.matrix([[ufloat((10, 1)), -3.1],
                       [0, ufloat((3, 0))],
                       [1, -3.1]])

    # Numerical and package (analytical) pseudo-inverses: they must be
    # the same:
    rcond = 1e-8  # Test of the second argument to pinv()
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core._pinv(m, rcond)
    assert matrices_close(m_pinv_num, m_pinv_package)

    ##########
    # Example with a non-full rank rectangular matrix:
    vector = [ufloat((10, 1)), -3.1, 11]
    m = unumpy.matrix([vector, vector])
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core._pinv(m, rcond)
    assert matrices_close(m_pinv_num, m_pinv_package)

    ##########
    # Example with a non-full-rank square matrix:
    m = unumpy.matrix([[ufloat((10, 1)), 0], [3, 0]])
    m_pinv_num = pinv_num(m, rcond)
    m_pinv_package = core._pinv(m, rcond)
    assert matrices_close(m_pinv_num, m_pinv_package)

def test_broadcast_funcs():
    """
    Test of mathematical functions that work with NumPy arrays of
    numbers with uncertainties.
    """

    x = uncertainties.ufloat((0.2, 0.1))
    arr = numpy.array([x, 2*x])
    assert unumpy.cos(arr)[1] == uncertainties.umath.cos(arr[1])

    # Some functions do not bear the same name in the math module and
    # in NumPy (acos instead of arccos, etc.):
    assert unumpy.arccos(arr)[1] == uncertainties.umath.acos(arr[1])
    # The acos() function should not exist in unumpy because it does
    # not exist in numpy:
    assert not hasattr(numpy, 'acos')
    assert not hasattr(unumpy, 'acos')

    # Test of the __all__ variable:
    assert 'acos' not in unumpy.__all__

def test_array_and_matrix_creation():
    "Test of custom array creation"

    arr = unumpy.uarray(([1, 2], [0.1, 0.2]))

    assert arr[1].nominal_value == 2
    assert arr[1].std_dev() == 0.2

    # Same thing for matrices:
    mat = unumpy.umatrix(([1, 2], [0.1, 0.2]))
    assert mat[0, 1].nominal_value == 2
    assert mat[0, 1].std_dev() == 0.2

def test_component_extraction():
    "Extracting the nominal values and standard deviations from an array"

    arr = unumpy.uarray(([1, 2], [0.1, 0.2]))

    assert numpy.all(unumpy.nominal_values(arr) == [1, 2])
    assert numpy.all(unumpy.std_devs(arr) == [0.1, 0.2])

    # unumpy matrices, in addition, should have nominal_values that
    # are simply numpy matrices (not unumpy ones, because they have no
    # uncertainties):
    mat = unumpy.matrix(arr)
    assert numpy.all(unumpy.nominal_values(mat) == [1, 2])
    assert numpy.all(unumpy.std_devs(mat) == [0.1, 0.2])
    assert type(unumpy.nominal_values(mat)) == numpy.matrix


def test_array_comparisons():
    "Test of array and matrix comparisons"

    arr = unumpy.uarray(([1, 2], [1, 4]))
    assert numpy.all((arr == [arr[0], 4]) == [True, False])

    # For matrices, 1D arrays are converted to 2D arrays:
    mat = unumpy.umatrix(([1, 2], [1, 4]))
    assert numpy.all((mat == [mat[0, 0], 4]) == [True, False])

