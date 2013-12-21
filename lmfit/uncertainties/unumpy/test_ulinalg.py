"""
Tests for uncertainties.unumpy.ulinalg.

These tests can be run through the Nose testing framework.

(c) 2010-2013 by Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>.
"""

# Some tests are already performed in test_unumpy (unumpy contains a
# matrix inversion, for instance).  They are not repeated here.

from __future__ import division

try:
    import numpy
except ImportError:
    import sys
    sys.exit()  # There is no reason to test the interface to NumPy

from uncertainties import unumpy, ufloat
from uncertainties.unumpy.test_unumpy import matrices_close

from uncertainties import __author__

def test_list_inverse():
    "Test of the inversion of a square matrix"

    mat_list = [[1, 1], [1, 0]]

    # numpy.linalg.inv(mat_list) does calculate the inverse even
    # though mat_list is a list of lists (and not a matrix).  Can
    # ulinalg do the same?  Here is a test:
    mat_list_inv = unumpy.ulinalg.inv(mat_list)

    # More type testing:
    mat_matrix = numpy.asmatrix(mat_list)
    assert isinstance(unumpy.ulinalg.inv(mat_matrix),
                      type(numpy.linalg.inv(mat_matrix)))

    # unumpy.ulinalg should behave in the same way as numpy.linalg,
    # with respect to types:
    mat_list_inv_numpy = numpy.linalg.inv(mat_list)
    assert type(mat_list_inv) == type(mat_list_inv_numpy)

    # The resulting matrix does not have to be a matrix that can
    # handle uncertainties, because the input matrix does not have
    # uncertainties:
    assert not isinstance(mat_list_inv, unumpy.matrix)

    # Individual element check:
    assert isinstance(mat_list_inv[1, 1], float)
    assert mat_list_inv[1, 1] == -1

    x = ufloat((1, 0.1))
    y = ufloat((2, 0.1))
    mat = unumpy.matrix([[x, x], [y, 0]])

    # Internal consistency: ulinalg.inv() must coincide with the
    # unumpy.matrix inverse, for square matrices (.I is the
    # pseudo-inverse, for non-square matrices, but inv() is not).
    assert matrices_close(unumpy.ulinalg.inv(mat), mat.I)


def test_list_pseudo_inverse():
    "Test of the pseudo-inverse"

    x = ufloat((1, 0.1))
    y = ufloat((2, 0.1))
    mat = unumpy.matrix([[x, x], [y, 0]])

    # Internal consistency: the inverse and the pseudo-inverse yield
    # the same result on square matrices:
    assert matrices_close(mat.I, unumpy.ulinalg.pinv(mat), 1e-4)
    assert matrices_close(unumpy.ulinalg.inv(mat),
                          # Support for the optional pinv argument is
                          # tested:
                          unumpy.ulinalg.pinv(mat, 1e-15), 1e-4)

    # Non-square matrices:
    x = ufloat((1, 0.1))
    y = ufloat((2, 0.1))
    mat1 = unumpy.matrix([[x, y]])  # "Long" matrix
    mat2 = unumpy.matrix([[x, y], [1, 3+x], [y, 2*x]])  # "Tall" matrix

    # Internal consistency:
    assert matrices_close(mat1.I, unumpy.ulinalg.pinv(mat1, 1e-10))
    assert matrices_close(mat2.I, unumpy.ulinalg.pinv(mat2, 1e-8))
