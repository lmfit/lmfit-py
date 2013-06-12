"""
Utilities for NumPy arrays and matrices that contain numbers with
uncertainties.

This package contains:

1) utilities that help with the creation and manipulation of NumPy
arrays and matrices of numbers with uncertainties;

2) generalizations of multiple NumPy functions so that they also work
with arrays that contain numbers with uncertainties.

- Arrays of numbers with uncertainties can be built as follows:

  arr = unumpy.uarray(([1, 2], [0.01, 0.002]))  # (values, uncertainties)

NumPy arrays of numbers with uncertainties can also be built directly
through NumPy, thanks to NumPy's support of arrays of arbitrary objects:

  arr = numpy.array([uncertainties.ufloat((1, 0.1)),...])
  
- Matrices of numbers with uncertainties are best created in one of
two ways:

  mat = unumpy.umatrix(([1, 2], [0.01, 0.002]))  # (values, uncertainties)

Matrices can also be built by converting arrays of numbers with
uncertainties, through the unumpy.matrix class:

  mat = unumpy.matrix(arr)

unumpy.matrix objects behave like numpy.matrix objects of numbers with
uncertainties, but with better support for some operations (such as
matrix inversion):

  # The inverse or pseudo-inverse of a unumpy.matrix can be calculated:
  print mat.I  # Would not work with numpy.matrix([[ufloat(...),...]]).I

- Nominal values and uncertainties of arrays can be directly accessed:

  print unumpy.nominal_values(arr)  # [ 1.  2.]
  print unumpy.std_devs(mat)  # [ 0.01   0.002]

- This module defines uncertainty-aware mathematical functions that
generalize those from uncertainties.umath so that they work on NumPy
arrays of numbers with uncertainties instead of just scalars:

  print unumpy.cos(arr)  # Array with the cosine of each element

NumPy's function names are used, and not those of the math module (for
instance, unumpy.arccos is defined, like in NumPy, and is not named
acos like in the standard math module).

The definitions of the mathematical quantities calculated by these
functions are available in the documentation of uncertainties.umath.

- The unumpy.ulinalg module contains more uncertainty-aware functions
for arrays that contain numbers with uncertainties (see the
documentation for this module).

This module requires the NumPy package.

(c) 2009-2013 by Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author."""

# Local modules:
from core import *
from uncertainties.unumpy import core
from uncertainties.unumpy import ulinalg  # Local sub-module

from uncertainties import __author__

# __all__ is set so that pydoc shows all important functions:
__all__ = core.__all__
# "import numpy" makes numpy.linalg available.  This behavior is
# copied here, for maximum compatibility:
__all__.append('ulinalg')

