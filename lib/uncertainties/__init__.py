#!! Whenever the documentation below is updated, setup.py should be
# checked for consistency.

'''
Calculations with full error propagation for quantities with uncertainties.
Derivatives can also be calculated.

Web user guide: http://packages.python.org/uncertainties/.

Example of possible calculation: (0.2 +/- 0.01)**2 = 0.04 +/- 0.004.

Correlations between expressions are correctly taken into account (for
instance, with x = 0.2+/-0.01, 2*x-x-x is exactly zero, as is y-x-x
with y = 2*x).

Examples:

  import uncertainties
  from uncertainties import ufloat
  from uncertainties.umath import *  # sin(), etc.

  # Mathematical operations:
  x = ufloat((0.20, 0.01))  # x = 0.20+/-0.01
  x = ufloat("0.20+/-0.01")  # Other representation
  x = ufloat("0.20(1)")  # Other representation
  x = ufloat("0.20")  # Implicit uncertainty of +/-1 on the last digit
  print x**2  # Square: prints "0.04+/-0.004"
  print sin(x**2)  # Prints "0.0399...+/-0.00399..."

  print x.std_score(0.17)  # Prints "-3.0": deviation of -3 sigmas

  # Access to the nominal value, and to the uncertainty:
  square = x**2  # Square
  print square  # Prints "0.04+/-0.004"
  print square.nominal_value  # Prints "0.04"
  print square.std_dev()  # Prints "0.004..."

  print square.derivatives[x]  # Partial derivative: 0.4 (= 2*0.20)

  # Correlations:
  u = ufloat((1, 0.05), "u variable")  # Tag
  v = ufloat((10, 0.1), "v variable")
  sum_value = u+v

  u.set_std_dev(0.1)  # Standard deviations can be updated on the fly
  print sum_value - u - v  # Prints "0.0" (exact result)

  # List of all sources of error:
  print sum_value  # Prints "11+/-0.1414..."
  for (var, error) in sum_value.error_components().iteritems():
      print "%s: %f" % (var.tag, error)  # Individual error components

  # Covariance matrices:
  cov_matrix = uncertainties.covariance_matrix([u, v, sum_value])
  print cov_matrix  # 3x3 matrix

  # Correlated variables can be constructed from a covariance matrix, if
  # NumPy is available:
  (u2, v2, sum2) = uncertainties.correlated_values([1, 10, 11],
                                                   cov_matrix)
  print u2  # Value and uncertainty of u: correctly recovered (1+/-0.1)
  print uncertainties.covariance_matrix([u2, v2, sum2])  # == cov_matrix

- The main function provided by this module is ufloat, which creates
numbers with uncertainties (Variable objects).  Variable objects can
be used as if they were regular Python numbers.  The main attributes
and methods of Variable objects are defined in the documentation of
the Variable class.

- Valid operations on numbers with uncertainties include basic
mathematical functions (addition, etc.).

Most operations from the standard math module (sin, etc.) can be applied
on numbers with uncertainties by using their generalization from the
uncertainties.umath module:

  from uncertainties.umath import sin
  print sin(ufloat("1+/-0.01"))  # 0.841...+/-0.005...
  print sin(1)  # umath.sin() also works on floats, exactly like math.sin()

Logical operations (>, ==, etc.) are also supported.

Basic operations on NumPy arrays or matrices of numbers with
uncertainties can be performed:

  2*numpy.array([ufloat((1, 0.01)), ufloat((2, 0.1))])

More complex operations on NumPy arrays can be performed through the
dedicated uncertainties.unumpy sub-module (see its documentation).

Calculations that are performed through non-Python code (Fortran, C,
etc.) can handle numbers with uncertainties instead of floats through
the provided wrap() wrapper:

  import uncertainties

  # wrapped_f is a version of f that can take arguments with
  # uncertainties, even if f only takes floats:
  wrapped_f = uncertainties.wrap(f)

If some derivatives of the wrapped function f are known (analytically,
or numerically), they can be given to wrap()--see the documentation
for wrap().

- Utility functions are also provided: the covariance matrix between
random variables can be calculated with covariance_matrix(), or used
as input for the definition of correlated quantities (correlated_values()
function--defined only if the NumPy module is available).

- Mathematical expressions involving numbers with uncertainties
generally return AffineScalarFunc objects, which also print as a value
with uncertainty.  Their most useful attributes and methods are
described in the documentation for AffineScalarFunc.  Note that
Variable objects are also AffineScalarFunc objects.  UFloat is an
alias for AffineScalarFunc, provided as a convenience: testing whether
a value carries an uncertainty handled by this module should be done
with insinstance(my_value, UFloat).

- Mathematically, numbers with uncertainties are, in this package,
probability distributions.  These probabilities are reduced to two
numbers: a nominal value and an uncertainty.  Thus, both variables
(Variable objects) and the result of mathematical operations
(AffineScalarFunc objects) contain these two values (respectively in
their nominal_value attribute and through their std_dev() method).

The uncertainty of a number with uncertainty is simply defined in
this package as the standard deviation of the underlying probability
distribution.

The numbers with uncertainties manipulated by this package are assumed
to have a probability distribution mostly contained around their
nominal value, in an interval of about the size of their standard
deviation.  This should cover most practical cases.  A good choice of
nominal value for a number with uncertainty is thus the median of its
probability distribution, the location of highest probability, or the
average value.

- When manipulating ensembles of numbers, some of which contain
uncertainties, it can be useful to access the nominal value and
uncertainty of all numbers in a uniform manner:

  x = ufloat("3+/-0.1")
  print nominal_value(x)  # Prints 3
  print std_dev(x)  # Prints 0.1
  print nominal_value(3)  # Prints 3: nominal_value works on floats
  print std_dev(3)  # Prints 0: std_dev works on floats

- Probability distributions (random variables and calculation results)
are printed as:

  nominal value +/- standard deviation

but this does not imply any property on the nominal value (beyond the
fact that the nominal value is normally inside the region of high
probability density), or that the probability distribution of the
result is symmetrical (this is rarely strictly the case).

- Linear approximations of functions (around the nominal values) are
used for the calculation of the standard deviation of mathematical
expressions with this package.

The calculated standard deviations and nominal values are thus
meaningful approximations as long as the functions involved have
precise linear expansions in the region where the probability
distribution of their variables is the largest.  It is therefore
important that uncertainties be small.  Mathematically, this means
that the linear term of functions around the nominal values of their
variables should be much larger than the remaining higher-order terms
over the region of significant probability.

For instance, sin(0+/-0.01) yields a meaningful standard deviation
since it is quite linear over 0+/-0.01.  However, cos(0+/-0.01) yields
an approximate standard deviation of 0 (because the cosine is not well
approximated by a line around 0), which might not be precise enough
for all applications.

- Comparison operations (>, ==, etc.) on numbers with uncertainties
have a pragmatic semantics, in this package: numbers with
uncertainties can be used wherever Python numbers are used, most of
the time with a result identical to the one that would be obtained
with their nominal value only.  However, since the objects defined in
this module represent probability distributions and not pure numbers,
comparison operator are interpreted in a specific way.

The result of a comparison operation ("==", ">", etc.) is defined so as
to be essentially consistent with the requirement that uncertainties
be small: the value of a comparison operation is True only if the
operation yields True for all infinitesimal variations of its random
variables, except, possibly, for an infinitely small number of cases.

Example:

  "x = 3.14; y = 3.14" is such that x == y

but

  x = ufloat((3.14, 0.01))
  y = ufloat((3.14, 0.01))

is not such that x == y, since x and y are independent random
variables that almost never give the same value.  However, x == x
still holds.

The boolean value (bool(x), "if x...") of a number with uncertainty x
is the result of x != 0.

- The uncertainties package is for Python 2.5 and above.

- This package contains tests.  They can be run either manually or
automatically with the nose unit testing framework (nosetests).

(c) 2009-2013 by Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>.
Please send feature requests, bug reports, or feedback to this address.

Please support future development by donating $5 or more through PayPal!

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.'''

# The idea behind this module is to replace the result of mathematical
# operations by a local approximation of the defining function.  For
# example, sin(0.2+/-0.01) becomes the affine function
# (AffineScalarFunc object) whose nominal value is sin(0.2) and
# whose variations are given by sin(0.2+delta) = 0.98...*delta.
# Uncertainties can then be calculated by using this local linear
# approximation of the original function.

from __future__ import division  # Many analytical derivatives depend on this

import re
import math
from math import sqrt, log  # Optimization: no attribute look-up
import copy
import warnings

# Numerical version:
__version_info__ = (1, 9)
__version__ = '.'.join(map(str, __version_info__))

__author__ = 'Eric O. LEBIGOT (EOL) <eric.lebigot@normalesup.org>'

# Attributes that are always exported (some other attributes are
# exported only if the NumPy module is available...):
__all__ = [

    # All sub-modules and packages are not imported by default,
    # in particular because NumPy might be unavailable.
    'ufloat',  # Main function: returns a number with uncertainty

    # Uniform access to nominal values and standard deviations:
    'nominal_value',
    'std_dev',

    # Utility functions (more are exported if NumPy is present):
    'covariance_matrix',

    # Class for testing whether an object is a number with
    # uncertainty.  Not usually created by users (except through the
    # Variable subclass), but possibly manipulated by external code
    # ['derivatives()' method, etc.].
    'UFloat',

    # Wrapper for allowing non-pure-Python function to handle
    # quantities with uncertainties:
    'wrap',

    # The documentation for wrap() indicates that numerical
    # derivatives are calculated through partial_derivative().  The
    # user might also want to change the size of the numerical
    # differentiation step.
    'partial_derivative'
    ]

###############################################################################

def set_doc(doc_string):
    """
    Decorator function that sets the docstring to the given text.

    It is useful for functions whose docstring is calculated
    (including string substitutions).
    """
    def set_doc_string(func):
        func.__doc__ = doc_string
        return func
    return set_doc_string

# Some types known to not depend on Variable objects are put in
# CONSTANT_TYPES.  The most common types can be put in front, as this
# may slightly improve the execution speed.
CONSTANT_TYPES = (float, int, complex) # , long)

###############################################################################
# Utility for issuing deprecation warnings

def deprecation(message):
    '''
    Warns the user with the given message, by issuing a
    DeprecationWarning.
    '''
    warnings.warn(message, DeprecationWarning, stacklevel=2)


###############################################################################

## Definitions that depend on the availability of NumPy:


try:
    import numpy
except ImportError:
    pass
else:

    # NumPy numbers do not depend on Variable objects:
    CONSTANT_TYPES += (numpy.number,)

    # Entering variables as a block of correlated values.  Only available
    # if NumPy is installed.

    #! It would be possible to dispense with NumPy, but a routine should be
    # written for obtaining the eigenvectors of a symmetric matrix.  See
    # for instance Numerical Recipes: (1) reduction to tri-diagonal
    # [Givens or Householder]; (2) QR / QL decomposition.

    def correlated_values(nom_values, covariance_mat, tags=None):
        """
        Returns numbers with uncertainties (AffineScalarFunc objects)
        that correctly reproduce the given covariance matrix, and have
        the given (float) values as their nominal value.

        The correlated_values_norm() function returns the same result,
        but takes a correlation matrix instead of a covariance matrix.

        The list of values and the covariance matrix must have the
        same length, and the matrix must be a square (symmetric) one.

        The numbers with uncertainties returned depend on newly
        created, independent variables (Variable objects).

        If 'tags' is not None, it must list the tag of each new
        independent variable.

        nom_values -- sequence with the nominal (real) values of the
        numbers with uncertainties to be returned.

        covariance_mat -- full covariance matrix of the returned
        numbers with uncertainties (not the statistical correlation
        matrix, i.e., not the normalized covariance matrix). For
        example, the first element of this matrix is the variance of
        the first returned number with uncertainty.
        """

        # If no tags were given, we prepare tags for the newly created
        # variables:
        if tags is None:
            tags = (None,) * len(nom_values)

        # The covariance matrix is diagonalized in order to define
        # the independent variables that model the given values:

        (variances, transform) = numpy.linalg.eigh(covariance_mat)

        # Numerical errors might make some variances negative: we set
        # them to zero:
        variances[variances < 0] = 0.

        # Creation of new, independent variables:

        # We use the fact that the eigenvectors in 'transform' are
        # special: 'transform' is unitary: its inverse is its transpose:

        variables = tuple(
            # The variables represent "pure" uncertainties:
            Variable(0, sqrt(variance), tag)
            for (variance, tag) in zip(variances, tags))

        # Representation of the initial correlated values:
        values_funcs = tuple(
            AffineScalarFunc(value, dict(zip(variables, coords)))
            for (coords, value) in zip(transform, nom_values))

        return values_funcs

    __all__.append('correlated_values')

    def correlated_values_norm(values_with_std_dev, correlation_mat,
                               tags=None):
        '''
        Returns correlated values like correlated_values(), but takes
        instead as input:

        - nominal (float) values along with their standard deviation, and

        - a correlation matrix (i.e. a normalized covariance matrix
          normalized with individual standard deviations).

        values_with_std_dev -- sequence of (nominal value, standard
        deviation) pairs. The returned, correlated values have these
        nominal values and standard deviations.

        correlation_mat -- correlation matrix (i.e. the normalized
        covariance matrix, a matrix with ones on its diagonal).
        '''

        (nominal_values, std_devs) = numpy.transpose(values_with_std_dev)

        return correlated_values(
            nominal_values,
            correlation_mat*std_devs*std_devs[numpy.newaxis].T,
            tags)

    __all__.append('correlated_values_norm')

###############################################################################

# Mathematical operations with local approximations (affine scalar
# functions)

class NotUpcast(Exception):
    'Raised when an object cannot be converted to a number with uncertainty'

def to_affine_scalar(x):
    """
    Transforms x into a constant affine scalar function
    (AffineScalarFunc), unless it is already an AffineScalarFunc (in
    which case x is returned unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on AffineScalarFunc objects
    (which then cannot be considered as constants).
    """

    if isinstance(x, AffineScalarFunc):
        return x

    #! In Python 2.6+, numbers.Number could be used instead, here:
    if isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return AffineScalarFunc(x, {})

    # Case of lists, etc.
    raise NotUpcast("%s cannot be converted to a number with"
                    " uncertainty" % type(x))

def partial_derivative(f, param_num):
    """
    Returns a function that numerically calculates the partial
    derivative of function f with respect to its argument number
    param_num.

    The step parameter represents the shift of the parameter used in
    the numerical approximation.
    """

    def partial_derivative_of_f(*args, **kws):
        """
        Partial derivative, calculated with the (-epsilon, +epsilon)
        method, which is more precise than the (0, +epsilon) method.
        """
        # f_nominal_value = f(*args)
        param_kw = None
        if '__param__kw__' in kws:
            param_kw = kws.pop('__param__kw__')
        shifted_args = list(args)  # Copy, and conversion to a mutable
        shifted_kws  = {}
        for k, v in kws.items():
            shifted_kws[k] = v
        step = 1.e-8
        if param_kw in shifted_kws:
            step = step*abs(shifted_kws[param_kw])
        elif param_num < len(shifted_args):
            # The step is relative to the parameter being varied, so that
            # shsifting it does not suffer from finite precision:
            step = step*abs(shifted_args[param_num])

        if param_kw in shifted_kws:
            shifted_kws[param_kw] += step
        elif param_num < len(shifted_args):
            shifted_args[param_num] += step

        shifted_f_plus = f(*shifted_args, **shifted_kws)

        if param_kw in shifted_kws:
            shifted_kws[param_kw] -= 2*step
        elif param_num < len(shifted_args):
            shifted_args[param_num] -= 2*step
        shifted_f_minus = f(*shifted_args, **shifted_kws)

        return (shifted_f_plus - shifted_f_minus)/2/step

    return partial_derivative_of_f

class NumericalDerivatives(object):
    """
    Convenient access to the partial derivatives of a function,
    calculated numerically.
    """
    # This is not a list because the number of arguments of the
    # function is not known in advance, in general.

    def __init__(self, function):
        """
        'function' is the function whose derivatives can be computed.
        """
        self._function = function

    def __getitem__(self, n):
        """
        Returns the n-th numerical derivative of the function.
        """
        return partial_derivative(self._function, n)

def wrap(f, derivatives_iter=None):
    """
    Wraps a function f into a function that also accepts numbers with
    uncertainties (UFloat objects) and returns a number with
    uncertainties.  Doing so may be necessary when function f cannot
    be expressed analytically (with uncertainties-compatible operators
    and functions like +, *, umath.sin(), etc.).

    f must return a scalar (not a list, etc.).

    In the wrapped function, the standard Python scalar arguments of f
    (float, int, etc.) can be replaced by numbers with
    uncertainties. The result will contain the appropriate
    uncertainty.

    If no argument to the wrapped function has an uncertainty, f
    simply returns its usual, scalar result.

    If supplied, derivatives_iter can be an iterable that generally
    contains functions; each successive function is the partial
    derivative of f with respect to the corresponding variable (one
    function for each argument of f, which takes as many arguments as
    f).  If instead of a function, an element of derivatives_iter
    contains None, then it is automatically replaced by the relevant
    numerical derivative; this can be used for non-scalar arguments of
    f (like string arguments).

    If derivatives_iter is None, or if derivatives_iter contains a
    fixed (and finite) number of elements, then any missing derivative
    is calculated numerically.

    An infinite number of derivatives can be specified by having
    derivatives_iter be an infinite iterator; this can for instance
    be used for specifying the derivatives of functions with a
    undefined number of argument (like sum(), whose partial
    derivatives all return 1).

    Example (for illustration purposes only, as
    uncertainties.umath.sin() runs faster than the examples that
    follow): wrap(math.sin) is a sine function that can be applied to
    numbers with uncertainties.  Its derivative will be calculated
    numerically.  wrap(math.sin, [None]) would have produced the same
    result.  wrap(math.sin, [math.cos]) is the same function, but with
    an analytically defined derivative.
    """

    if derivatives_iter is None:
        derivatives_iter = NumericalDerivatives(f)
    else:
        # Derivatives that are not defined are calculated numerically,
        # if there is a finite number of them (the function lambda
        # *args: fsum(args) has a non-defined number of arguments, as
        # it just performs a sum):
        try:  # Is the number of derivatives fixed?
            len(derivatives_iter)
        except TypeError:
            pass
        else:
            derivatives_iter = [
                partial_derivative(f, k) if derivative is None
                else derivative
                for (k, derivative) in enumerate(derivatives_iter)]

    #! Setting the doc string after "def f_with...()" does not
    # seem to work.  We define it explicitly:
    @set_doc("""\
    Version of %s(...) that returns an affine approximation
    (AffineScalarFunc object), if its result depends on variables
    (Variable objects).  Otherwise, returns a simple constant (when
    applied to constant arguments).

    Warning: arguments of the function that are not AffineScalarFunc
    objects must not depend on uncertainties.Variable objects in any
    way.  Otherwise, the dependence of the result in
    uncertainties.Variable objects will be incorrect.

    Original documentation:
    %s""" % (f.__name__, f.__doc__))
    def f_with_affine_output(*args, **kwargs):
        # Can this function perform the calculation of an
        # AffineScalarFunc (or maybe float) result?
        try:
            old_funcs = map(to_affine_scalar, args)
            aff_funcs = [to_affine_scalar(a) for a in args]
            aff_kws = kwargs
            aff_varkws = []
            for key, val in kwargs.items():
                if isinstance(val, Variable):
                    aff_kws[key] = to_affine_scalar(val)
                    aff_varkws.append(key)

        except NotUpcast:

            # This function does not know how to itself perform
            # calculations with non-float-like arguments (as they
            # might for instance be objects whose value really changes
            # if some Variable objects had different values):

            # Is it clear that we can't delegate the calculation?

            if any(isinstance(arg, AffineScalarFunc) for arg in args):
                # This situation arises for instance when calculating
                # AffineScalarFunc(...)*numpy.array(...).  In this
                # case, we must let NumPy handle the multiplication
                # (which is then performed element by element):
                return NotImplemented
            else:
                # If none of the arguments is an AffineScalarFunc, we
                # can delegate the calculation to the original
                # function.  This can be useful when it is called with
                # only one argument (as in
                # numpy.log10(numpy.ndarray(...)):
                return f(*args, **kwargs)

        ########################################
        # Nominal value of the constructed AffineScalarFunc:
        args_values = [e.nominal_value for e in aff_funcs]
        kw_values = {}
        for key, val in aff_kws.items():
            kw_values[key] = val
            if key in aff_varkws:
                kw_values[key] = val.nominal_value
        f_nominal_value = f(*args_values, **kw_values)

        ########################################

        # List of involved variables (Variable objects):
        variables = set()
        for expr in aff_funcs:
            variables |= set(expr.derivatives)
        for vname  in aff_varkws:
            variables |= set(aff_kws[vname].derivatives)
        ## It is sometimes useful to only return a regular constant:

        # (1) Optimization / convenience behavior: when 'f' is called
        # on purely constant values (e.g., sin(2)), there is no need
        # for returning a more complex AffineScalarFunc object.

        # (2) Functions that do not return a "float-like" value might
        # not have a relevant representation as an AffineScalarFunc.
        # This includes boolean functions, since their derivatives are
        # either 0 or are undefined: they are better represented as
        # Python constants than as constant AffineScalarFunc functions.

        if not variables or isinstance(f_nominal_value, bool):
            return f_nominal_value

        # The result of 'f' does depend on 'variables'...

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (aff_funcs):

        # The chain rule is applied.  This is because, in the case of
        # numerical derivatives, it allows for a better-controlled
        # numerical stability than numerically calculating the partial
        # derivatives through '[f(x + dx, y + dy, ...) -
        # f(x,y,...)]/da' where dx, dy,... are calculated by varying
        # 'a'.  In fact, it is numerically better to control how big
        # (dx, dy,...) are: 'f' is a simple mathematical function and
        # it is possible to know how precise the df/dx are (which is
        # not possible with the numerical df/da calculation above).

        # We use numerical derivatives, if we don't already have a
        # list of derivatives:

        #! Note that this test could be avoided by requiring the
        # caller to always provide derivatives.  When changing the
        # functions of the math module, this would force this module
        # to know about all the math functions.  Another possibility
        # would be to force derivatives_iter to contain, say, the
        # first 3 derivatives of f.  But any of these two ideas has a
        # chance to break, one day... (if new functions are added to
        # the math module, or if some function has more than 3
        # arguments).

        derivatives_wrt_args = []
        for (arg, derivative) in zip(aff_funcs, derivatives_iter):
            derivatives_wrt_args.append(derivative(*args_values, **aff_kws)
                                        if arg.derivatives
                                        else 0)


        kws_values = []
        for vname in aff_varkws:
            kws_values.append( aff_kws[vname].nominal_value)
        for (vname, derivative) in zip(aff_varkws, derivatives_iter):
            derivatives_wrt_args.append(derivative(__param__kw__=vname,
                                                   **kw_values)
                                        if aff_kws[vname].derivatives
                                        else 0)

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        # Initial value (is updated below):
        derivatives_wrt_vars = dict((var, 0.) for var in variables)

        # The chain rule is used (we already have
        # derivatives_wrt_args):

        for (func, f_derivative) in zip(aff_funcs, derivatives_wrt_args):
            for (var, func_derivative) in func.derivatives.items():
                derivatives_wrt_vars[var] += f_derivative * func_derivative

        for (vname, f_derivative) in zip(aff_varkws, derivatives_wrt_args):
            func = aff_kws[vname]
            for (var, func_derivative) in func.derivatives.items():
                derivatives_wrt_vars[var] += f_derivative * func_derivative

        # The function now returns an AffineScalarFunc object:
        return AffineScalarFunc(f_nominal_value, derivatives_wrt_vars)

    # It is easier to work with f_with_affine_output, which represents
    # a wrapped version of 'f', when it bears the same name as 'f':
    f_with_affine_output.__name__ = f.__name__

    return f_with_affine_output

def _force_aff_func_args(func):
    """
    Takes an operator op(x, y) and wraps it.

    The constructed operator returns func(x, to_affine_scalar(y)) if y
    can be upcast with to_affine_scalar(); otherwise, it returns
    NotImplemented.

    Thus, func() is only called on two AffineScalarFunc objects, if
    its first argument is an AffineScalarFunc.
    """

    def op_on_upcast_args(x, y):
        """
        Returns %s(self, to_affine_scalar(y)) if y can be upcast
        through to_affine_scalar.  Otherwise returns NotImplemented.
        """ % func.__name__

        try:
            y_with_uncert = to_affine_scalar(y)
        except NotUpcast:
            # This module does not know how to handle the comparison:
            # (example: y is a NumPy array, in which case the NumPy
            # array will decide that func() should be applied
            # element-wise between x and all the elements of y):
            return NotImplemented
        else:
            return func(x, y_with_uncert)

    return op_on_upcast_args

########################################

# Definition of boolean operators, that assume that self and
# y_with_uncert are AffineScalarFunc.

# The fact that uncertainties must be smalled is used, here: the
# comparison functions are supposed to be constant for most values of
# the random variables.

# Even though uncertainties are supposed to be small, comparisons
# between 3+/-0.1 and 3.0 are handled (even though x == 3.0 is not a
# constant function in the 3+/-0.1 interval).  The comparison between
# x and x is handled too, when x has an uncertainty.  In fact, as
# explained in the main documentation, it is possible to give a useful
# meaning to the comparison operators, in these cases.

def _eq_on_aff_funcs(self, y_with_uncert):
    """
    __eq__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """
    difference = self - y_with_uncert
    # Only an exact zero difference means that self and y are
    # equal numerically:
    return not(difference._nominal_value or difference.std_dev())

def _ne_on_aff_funcs(self, y_with_uncert):
    """
    __ne__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """

    return not _eq_on_aff_funcs(self, y_with_uncert)

def _gt_on_aff_funcs(self, y_with_uncert):
    """
    __gt__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """
    return self._nominal_value > y_with_uncert._nominal_value

def _ge_on_aff_funcs(self, y_with_uncert):
    """
    __ge__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """

    return (_gt_on_aff_funcs(self, y_with_uncert)
            or _eq_on_aff_funcs(self, y_with_uncert))

def _lt_on_aff_funcs(self, y_with_uncert):
    """
    __lt__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """
    return self._nominal_value < y_with_uncert._nominal_value

def _le_on_aff_funcs(self, y_with_uncert):
    """
    __le__ operator, assuming that both self and y_with_uncert are
    AffineScalarFunc objects.
    """

    return (_lt_on_aff_funcs(self, y_with_uncert)
            or _eq_on_aff_funcs(self, y_with_uncert))

########################################

class AffineScalarFunc(object):
    """
    Affine functions that support basic mathematical operations
    (addition, etc.).  Such functions can for instance be used for
    representing the local (linear) behavior of any function.

    This class is mostly meant to be used internally.

    This class can also be used to represent constants.

    The variables of affine scalar functions are Variable objects.

    AffineScalarFunc objects include facilities for calculating the
    'error' on the function, from the uncertainties on its variables.

    Main attributes and methods:

    - nominal_value, std_dev(): value at the origin / nominal value,
      and standard deviation.

    - error_components(): error_components()[x] is the error due to
      Variable x.

    - derivatives: derivatives[x] is the (value of the) derivative
      with respect to Variable x.  This attribute is a dictionary
      whose keys are the Variable objects on which the function
      depends.

      All the Variable objects on which the function depends are in
      'derivatives'.

    - std_score(x): position of number x with respect to the
      nominal value, in units of the standard deviation.
    """

    # To save memory in large arrays:
    __slots__ = ('_nominal_value', 'derivatives')

    #! The code could be modify in order to accommodate for non-float
    # nominal values.  This could for instance be done through
    # the operator module: instead of delegating operations to
    # float.__*__ operations, they could be delegated to
    # operator.__*__ functions (while taking care of properly handling
    # reverse operations: __radd__, etc.).

    def __init__(self, nominal_value, derivatives):
        """
        nominal_value -- value of the function at the origin.
        nominal_value must not depend in any way of the Variable
        objects in 'derivatives' (the value at the origin of the
        function being defined is a constant).

        derivatives -- maps each Variable object on which the function
        being defined depends to the value of the derivative with
        respect to that variable, taken at the nominal value of all
        variables.

        Warning: the above constraint is not checked, and the user is
        responsible for complying with it.
        """

        # Defines the value at the origin:

        # Only float-like values are handled.  One reason is that it
        # does not make sense for a scalar function to be affine to
        # not yield float values.  Another reason is that it would not
        # make sense to have a complex nominal value, here (it would
        # not be handled correctly at all): converting to float should
        # be possible.
        self._nominal_value = float(nominal_value)
        self.derivatives = derivatives

    # The following prevents the 'nominal_value' attribute from being
    # modified by the user:
    @property
    def nominal_value(self):
        "Nominal value of the random number."
        return self._nominal_value

    ############################################################


    ### Operators: operators applied to AffineScalarFunc and/or
    ### float-like objects only are supported.  This is why methods
    ### from float are used for implementing these operators.

    # Operators with no reflection:

    ########################################

    # __nonzero__() is supposed to return a boolean value (it is used
    # by bool()).  It is for instance used for converting the result
    # of comparison operators to a boolean, in sorted().  If we want
    # to be able to sort AffineScalarFunc objects, __nonzero__ cannot
    # return a AffineScalarFunc object.  Since boolean results (such
    # as the result of bool()) don't have a very meaningful
    # uncertainty unless it is zero, this behavior is fine.

    def __nonzero__(self):
        """
        Equivalent to self != 0.
        """
        #! This might not be relevant for AffineScalarFunc objects
        # that contain values in a linear space which does not convert
        # the float 0 into the null vector (see the __eq__ function:
        # __nonzero__ works fine if subtracting the 0 float from a
        # vector of the linear space works as if 0 were the null
        # vector of that space):
        return self != 0.  # Uses the AffineScalarFunc.__ne__ function

    ########################################

    ## Logical operators: warning: the resulting value cannot always
    ## be differentiated.

    # The boolean operations are not differentiable everywhere, but
    # almost...

    # (1) I can rely on the assumption that the user only has "small"
    # errors on variables, as this is used in the calculation of the
    # standard deviation (which performs linear approximations):

    # (2) However, this assumption is not relevant for some
    # operations, and does not have to hold, in some cases.  This
    # comes from the fact that logical operations (e.g. __eq__(x,y))
    # are not differentiable for many usual cases.  For instance, it
    # is desirable to have x == x for x = n+/-e, whatever the size of e.
    # Furthermore, n+/-e != n+/-e', if e != e', whatever the size of e or
    # e'.

    # (3) The result of logical operators does not have to be a
    # function with derivatives, as these derivatives are either 0 or
    # don't exist (i.e., the user should probably not rely on
    # derivatives for his code).

    # __eq__ is used in "if data in [None, ()]", for instance.  It is
    # therefore important to be able to handle this case too, which is
    # taken care of when _force_aff_func_args(_eq_on_aff_funcs)
    # returns NotImplemented.
    __eq__ = _force_aff_func_args(_eq_on_aff_funcs)

    __ne__ = _force_aff_func_args(_ne_on_aff_funcs)
    __gt__ = _force_aff_func_args(_gt_on_aff_funcs)

    # __ge__ is not the opposite of __lt__ because these operators do
    # not always yield a boolean (for instance, 0 <= numpy.arange(10)
    # yields an array).
    __ge__ = _force_aff_func_args(_ge_on_aff_funcs)

    __lt__ = _force_aff_func_args(_lt_on_aff_funcs)
    __le__ = _force_aff_func_args(_le_on_aff_funcs)

    ########################################

    # Uncertainties handling:

    def error_components(self):
        """
        Individual components of the standard deviation of the affine
        function (in absolute value), returned as a dictionary with
        Variable objects as keys.

        This method assumes that the derivatives contained in the
        object take scalar values (and are not a tuple, like what
        math.frexp() returns, for instance).
        """

        # Calculation of the variance:
        error_components = {}
        for (variable, derivative) in self.derivatives.items():
            # Individual standard error due to variable:
            error_components[variable] = abs(derivative*variable._std_dev)

        return error_components

    def std_dev(self):
        """
        Standard deviation of the affine function.

        This method assumes that the function returns scalar results.

        This returned standard deviation depends on the current
        standard deviations [std_dev()] of the variables (Variable
        objects) involved.
        """
        #! It would be possible to not allow the user to update the
        #std dev of Variable objects, in which case AffineScalarFunc
        #objects could have a pre-calculated or, better, cached
        #std_dev value (in fact, many intermediate AffineScalarFunc do
        #not need to have their std_dev calculated: only the final
        #AffineScalarFunc returned to the user does).
        return sqrt(sum(
            delta**2 for delta in self.error_components().itervalues()))

    def _general_representation(self, to_string):
        """
        Uses the to_string() conversion function on both the nominal
        value and the standard deviation, and returns a string that
        describes them.

        to_string() is typically repr() or str().
        """

        (nominal_value, std_dev) = (self._nominal_value, self.std_dev())

        # String representation:

        # Not putting spaces around "+/-" helps with arrays of
        # Variable, as each value with an uncertainty is a
        # block of signs (otherwise, the standard deviation can be
        # mistaken for another element of the array).

        return ("%s+/-%s" % (to_string(nominal_value), to_string(std_dev))
                if std_dev
                else to_string(nominal_value))

    def __repr__(self):
        return self._general_representation(repr)

    def __str__(self):
        return self._general_representation(str)

    def std_score(self, value):
        """
        Returns 'value' - nominal value, in units of the standard
        deviation.

        Raises a ValueError exception if the standard deviation is zero.
        """
        try:
            # The ._nominal_value is a float: there is no integer division,
            # here:
            return (value - self._nominal_value) / self.std_dev()
        except ZeroDivisionError:
            raise ValueError("The standard deviation is zero:"
                             " undefined result.")

    def __deepcopy__(self, memo):
        """
        Hook for the standard copy module.

        The returned AffineScalarFunc is a completely fresh copy,
        which is fully independent of any variable defined so far.
        New variables are specially created for the returned
        AffineScalarFunc object.
        """
        return AffineScalarFunc(
            self._nominal_value,
            dict((copy.deepcopy(var), deriv)
                 for (var, deriv) in self.derivatives.items()))

    def __getstate__(self):
        """
        Hook for the pickle module.
        """
        obj_slot_values = dict((k, getattr(self, k)) for k in
                               # self.__slots__ would not work when
                               # self is an instance of a subclass:
                               AffineScalarFunc.__slots__)
        return obj_slot_values

    def __setstate__(self, data_dict):
        """
        Hook for the pickle module.
        """
        for (name, value) in data_dict.items():
            setattr(self, name, value)

# Nicer name, for users: isinstance(ufloat(...), UFloat) is True:
UFloat = AffineScalarFunc

def get_ops_with_reflection():

    """
    Returns operators with a reflection, along with their derivatives
    (for float operands).
    """

    # Operators with a reflection:

    # We do not include divmod().  This operator could be included, by
    # allowing its result (a tuple) to be differentiated, in
    # derivative_value().  However, a similar result can be achieved
    # by the user by calculating separately the division and the
    # result.

    # {operator(x, y): (derivative wrt x, derivative wrt y)}:

    # Note that unknown partial derivatives can be numerically
    # calculated by expressing them as something like
    # "partial_derivative(float.__...__, 1)(x, y)":

    # String expressions are used, so that reversed operators are easy
    # to code, and execute relatively efficiently:

    derivatives_list = {
        'add': ("1.", "1."),
        # 'div' is the '/' operator when __future__.division is not in
        # effect.  Since '/' is applied to
        # AffineScalarFunc._nominal_value numbers, it is applied on
        # floats, and is therefore the "usual" mathematical division.
        'div': ("1/y", "-x/y**2"),
        'floordiv': ("0.", "0."),  # Non exact: there is a discontinuities
        # The derivative wrt the 2nd arguments is something like (..., x//y),
        # but it is calculated numerically, for convenience:
        'mod': ("1.", "partial_derivative(float.__mod__, 1)(x, y)"),
        'mul': ("y", "x"),
        'pow': ("y*x**(y-1)", "log(x)*x**y"),
        'sub': ("1.", "-1."),
        'truediv': ("1/y", "-x/y**2")
        }

    # Conversion to Python functions:
    ops_with_reflection = {}
    for (op, derivatives) in derivatives_list.items():
        ops_with_reflection[op] = [
            eval("lambda x, y: %s" % expr) for expr in derivatives ]

        ops_with_reflection["r"+op] = [
            eval("lambda y, x: %s" % expr) for expr in reversed(derivatives)]

    return ops_with_reflection

# Operators that have a reflection, along with their derivatives:
_ops_with_reflection = get_ops_with_reflection()

# Some effectively modified operators (for the automated tests):
_modified_operators = []
_modified_ops_with_reflection = []

def add_operators_to_AffineScalarFunc():
    """
    Adds many operators (__add__, etc.) to the AffineScalarFunc class.
    """

    ########################################

    #! Derivatives are set to return floats.  For one thing,
    # uncertainties generally involve floats, as they are based on
    # small variations of the parameters.  It is also better to
    # protect the user from unexpected integer result that behave
    # badly with the division.

    ## Operators that return a numerical value:

    # Single-argument operators that should be adapted from floats to
    # AffineScalarFunc objects, associated to their derivative:
    simple_numerical_operators_derivatives = {
        'abs': lambda x: 1. if x>=0 else -1.,
        'neg': lambda x: -1.,
        'pos': lambda x: 1.,
        'trunc': lambda x: 0.
        }

    for (op, derivative) in (
          simple_numerical_operators_derivatives.items()):

        attribute_name = "__%s__" % op
        # float objects don't exactly have the same attributes between
        # different versions of Python (for instance, __trunc__ was
        # introduced with Python 2.6):
        try:
            setattr(AffineScalarFunc, attribute_name,
                    wrap(getattr(float, attribute_name),
                                 [derivative]))
        except AttributeError:
            pass
        else:
            _modified_operators.append(op)

    ########################################

    # Reversed versions (useful for float*AffineScalarFunc, for instance):
    for (op, derivatives) in _ops_with_reflection.items():
        attribute_name = '__%s__' % op
        # float objects don't exactly have the same attributes between
        # different versions of Python (for instance, __div__ and
        # __rdiv__ were removed, in Python 3):
        try:
            setattr(AffineScalarFunc, attribute_name,
                    wrap(getattr(float, attribute_name), derivatives))
        except AttributeError:
            pass
        else:
            _modified_ops_with_reflection.append(op)

    ########################################
    # Conversions to pure numbers are meaningless.  Note that the
    # behavior of float(1j) is similar.
    for coercion_type in ('complex', 'int', 'long', 'float'):
        def raise_error(self):
            raise TypeError("can't convert an affine function (%s)"
                            ' to %s; use x.nominal_value'
                            # In case AffineScalarFunc is sub-classed:
                            % (self.__class__, coercion_type))

        setattr(AffineScalarFunc, '__%s__' % coercion_type, raise_error)

add_operators_to_AffineScalarFunc()  # Actual addition of class attributes

class Variable(AffineScalarFunc):
    """
    Representation of a float-like scalar random variable, along with
    its uncertainty.

    Objects are meant to represent variables that are independent from
    each other (correlations are handled through the AffineScalarFunc
    class).
    """

    # To save memory in large arrays:
    __slots__ = ('_std_dev', 'tag')

    def __init__(self, value, std_dev, tag=None):
        """
        The nominal value and the standard deviation of the variable
        are set.  These values must be scalars.

        'tag' is a tag that the user can associate to the variable.  This
        is useful for tracing variables.

        The meaning of the nominal value is described in the main
        module documentation.
        """

        #! The value, std_dev, and tag are assumed by __copy__() not to
        # be copied.  Either this should be guaranteed here, or __copy__
        # should be updated.

        # Only float-like values are handled.  One reason is that the
        # division operator on integers would not produce a
        # differentiable functions: for instance, Variable(3, 0.1)/2
        # has a nominal value of 3/2 = 1, but a "shifted" value
        # of 3.1/2 = 1.55.
        value = float(value)

        # If the variable changes by dx, then the value of the affine
        # function that gives its value changes by 1*dx:

        # ! Memory cycles are created.  However, they are garbage
        # collected, if possible.  Using a weakref.WeakKeyDictionary
        # takes much more memory.  Thus, this implementation chooses
        # more cycles and a smaller memory footprint instead of no
        # cycles and a larger memory footprint.

        # ! Using AffineScalarFunc instead of super() results only in
        # a 3 % speed loss (Python 2.6, Mac OS X):
        super(Variable, self).__init__(value, {self: 1.})

        # We force the error to be float-like.  Since it is considered
        # as a Gaussian standard deviation, it is semantically
        # positive (even though there would be no problem defining it
        # as a sigma, where sigma can be negative and still define a
        # Gaussian):

        assert std_dev >= 0, "the error must be a positive number"
        # Since AffineScalarFunc.std_dev is a property, we cannot do
        # "self.std_dev = ...":
        self._std_dev = std_dev

        self.tag = tag

    # Standard deviations can be modified (this is a feature).
    # AffineScalarFunc objects that depend on the Variable have their
    # std_dev() automatically modified (recalculated with the new
    # std_dev of their Variables):
    def set_std_dev(self, value):
        """
        Updates the standard deviation of the variable to a new value.
        """

        # A zero variance is accepted.  Thus, it is possible to
        # conveniently use infinitely precise variables, for instance
        # to study special cases.

        self._std_dev = value

    # The following method is overridden so that we can represent the tag:
    def _general_representation(self, to_string):
        """
        Uses the to_string() conversion function on both the nominal
        value and standard deviation and returns a string that
        describes the number.

        to_string() is typically repr() or str().
        """
        num_repr  = super(Variable, self)._general_representation(to_string)

        # Optional tag: only full representations (to_string == repr)
        # contain the tag, as the tag is required in order to recreate
        # the variable.  Outputting the tag for regular string ("print
        # x") would be too heavy and produce an unusual representation
        # of a number with uncertainty.
        return (num_repr if ((self.tag is None) or (to_string != repr))
                else "< %s = %s >" % (self.tag, num_repr))

    def __hash__(self):
        # All Variable objects are by definition independent
        # variables, so they never compare equal; therefore, their
        # id() are therefore allowed to differ
        # (http://docs.python.org/reference/datamodel.html#object.__hash__):
        return id(self)

    def __copy__(self):
        """
        Hook for the standard copy module.
        """

        # This copy implicitly takes care of the reference of the
        # variable to itself (in self.derivatives): the new Variable
        # object points to itself, not to the original Variable.

        # Reference: http://www.doughellmann.com/PyMOTW/copy/index.html

        #! The following assumes that the arguments to Variable are
        # *not* copied upon construction, since __copy__ is not supposed
        # to copy "inside" information:
        return Variable(self.nominal_value, self.std_dev(), self.tag)

    def __deepcopy__(self, memo):
        """
        Hook for the standard copy module.

        A new variable is created.
        """

        # This deep copy implicitly takes care of the reference of the
        # variable to itself (in self.derivatives): the new Variable
        # object points to itself, not to the original Variable.

        # Reference: http://www.doughellmann.com/PyMOTW/copy/index.html

        return self.__copy__()

    def __getstate__(self):
        """
        Hook for the standard pickle module.
        """
        obj_slot_values = dict((k, getattr(self, k)) for k in self.__slots__)
        obj_slot_values.update(AffineScalarFunc.__getstate__(self))
        # Conversion to a usual dictionary:
        return obj_slot_values

    def __setstate__(self, data_dict):
        """
        Hook for the standard pickle module.
        """
        for (name, value) in data_dict.items():
            setattr(self, name, value)

###############################################################################

# Utilities

def nominal_value(x):
    """
    Returns the nominal value of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    x unchanged.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """

    return x.nominal_value if isinstance(x, AffineScalarFunc) else x

def std_dev(x):
    """
    Returns the standard deviation of x if it is a quantity with
    uncertainty (i.e., an AffineScalarFunc object); otherwise, returns
    the float 0.

    This utility function is useful for transforming a series of
    numbers, when only some of them generally carry an uncertainty.
    """

    return x.std_dev() if isinstance(x, AffineScalarFunc) else 0.

def covariance_matrix(nums_with_uncert):
    """
    Returns a matrix that contains the covariances between the given
    sequence of numbers with uncertainties (AffineScalarFunc objects).
    The resulting matrix implicitly depends on their ordering in
    'nums_with_uncert'.

    The covariances are floats (never int objects).

    The returned covariance matrix is the exact linear approximation
    result, if the nominal values of the numbers with uncertainties
    and of their variables are their mean.  Otherwise, the returned
    covariance matrix should be close to its linear approximation
    value.

    The returned matrix is a list of lists.
    """
    # See PSI.411 in EOL's notes.

    covariance_matrix = []
    for (i1, expr1) in enumerate(nums_with_uncert):
        derivatives1 = expr1.derivatives  # Optimization
        vars1 = set(derivatives1)
        coefs_expr1 = []
        for (i2, expr2) in enumerate(nums_with_uncert[:i1+1]):
            derivatives2 = expr2.derivatives  # Optimization
            coef = 0.
            for var in vars1.intersection(derivatives2):
                # var is a variable common to both numbers with
                # uncertainties:
                coef += (derivatives1[var]*derivatives2[var]*var._std_dev**2)
            coefs_expr1.append(coef)
        covariance_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for (i, covariance_coefs) in enumerate(covariance_matrix):
        covariance_coefs.extend(covariance_matrix[j][i]
                                for j in range(i+1, len(covariance_matrix)))

    return covariance_matrix

try:
    import numpy
except ImportError:
    pass
else:
    def correlation_matrix(nums_with_uncert):
        '''
        Returns the correlation matrix of the given sequence of
        numbers with uncertainties, as a NumPy array of floats.
        '''

        cov_mat = numpy.array(covariance_matrix(nums_with_uncert))

        std_devs = numpy.sqrt(cov_mat.diagonal())

        return cov_mat/std_devs/std_devs[numpy.newaxis].T

    __all__.append('correlation_matrix')

###############################################################################
# Parsing of values with uncertainties:

POSITIVE_DECIMAL_UNSIGNED = r'(\d+)(\.\d*)?'

# Regexp for a number with uncertainty (e.g., "-1.234(2)e-6"), where the
# uncertainty is optional (in which case the uncertainty is implicit):
NUMBER_WITH_UNCERT_RE_STR = '''
    ([+-])?  # Sign
    %s  # Main number
    (?:\(%s\))?  # Optional uncertainty
    ([eE][+-]?\d+)?  # Optional exponent
    ''' % (POSITIVE_DECIMAL_UNSIGNED, POSITIVE_DECIMAL_UNSIGNED)

NUMBER_WITH_UNCERT_RE = re.compile(
    "^%s$" % NUMBER_WITH_UNCERT_RE_STR, re.VERBOSE)

def parse_error_in_parentheses(representation):
    """
    Returns (value, error) from a string representing a number with
    uncertainty like 12.34(5), 12.34(142), 12.5(3.4) or 12.3(4.2)e3.
    If no parenthesis is given, an uncertainty of one on the last
    digit is assumed.

    Raises ValueError if the string cannot be parsed.
    """

    match = NUMBER_WITH_UNCERT_RE.search(representation)

    if match:
        # The 'main' part is the nominal value, with 'int'eger part, and
        # 'dec'imal part.  The 'uncert'ainty is similarly broken into its
        # integer and decimal parts.
        (sign, main_int, main_dec, uncert_int, uncert_dec,
         exponent) = match.groups()
    else:
        raise ValueError("Unparsable number representation: '%s'."
                         " Was expecting a string of the form 1.23(4)"
                         " or 1.234" % representation)

    # The value of the number is its nominal value:
    value = float(''.join((sign or '',
                           main_int,
                           main_dec or '.0',
                           exponent or '')))

    if uncert_int is None:
        # No uncertainty was found: an uncertainty of 1 on the last
        # digit is assumed:
        uncert_int = '1'

    # Do we have a fully explicit uncertainty?
    if uncert_dec is not None:
        uncert = float("%s%s" % (uncert_int, uncert_dec or ''))
    else:
        # uncert_int represents an uncertainty on the last digits:

        # The number of digits after the period defines the power of
        # 10 than must be applied to the provided uncertainty:
        num_digits_after_period = (0 if main_dec is None
                                   else len(main_dec)-1)
        uncert = int(uncert_int)/10**num_digits_after_period

    # We apply the exponent to the uncertainty as well:
    uncert *= float("1%s" % (exponent or ''))

    return (value, uncert)


# The following function is not exposed because it can in effect be
# obtained by doing x = ufloat(representation) and
# x.nominal_value and x.std_dev():
def str_to_number_with_uncert(representation):
    """
    Given a string that represents a number with uncertainty, returns the
    nominal value and the uncertainty.

    The string can be of the form:
    - 124.5+/-0.15
    - 124.50(15)
    - 124.50(123)
    - 124.5

    When no numerical error is given, an uncertainty of 1 on the last
    digit is implied.

    Raises ValueError if the string cannot be parsed.
    """

    try:
        # Simple form 1234.45+/-1.2:
        (value, uncert) = representation.split('+/-')
    except ValueError:
        # Form with parentheses or no uncertainty:
        parsed_value = parse_error_in_parentheses(representation)
    else:
        try:
            parsed_value = (float(value), float(uncert))
        except ValueError:
            raise ValueError('Cannot parse %s: was expecting a number'
                             ' like 1.23+/-0.1' % representation)

    return parsed_value

def ufloat(representation, tag=None):
    """
    Returns a random variable (Variable object).

    Converts the representation of a number into a number with
    uncertainty (a random variable, defined by a nominal value and
    a standard deviation).

    The representation can be a (value, standard deviation) sequence,
    or a string.

    Strings of the form '12.345+/-0.015', '12.345(15)', or '12.3' are
    recognized (see full list below).  In the last case, an
    uncertainty of +/-1 is assigned to the last digit.

    'tag' is an optional string tag for the variable.  Variables
    don't have to have distinct tags.  Tags are useful for tracing
    what values (and errors) enter in a given result (through the
    error_components() method).

    Examples of valid string representations:

        -1.23(3.4)
        -1.34(5)
        1(6)
        3(4.2)
        -9(2)
        1234567(1.2)
        12.345(15)
        -12.3456(78)e-6
        12.3(0.4)e-5
        0.29
        31.
        -31.
        31
        -3.1e10
        169.0(7)
        169.1(15)
    """

    # This function is somewhat optimized so as to help with the
    # creation of lots of Variable objects (through unumpy.uarray, for
    # instance).

    # representations is "normalized" so as to be a valid sequence of
    # 2 arguments for Variable().

    #! Accepting strings and any kind of sequence slows down the code
    # by about 5 %.  On the other hand, massive initializations of
    # numbers with uncertainties are likely to be performed with
    # unumpy.uarray, which does not support parsing from strings and
    # thus does not have any overhead.

    #! Different, in Python 3:
    if isinstance(representation, basestring):
        representation = str_to_number_with_uncert(representation)

    #! The tag is forced to be a string, so that the user does not
    # create a Variable(2.5, 0.5) in order to represent 2.5 +/- 0.5.
    # Forcing 'tag' to be a string prevents numerical uncertainties
    # from being considered as tags, here:
    if tag is not None:
        #! 'unicode' is removed in Python3:
        assert isinstance(tag, (str, unicode)), "The tag can only be a string."

    #! The special ** syntax is for Python 2.5 and before (Python 2.6+
    # understands tag=tag):
    return Variable(*representation, **{'tag': tag})

