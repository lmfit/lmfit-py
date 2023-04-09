.. _constraints_chapter:

==============================
Using Mathematical Constraints
==============================

.. _asteval: https://newville.github.io/asteval/

Being able to fix variables to a constant value or place upper and lower
bounds on their values can greatly simplify modeling real data. These
capabilities are key to lmfit's Parameters. In addition, it is sometimes
highly desirable to place mathematical constraints on parameter values.
For example, one might want to require that two Gaussian peaks have the
same width, or have amplitudes that are constrained to add to some value.
Of course, one could rewrite the objective or model function to place such
requirements, but this is somewhat error-prone, and limits the flexibility
so that exploring constraints becomes laborious.

To simplify the setting of constraints, Parameters can be assigned a
mathematical expression of other Parameters, builtin constants, and builtin
mathematical functions that will be used to determine its value. The
expressions used for constraints are evaluated using the `asteval`_ module,
which uses Python syntax, and evaluates the constraint expressions in a safe
and isolated namespace.

This approach to mathematical constraints allows one to not have to write a
separate model function for two Gaussians where the two ``sigma`` values are
forced to be equal, or where amplitudes are related. Instead, one can write a
more general two Gaussian model (perhaps using :class:`GaussianModel`) and
impose such constraints on the Parameters for a particular fit.


Overview
========

Just as one can place bounds on a Parameter, or keep it fixed during the
fit, so too can one place mathematical constraints on parameters. The way
this is done with lmfit is to write a Parameter as a mathematical
expression of the other parameters and a set of pre-defined operators and
functions. The constraint expressions are simple Python statements,
allowing one to place constraints like:

.. jupyter-execute::

    from lmfit import Parameters

    pars = Parameters()
    pars.add('frac_curve1', value=0.5, min=0, max=1)
    pars.add('frac_curve2', expr='1-frac_curve1')

as the value of the ``frac_curve1`` parameter is updated at each step in the
fit, the value of ``frac_curve2`` will be updated so that the two values are
constrained to add to 1.0. Of course, such a constraint could be placed in
the fitting function, but the use of such constraints allows the end-user
to modify the model of a more general-purpose fitting function.

Nearly any valid mathematical expression can be used, and a variety of
built-in functions are available for flexible modeling.

Supported Operators, Functions, and Constants
=============================================

The mathematical expressions used to define constrained Parameters need to
be valid Python expressions. As you would expect, the operators ``+``, ``-``,
``*``, ``/``, and ``**``, are supported. In fact, a much more complete set can
be used, including Python's bit- and logical operators::

    +, -, *, /, **, &, |, ^, <<, >>, %, and, or,
    ==, >, >=, <, <=, !=, ~, not, is, is not, in, not in


The values for ``e`` (2.7182818...) and ``pi`` (3.1415926...) are available,
as are several supported mathematical and trigonometric function::

  abs, acos, acosh, asin, asinh, atan, atan2, atanh, ceil,
  copysign, cos, cosh, degrees, exp, fabs, factorial,
  floor, fmod, frexp, fsum, hypot, isinf, isnan, ldexp,
  log, log10, log1p, max, min, modf, pow, radians, sin,
  sinh, sqrt, tan, tanh, trunc


In addition, all Parameter names will be available in the mathematical
expressions. Thus, with parameters for a few peak-like functions:

.. jupyter-execute::

    pars = Parameters()
    pars.add('amp_1', value=0.5, min=0, max=1)
    pars.add('cen_1', value=2.2)
    pars.add('wid_1', value=0.2)

The following expression are all valid:

.. jupyter-execute::

    pars.add('amp_2', expr='(2.0 - amp_1**2)')
    pars.add('wid_2', expr='sqrt(pi)*wid_1')
    pars.add('cen_2', expr='cen_1 * wid_2 / max(wid_1, 0.001)')

In fact, almost any valid Python expression is allowed. A notable example
is that Python's 1-line *if expression* is supported:

.. jupyter-execute::

    pars.add('param_a', value=1)
    pars.add('param_b', value=2)
    pars.add('test_val', value=100)

    pars.add('bounded', expr='param_a if test_val/2. > 100 else param_b')

which is equivalent to the more familiar:

.. jupyter-execute::

    if pars['test_val'].value/2. > 100:
        bounded = pars['param_a'].value
    else:
        bounded = pars['param_b'].value

Using Inequality Constraints
============================

A rather common question about how to set up constraints
that use an inequality, say, :math:`x + y \le 10`. This
can be done with algebraic constraints by recasting the
problem, as :math:`x + y = \delta` and :math:`\delta \le
10`. That is, first, allow :math:`x` to be held by the
freely varying parameter ``x``. Next, define a parameter
``delta`` to be variable with a maximum value of 10, and
define parameter ``y`` as ``delta - x``:

.. jupyter-execute::

    pars = Parameters()
    pars.add('x', value=5, vary=True)
    pars.add('delta', value=5, max=10, vary=True)
    pars.add('y', expr='delta-x')

The essential point is that an inequality still implies
that a variable (here, ``delta``) is needed to describe the
constraint. The secondary point is that upper and lower
bounds can be used as part of the inequality to make the
definitions more convenient.


Advanced usage of Expressions in lmfit
======================================

The expression used in a constraint is converted to a
Python `Abstract Syntax Tree
<https://docs.python.org/library/ast.html>`_, which is an
intermediate version of the expression -- a syntax-checked,
partially compiled expression. Among other things, this
means that Python's own parser is used to parse and convert
the expression into something that can easily be evaluated
within Python. It also means that the symbols in the
expressions can point to any Python object.

In fact, the use of Python's AST allows a nearly full version of Python to
be supported, without using Python's built-in :meth:`eval` function. The
`asteval`_ module actually supports most Python syntax, including for- and
while-loops, conditional expressions, and user-defined functions. There
are several unsupported Python constructs, most notably the class
statement, so that new classes cannot be created, and the import statement,
which helps make the `asteval`_ module safe from malicious use.

One important feature of the `asteval`_ module is that you can add
domain-specific functions into the it, for later use in constraint
expressions. To do this, you would use the :attr:`_asteval` attribute of
the :class:`Parameters` class, which contains a complete AST interpreter.
The `asteval`_ interpreter uses a flat namespace, implemented as a single
dictionary. That means you can preload any Python symbol into the namespace
for the constraints, for example this Lorentzian function:

.. jupyter-execute::

    def mylorentzian(x, amp, cen, wid):
        "lorentzian function: wid = half-width at half-max"
        return (amp / (1 + ((x-cen) / wid)**2))


You can add this user-defined function to the `asteval`_ interpreter of
the :class:`Parameters` class:

.. jupyter-execute::

    from lmfit import Parameters

    pars = Parameters()
    pars._asteval.symtable['lorentzian'] = mylorentzian

and then initialize the :class:`Minimizer` class with this parameter set:

.. jupyter-execute::

    from lmfit import Minimizer


    def userfcn(x, params):
        pass


    fitter = Minimizer(userfcn, pars)

Alternatively, one can first initialize the :class:`Minimizer` class and
add the function to the `asteval`_ interpreter of :attr:`Minimizer.params`
afterwards:

.. jupyter-execute::

    pars = Parameters()
    fitter = Minimizer(userfcn, pars)
    fitter.params._asteval.symtable['lorentzian'] = mylorentzian

In both cases the user-defined :meth:`lorentzian` function can now be
used in constraint expressions.
