.. _parameters_chapter:

.. module:: lmfit.parameter

================================================
:class:`Parameter`  and :class:`Parameters`
================================================

This chapter describes :class:`Parameter` objects which is a key concept of
lmfit.

A :class:`Parameter` is the quantity to be optimized in all minimization
problems, replacing the plain floating point number used in the
optimization routines from :mod:`scipy.optimize`.  A :class:`Parameter` has
a value that can either be varied in the fit or held at a fixed value, and
can have upper and/or lower bounds placd on the value.  It can even have a
value that is constrained by an algebraic expression of other Parameter
values.  Since :class:`Parameter` objects live outside the core
optimization routines, they can be used in **all** optimization routines
from :mod:`scipy.optimize`.  By using :class:`Parameter` objects instead of
plain variables, the objective function does not have to be modified to
reflect every change of what is varied in the fit, or whether bounds can be
applied.  This simplifies the writing of models, allowing general models
that describe the phenomenon and gives the user more flexibility in using
and testing variations of that model.

Whereas a :class:`Parameter` expands on an individual floating point
variable, the optimization methods still actually need an ordered group of
floating point variables.  In the :mod:`scipy.optimize` routines this is
required to be a 1-dimensional numpy ndarray.  In lmfit, this 1-dimensional
array is replaced by a :class:`Parameters` object, which works as an
ordered dictionary of :class:`Parameter` objects, with a few additional
features and methods.  That is, while the concept of a :class:`Parameter`
is central to lmfit, one normally creates and interacts with a
:class:`Parameters` instance that contains many :class:`Parameter` objects.
For example, the objective functions you write for lmfit will take an
instance of :class:`Parameters` as its first argument.  A table of
parameter values, bounds and other attributes can be printed using
:meth:`Parameters.pretty_print`.


The :class:`Parameter` class
========================================

.. autoclass:: Parameter


    See :ref:`bounds_chapter` for details on the math used to implement the
    bounds with :attr:`min` and :attr:`max`.

    The :attr:`expr` attribute can contain a mathematical expression that will
    be used to compute the value for the Parameter at each step in the fit.
    See :ref:`constraints_chapter` for more details and examples of this
    feature.

    .. index:: Removing a Constraint Expression

    .. automethod:: Parameter.set


The :class:`Parameters` class
========================================

.. class:: Parameters()

   Create a Parameters object.  This is little more than a fancy ordered
   dictionary, with the restrictions that:

   1. keys must be valid Python symbol names, so that they can be used in
      expressions of mathematical constraints.  This means the names must
      match ``[a-z_][a-z0-9_]*``  and cannot be a Python reserved word.

   2. values must be valid :class:`Parameter` objects.

   Two methods are provided for convenient initialization of a :class:`Parameters`,
   and one for extracting :class:`Parameter` values into a plain dictionary.

    .. method:: add(name[, value=None[, vary=True[, min=-np.inf[, max=np.inf[, expr=None[, brute_step=None]]]]]])

       Add a named parameter.  This creates a :class:`Parameter`
       object associated with the key `name`, with optional arguments
       passed to :class:`Parameter`::

         p = Parameters()
         p.add('myvar', value=1, vary=True)

    .. method:: add_many(self, paramlist)

       Add a list of named parameters.  Each entry must be a tuple
       with the following entries::

            name, value, vary, min, max, expr, brute_step

       This method is somewhat rigid and verbose (no default values), but can
       be useful when initially defining a parameter list so that it looks
       table-like::

         p = Parameters()
         #          (Name,  Value,  Vary,   Min,  Max,  Expr, Brute_step)
         p.add_many(('amp1',    10,  True, None, None,  None, None),
                    ('cen1',   1.2,  True,  0.5,  2.0,  None, None),
                    ('wid1',   0.8,  True,  0.1, None,  None, None),
                    ('amp2',   7.5,  True, None, None,  None, None),
                    ('cen2',   1.9,  True,  1.0,  3.0,  None, 0.1),
                    ('wid2',  None, False, None, None, '2*wid1/3', None))


    .. automethod:: Parameters.pretty_print

    .. method:: valuesdict()

       Return an ordered dictionary of name:value pairs with the
       Paramater name as the key and Parameter value as value.

       This is distinct from the :class:`Parameters` itself, as the dictionary
       values are not :class:`Parameter` objects, just the :attr:`value`.
       Using :meth:`valuesdict` can be a very convenient way to get updated
       values in a objective function.

    .. method:: dumps(**kws)

       Return a JSON string representation of the :class:`Parameter` object.
       This can be saved or used to re-create or re-set parameters, using the
       :meth:`loads` method.

       Optional keywords are sent :py:func:`json.dumps`.

    .. method:: dump(file, **kws)

       Write a JSON representation of the :class:`Parameter` object to a file
       or file-like object in `file` -- really any object with a :meth:`write`
       method.  Optional keywords are sent :py:func:`json.dumps`.

    .. method:: loads(sval, **kws)

       Use a JSON string representation of the :class:`Parameter` object in
       `sval` to set all parameter settings. Optional keywords are sent
       :py:func:`json.loads`.

    .. method:: load(file, **kws)

       Read and use a JSON string representation of the :class:`Parameter`
       object from a file or file-like object in `file` -- really any object
       with a :meth:`read` method.  Optional keywords are sent
       :py:func:`json.loads`.


Simple Example
==================

Using :class:`~lmfit.parameter.Parameters` and :func:`~lmfit.minimizer.minimize`
function (discussed in the next chapter) might look like this:

.. literalinclude:: ../examples/doc_basic.py


Here, the objective function explicitly unpacks each Parameter value.  This
can be simplified using the :class:`Parameters` :meth:`valuesdict` method,
which would make the objective function ``fcn2min`` above look like::

    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        v = params.valuesdict()

        model = v['amp'] * np.sin(x * v['omega'] + v['shift']) * np.exp(-x*x*v['decay'])
        return model - data

The results are identical, and the difference is a stylistic choice.
