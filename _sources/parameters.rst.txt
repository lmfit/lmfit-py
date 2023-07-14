.. _parameters_chapter:

.. module:: lmfit.parameter

==========================================
:class:`Parameter` and :class:`Parameters`
==========================================

This chapter describes the :class:`Parameter` object, which is a key concept of
lmfit.

A :class:`Parameter` is the quantity to be optimized in all minimization
problems, replacing the plain floating point number used in the
optimization routines from :mod:`scipy.optimize`. A :class:`Parameter` has
a value that can either be varied in the fit or held at a fixed value, and
can have lower and/or upper bounds placed on the value. It can even have a
value that is constrained by an algebraic expression of other Parameter
values. Since :class:`Parameter` objects live outside the core
optimization routines, they can be used in **all** optimization routines
from :mod:`scipy.optimize`. By using :class:`Parameter` objects instead of
plain variables, the objective function does not have to be modified to
reflect every change of what is varied in the fit, or whether bounds can be
applied. This simplifies the writing of models, allowing general models
that describe the phenomenon and gives the user more flexibility in using
and testing variations of that model.

Whereas a :class:`Parameter` expands on an individual floating point
variable, the optimization methods actually still need an ordered group of
floating point variables. In the :mod:`scipy.optimize` routines this is
required to be a one-dimensional :numpydoc:`ndarray`. In lmfit, this one-dimensional
array is replaced by a :class:`Parameters` object, which works as an
ordered dictionary of :class:`Parameter` objects with a few additional
features and methods. That is, while the concept of a :class:`Parameter`
is central to lmfit, one normally creates and interacts with a
:class:`Parameters` instance that contains many :class:`Parameter` objects.
For example, the objective functions you write for lmfit will take an
instance of :class:`Parameters` as its first argument. A table of
parameter values, bounds, and other attributes can be printed using
:meth:`Parameters.pretty_print`.


The :class:`Parameter` class
============================

.. autoclass:: Parameter

    See :ref:`bounds_chapter` for details on the math used to implement the
    bounds with :attr:`min` and :attr:`max`.

    The :attr:`expr` attribute can contain a mathematical expression that will
    be used to compute the value for the Parameter at each step in the fit.
    See :ref:`constraints_chapter` for more details and examples of this
    feature.

    .. index:: Removing a Constraint Expression

    .. automethod:: set


The :class:`Parameters` class
=============================

.. autoclass:: Parameters

    .. automethod:: add

    .. automethod:: add_many

    .. automethod:: pretty_print

    .. automethod:: valuesdict

    .. automethod:: create_uvars

    .. automethod:: dumps

    .. automethod:: dump

    .. automethod:: eval

    .. automethod:: loads

    .. automethod:: load


.. _dumpload_warning:

.. warning::

   Saving Parameters with user-added functions to the ``_asteval``
   interpreter using :meth::`dump` and :meth:`dumps` may not be easily
   recovered with the :meth:`load` and :meth:`loads`. See
   :ref:`model_saveload_sec` for further discussion.


The :func:`create_params` function
==================================

.. versionadded:: 1.2.0

The :func:`create_params` function is probably the easiest method for making
:class:`Parameters` objects, as it allows defining Parameter names by keyword
with values either being the numerical initial value for the Parameter or being
a dictionary with keyword/value pairs for ``value`` as well as other Parameter
attribute such as ``min``, ``max``, ``expr``, and so forth.

.. autofunction:: create_params


Simple Example
==============

A basic example making use of :class:`~lmfit.parameter.Parameters` and the
:func:`~lmfit.minimizer.minimize` function (discussed in the next chapter)
might look like this:

.. jupyter-execute:: ../examples/doc_parameters_basic.py
    :hide-output:


Here, the objective function explicitly unpacks each Parameter value. This
can be simplified using the :class:`Parameters` :meth:`valuesdict` method,
which would make the objective function ``fcn2min`` above look like:

.. jupyter-execute::

    def fcn2min(params, x, data):
        """Model a decaying sine wave and subtract data."""
        v = params.valuesdict()

        model = v['amp'] * np.sin(x*v['omega'] + v['shift']) * np.exp(-x*x*v['decay'])
        return model - data

The results are identical, and the difference is a stylistic choice.
