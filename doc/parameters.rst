.. _parameters-label:

================================================
:class:`Parameter`  and :class:`Parameters`
================================================

This chapter describes :class:`Parameter` objects, which are
fundamental to the lmfit approach to optimization.   Most real use cases
will use the :class:`Parameters` class, which provides an (ordered)
dictionary of :class:`Parameter` objects.

The :class:`Parameter` class
========================================

.. class:: Parameter(name=None[, value=None[, vary=True[, min=None[, max=None[, expr=None]]]]])

   create a Parameter object.

   :param name: parameter name
   :type name: ``None`` or string -- will be overwritten during fit if ``None``.
   :param value: the numerical value for the parameter
   :param vary:  whether to vary the parameter or not.
   :type vary:  boolean (``True``/``False``)
   :param min:  lower bound for value (``None`` = no lower bound).
   :param max:  upper bound for value (``None`` = no upper bound).
   :param expr:  mathematical expression to use to evaluate value during fit.
   :type expr: ``None`` or string


Each of these inputs is turned into an attribute of the same name.

After a fit, a Parameter for a fitted variable (ie with vary = ``True``)
will have the :attr:`value` attribute holding the best-fit value, and may
(depending on the success of the fit) have obtain additional attributes.

.. attribute:: stderr

   the estimated standard error for the best-fit value.

.. attribute:: correl

   a dictionary of the correlation with the other fitted variables in the
   fit, of the form::

   {'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}

For details of the use of the bounds :attr:`min` and :attr:`max`,
see :ref:`parameter-bounds-label`.

The :attr:`expr` attribute can contain a mathematical expression that will
be used to compute the value for the Parameter at each step in the fit.
See :ref:`math-constraints-label` for more details and examples of this
feature.


The :class:`Parameters` class
========================================

.. class:: Parameters()

   create a Parameters object.  This is little more than a fancy
   dictionary, with the restrictions that

   1. keys must be valid Python symbol names (so that they can be used in
      expressions of mathematical constraints).  This means the names must
      match ``[a-z_][a-z0-9_]*``  and cannot be a Python reserved word.

   2. values must be valid :class:`Parameter` objects.


   Two methods for provided for convenience of initializing Parameters.

.. method:: add(name[, value=None[, vary=True[, min=None[, max=None[, expr=None]]]]])

   add a named parameter.  This simply creates a :class:`Parameter`
   object associated with the key `name`, with optional arguments
   passed to :class:`Parameter`::

     p = Parameters()
     p.add('myvar', value=1, vary=True)

.. method:: add_many(self, paramlist)

   add a list of named parameters.  Each entry must be a tuple
   with the following entries::

        name, value, vary, min, max, expr

   That is, this method is somewhat rigid and verbose (no default values),
   but can be useful when initially defining a parameter list so that it
   looks table-like::

     p = Parameters()
     #           (Name,  Value,  Vary,   Min,  Max,  Expr)
     p.add_many(('amp1',    10,  True, None, None,  None),
                ('cen1',   1.2,  True,  0.5,  2.0,  None),
                ('wid1',   0.8,  True,  0.1, None,  None),
                ('amp2',   7.5,  True, None, None,  None),
                ('cen2',   1.9,  True,  1.0,  3.0,  None),
                ('wid2',  None, False, None, None, '2*wid1/3'))


Simple Example
==================

Putting it all together, a simple example of using a dictionary of
:class:`Parameter` objects and :func:`minimize` might look like this:

.. literalinclude:: ../examples/simple.py


