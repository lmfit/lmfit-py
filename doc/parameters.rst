.. _parameters_chapter:

================================================
:class:`Parameter`  and :class:`Parameters`
================================================

This chapter describes :class:`Parameter` objects, which are
fundamental to the lmfit approach to optimization.   Most real use cases
will use the :class:`Parameters` class, which provides an (ordered)
dictionary of :class:`Parameter` objects.


The :class:`Parameter` class
========================================

.. module:: Parameter

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
see :ref:`bounds_chapter`.

The :attr:`expr` attribute can contain a mathematical expression that will
be used to compute the value for the Parameter at each step in the fit.
See :ref:`constraints_chapter` for more details and examples of this
feature.

.. method:: set(value=None[, vary=None[, min=None[, max=None[, expr=None]]]])

   set or update a Parameters value or other attributes.

   :param name:  parameter name
   :param value: the numerical value for the parameter
   :param vary:  whether to vary the parameter or not.
   :param min:   lower bound for value
   :param max:   upper bound for value
   :param expr:  mathematical expression to use to evaluate value during fit.

   Each parameter has a default value of ``None``, and will be set only if
   the provided value is not ``None``.   You can use this to update some
   Parameter attribute without affecting others, for example::

       p1 = Parameter('a', value=2.0)
       p2 = Parameter('b', value=0.0)
       p1.set(min=0)
       p2.set(vary=False)

   to set a lower bound, or to set a Parameter as have a fixed value.

   Note that to use this approach to lift a lower or upper bound, doing::

       p1.set(min=0)
       .....
       # now lift the lower bound
       p1.set(min=None)   # won't work!  lower bound NOT changed

   won't work -- this will not change the current lower bound.  Instead
   you'll have to use ``np.inf`` to remove a lower or upper bound::

       # now lift the lower bound
       p1.set(min=-np.inf)   # will work!




.. module:: Parameters

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


.. method:: valuesdict(self)

   return an ordered dictionary of name:value pairs for each Parameter.

   This is distinct from the Parameters itself, as it has values of
   the Parameeter values, not the full Parameter object.

   This can be a very convenient way to get Parameter values in a objective
   function.


Simple Example
==================

Putting it all together, a simple example of using a dictionary of
:class:`Parameter` objects and :func:`minimize` might look like this:

.. literalinclude:: ../examples/doc_basic.py


Here, the objective function explicitly unpacks each Parameter value.  This
can be simplified using the :class:`Parameters` :meth:`valuesdict` method,
which would make the objective function ``fcn2min`` above look like::

    def fcn2min(params, x, data):
        """ model decaying sine wave, subtract data"""
        v = params.valuesdict()

        model = v['amp'] * np.sin(x * v['omega'] + v['shift']) * np.exp(-x*x*v['decay'])
        return model - data

The results are identical, and the difference is a stylisic choice.


