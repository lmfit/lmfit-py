.. _parameters_chapter:

.. module:: lmfit.parameter

================================================
:class:`Parameter`  and :class:`Parameters`
================================================

This chapter describes :class:`Parameter` objects which is the key concept
of lmfit.

A :class:`Parameter` is the quantity to be optimized in all minimization
problems, replacing the plain floating point number used in the
optimization routines from :mod:`scipy.optimize`.  A :class:`Parameter` has
a value that can be varied in the fit or have a fixed value, have upper
and/or lower bounds.  It can even have a value that is constrained by an
algebraic expression of other Parameter values.  Since :class:`Parameters`
live outside the core optimization routines, they can be used in **all**
optimization routines from :mod:`scipy.optimize`.  By using
:class:`Parameter` objects instead of plain variables, the objective
function does not have to be modified to reflect every change of what is
varied in the fit.  This simplifies the writing of models, allowing general
models that describe the phenomenon to be written, and gives the user more
flexibility in using and testing variations of that model.

Whereas a :class:`Parameter` expands on an individual floating point
variable, the optimization methods need an ordered group of floating point
variables.  In the :mod:`scipy.optimize` routines this is required to be a
1-dimensional numpy ndarray.  For lmfit, where each :class:`Parameter` has
a name, this is replaced by a :class:`Parameters` class, which works as an
ordered dictionary of :class:`Parameter` objects, with a few additional
features and methods.  That is, while the concept of a :class:`Parameter`
is central to lmfit, one normally creates and interacts with a
:class:`Parameters` instance that contains many :class:`Parameter` objects.
The objective functions you write for lmfit will take an instance of
:class:`Parameters` as its first argument.


The :class:`Parameter` class
========================================

.. class:: Parameter(name=None[, value=None[, vary=True[, min=None[, max=None[, expr=None]]]]])

    create a Parameter object.

    :param name: parameter name
    :type name: ``None`` or string -- will be overwritten during fit if ``None``.
    :param value: the numerical value for the parameter
    :param vary:  whether to vary the parameter or not.
    :type vary:  boolean (``True``/``False``) [default ``True``]
    :param min:  lower bound for value (``None`` = no lower bound).
    :param max:  upper bound for value (``None`` = no upper bound).
    :param expr:  mathematical expression to use to evaluate value during fit.
    :type expr: ``None`` or string

    Each of these inputs is turned into an attribute of the same name.

    After a fit, a Parameter for a fitted variable (that is with ``vary =
    True``) may have its :attr:`value` attribute to hold the best-fit value.
    Depending on the success of the fit and fitting algorithm used, it may also
    have attributes :attr:`stderr` and :attr:`correl`.

    .. attribute:: stderr

       the estimated standard error for the best-fit value.

    .. attribute:: correl

       a dictionary of the correlation with the other fitted variables in the
       fit, of the form::

       {'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}

    See :ref:`bounds_chapter` for details on the math used to implement the
    bounds with :attr:`min` and :attr:`max`.

    The :attr:`expr` attribute can contain a mathematical expression that will
    be used to compute the value for the Parameter at each step in the fit.
    See :ref:`constraints_chapter` for more details and examples of this
    feature.

    .. index:: Removing a Constraint Expression

    .. method:: set(value=None[, vary=None[, min=None[, max=None[, expr=None]]]])

       set or update a Parameters value or other attributes.

       :param name:  parameter name
       :param value: the numerical value for the parameter
       :param vary:  whether to vary the parameter or not.
       :param min:   lower bound for value
       :param max:   upper bound for value
       :param expr:  mathematical expression to use to evaluate value during fit.

    Each argument of :meth:`set` has a default value of ``None``, and will
    be set only if the provided value is not ``None``.  You can use this to
    update some Parameter attribute without affecting others, for example::

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

    Similarly, to clear an expression of a parameter, you need to pass an
    empty string, not ``None``.  You also need to give a value and
    explicitly tell it to vary::

	p3 = Parameter('c', expr='(a+b)/2')
	p3.set(expr=None)     # won't work!  expression NOT changed

	# remove constraint expression
	p3.set(value=1.0, vary=True, expr='')  # will work!  parameter now unconstrained


The :class:`Parameters` class
========================================

.. class:: Parameters()

   create a Parameters object.  This is little more than a fancy ordered
   dictionary, with the restrictions that:

   1. keys must be valid Python symbol names, so that they can be used in
      expressions of mathematical constraints.  This means the names must
      match ``[a-z_][a-z0-9_]*``  and cannot be a Python reserved word.

   2. values must be valid :class:`Parameter` objects.

   Two methods are provided for convenient initialization of a :class:`Parameters`,
   and one for extracting :class:`Parameter` values into a plain dictionary.

    .. method:: add(name[, value=None[, vary=True[, min=None[, max=None[, expr=None]]]]])

       add a named parameter.  This creates a :class:`Parameter`
       object associated with the key `name`, with optional arguments
       passed to :class:`Parameter`::

	 p = Parameters()
	 p.add('myvar', value=1, vary=True)

    .. method:: add_many(self, paramlist)

       add a list of named parameters.  Each entry must be a tuple
       with the following entries::

	    name, value, vary, min, max, expr

       This method is somewhat rigid and verbose (no default values), but can
       be useful when initially defining a parameter list so that it looks
       table-like::

	 p = Parameters()
	 #           (Name,  Value,  Vary,   Min,  Max,  Expr)
	 p.add_many(('amp1',    10,  True, None, None,  None),
		    ('cen1',   1.2,  True,  0.5,  2.0,  None),
		    ('wid1',   0.8,  True,  0.1, None,  None),
		    ('amp2',   7.5,  True, None, None,  None),
		    ('cen2',   1.9,  True,  1.0,  3.0,  None),
		    ('wid2',  None, False, None, None, '2*wid1/3'))


    .. automethod:: Parameters.pretty_print

    .. method:: valuesdict()

       return an ordered dictionary of name:value pairs with the
       Paramater name as the key and Parameter value as value.

       This is distinct from the :class:`Parameters` itself, as the dictionary
       values are not :class:`Parameter` objects, just the :attr:`value`.
       Using :meth:`valuesdict` can be a very convenient way to get updated
       values in a objective function.

    .. method:: dumps(**kws)

       return a JSON string representation of the :class:`Parameter` object.
       This can be saved or used to re-create or re-set parameters, using the
       :meth:`loads` method.

       Optional keywords are sent :py:func:`json.dumps`.

    .. method:: dump(file, **kws)

       write a JSON representation of the :class:`Parameter` object to a file
       or file-like object in `file` -- really any object with a :meth:`write`
       method.  Optional keywords are sent :py:func:`json.dumps`.

    .. method:: loads(sval, **kws)

       use a JSON string representation of the :class:`Parameter` object in
       `sval` to set all parameter settings. Optional keywords are sent
       :py:func:`json.loads`.

    .. method:: load(file, **kws)

       read and use a JSON string representation of the :class:`Parameter`
       object from a file or file-like object in `file` -- really any object
       with a :meth:`read` method.  Optional keywords are sent
       :py:func:`json.loads`.


Simple Example
==================

Using :class:`Parameters`` and :func:`minimize` function (discussed in the
next chapter) might look like this:

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
