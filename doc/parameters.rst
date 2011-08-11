
===========================================================
Getting started with Non-Linear Least-Squares Fitting
===========================================================

The lmfit package is designed to provide a simple way to build complex
fitting models and apply them to real data.   This chapter describes how to
set up and perform simple fits, but does assume some basic knowledge of
Python, Numpy, and modeling data.

To model data in the least-squares sense, the most important step is
writing a function that takes the values of the fitting variables and
calculates a residual function (data-model) that is to be minimized in the
least-squares sense

.. math::

 \chi^2 =  \sum_i^{N} \frac{[y^{\rm meas}_i - y_i^{\rm model}({\bf{v}})]^2}{\epsilon_i^2}

where :math:`y_i^{\rm meas}` is the set of measured data, :math:`y_i^{\rm
model}({\bf{v}})` is the model calculation, :math:`{\bf{v}}` is the set of
variables in the model to be optimized in the fit, and :math:`\epsilon_i`
is the estimated uncertainty in the data.

In a traditional non-linear fit, one writes a function that takes the
variable values and calculates the residual :math:`y^{\rm meas}_i -
y_i^{\rm model}({\bf{v}})`, perhaps something like::

    def residual(vars, x, data):
        amp = vars[0]
        phaseshift = vars[1]
	freq = vars[2]
        decay = vars[3]

	model = amp * sin(x * freq  + phaseshift) * exp(-x*x*decay)

        return (data-model)

To perform the minimization with scipy, one would do::

    from scipy.optimize import leastsq
    vars = [10.0, 0.2, 3.0, 0.007]
    out = leastsq(residual, vars, args=(x, data))

Though in python, and fairly easy to use, this is not terribly different
from how one would do the same fit in C or Fortran.



.. _parameters-label:

Using :class:`Parameters` instead of Variables
=============================================================

As described above, there are several practical challenges in doing
least-squares fit with the traditional implementation (Fortran,
scipy.optimize.leastsq, and most other) in which a list of fitting
variables to the function to be minimized.  These challenges include:

  a) The user has to keep track of the order of the variables, and their
     meaning -- vars[2] is the frequency, and so on.

  b) If the user wants to fix a particular variable (*not* vary it in the fit),
     the residual function has to be altered.  While reasonable for simple
     cases, this quickly becomes significant work for more complex models,
     and greatly complicates modeling for people not intimately familiar
     with the code.

  c) There is no way to put bounds on values for the variables, or enforce
     mathematical relationships between the variables.

LMFIT is designed to void these shortcomings.

The key ideas of LMFIT are to expand a numerical variable with a
:class:`Parameter`, which have more attributes than simply their value.
Instead of a list of numbers, a dcitionary of these Parameters is used in
the function to be minimized.  With this transformation, the above example
would be translated to look like::

    from lmfit import minimize, Parameter

    def residual(params, x, data):
        amp = params['amp'].value
        pshift = params['phase'].value
	freq = params['frequency'].value
        decay = params['decay'].value

	model = amp * sin(x * freq  + pshift) * exp(-x*x*decay)

        return (data-model)

    params = {'amp': Parameter(value=10),
              'decay': Parameter(value=0.007),
              'phase': Parameter(value=0.2),
	      'frequency': Parameter(value=3.0)}

    out = minimize(residual, params, args=(x, data))



So far, this simply looks like it replaced a list of values with a
dictionary.  But Parameters objects can hold additional attributes to
modify the value during the fit.  For example, Parameters can be fixed or
bounded::

    params = {'amp': Parameter(value=10, vary=False),
              'decay': Parameter(value=0.007, min=0.0),
              'phase': Parameter(value=0.2),
	      'frequency': Parameter(value=3.0, max=10.0)}

Now the fit will **not** vary the amplitude parameter, and will also impose
a lower bound on the decay factor and an upper bound on the frequency.
Importantly, the residual function remains unchanged.


The :class:`Parameter` class
========================================

.. class:: Parameter(value=None[, vary=True[, min=None[, max=None[, name=None[, expr=None]]]]])

   create a Parameter object

   :param value: the numerical value for the parameter
   :param vary:  whether to vary the parameter or not.
   :type vary:  boolean (``True``/``False``)
   :param min:  lower bound for value (``None`` = no lower bound).
   :param max:  upper bound for value (``None`` = no upper bound).

   :param name: parameter name
   :type name: ``None`` or string -- will be overwritten during fit if ``None``.
   :param expr:  mathematical expression to use to evaluate value during fit.
   :type expr: ``None`` or string


Each of these inputs is turned into an attribute of the same name.   As
above, one hands a dictionary of Parameters to the fitting routines.   The
name for the Parameter will be set to be consistent

After a fit, a Parameter for a fitted variable (ie with vary = ``True``)
will have the :attr:`value` attribute holding the best-fit value, and may
(depending on the success of the fit) have obtain additional attributes.

.. attribute:: stderr

   the estimated standard error for the best-fit value.

.. attribute:: correl

   a dictionary of the correlation with the other fitted variables in the
   fit, of the form::

   {'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}


The :attr:`expr` attribute can contain a mathematical expression that will
be used to compute the value for the Parameter at each step in the fit.
See :ref:`math-constraints-label` for more details and examples of this
feature.


Simple Example:
==================

Putting it all together, a simple example of using a dictionary of
:class:`Parameter` s  and :func:`minimize` might look like this::

    from lmfit import minimize, Parameter

    def residual(params, x, data=None):
        amp = params['amp'].value
        shift = params['phase_shift'].value
	omega = params['omega'].value
        decay = params['decay'].value

	model = amp * sin(x * omega + shift) * exp(-x*x*decay)

        return (data-model)

    params = {'amp': Parameter(value=10),
              'decay': Parameter(value=0.007, vary=False),
              'phase_shift': Parameter(value=0.2),
	      'omega': Parameter(value=3.0)}

    result = minimize(residual, params, args=(x, data))

    print result.chisqr
    print 'Best-Fit Values:'
    for name, par in params.items():
        print '  %s = %.4f +/- %.4f ' % (name, par.value, par.stderr)

The results of this fit looks like this::

   OUTPUT HERE...




