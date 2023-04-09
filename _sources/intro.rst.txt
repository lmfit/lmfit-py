.. _intro_chapter:

=====================================================
Getting started with Non-Linear Least-Squares Fitting
=====================================================

The lmfit package provides simple tools to help you build complex fitting
models for non-linear least-squares problems and apply these models to real
data. This section gives an overview of the concepts and describes how to
set up and perform simple fits. Some basic knowledge of Python, NumPy, and
modeling data are assumed -- this is not a tutorial on why or how to
perform a minimization or fit data, but is rather aimed at explaining how
to use lmfit to do these things.

In order to do a non-linear least-squares fit of a model to data or for any
other optimization problem, the main task is to write an *objective
function* that takes the values of the fitting variables and calculates
either a scalar value to be minimized or an array of values that are to be
minimized, typically in the least-squares sense. For many data fitting
processes, the latter approach is used, and the objective function should
return an array of ``(data-model)``, perhaps scaled by some weighting factor
such as the inverse of the uncertainty in the data. For such a problem,
the chi-square (:math:`\chi^2`) statistic is often defined as:

.. math::

 \chi^2 =  \sum_i^{N} \frac{[y^{\rm meas}_i - y_i^{\rm model}({\bf{v}})]^2}{\epsilon_i^2}

where :math:`y_i^{\rm meas}` is the set of measured data, :math:`y_i^{\rm
model}({\bf{v}})` is the model calculation, :math:`{\bf{v}}` is the set of
variables in the model to be optimized in the fit, and :math:`\epsilon_i`
is the estimated uncertainty in the data, respectively.

In a traditional non-linear fit, one writes an objective function that
takes the variable values and calculates the residual array :math:`y^{\rm
meas}_i - y_i^{\rm model}({\bf{v}})`, or the residual array scaled by the
data uncertainties, :math:`[y^{\rm meas}_i - y_i^{\rm
model}({\bf{v}})]/{\epsilon_i}`, or some other weighting factor.

As a simple concrete example, one might want to model data with a decaying
sine wave, and so write an objective function like this:

.. jupyter-execute::

    from numpy import exp, sin

    def residual(variables, x, data, uncertainty):
        """Model a decaying sine wave and subtract data."""
        amp = variables[0]
        phaseshift = variables[1]
        freq = variables[2]
        decay = variables[3]

        model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

        return (data-model) / uncertainty

To perform the minimization with :mod:`scipy.optimize`, one would do this:

.. jupyter-execute::

    from numpy import linspace, random
    from scipy.optimize import leastsq

    # generate synthetic data with noise
    x = linspace(0, 100)
    noise = random.normal(size=x.size, scale=0.2)
    data = 7.5 * sin(x*0.22 + 2.5) * exp(-x*x*0.01) + noise

    # generate experimental uncertainties
    uncertainty = abs(0.16 + random.normal(size=x.size, scale=0.05))

    variables = [10.0, 0.2, 3.0, 0.007]
    out = leastsq(residual, variables, args=(x, data, uncertainty))

Though it is wonderful to be able to use Python for such optimization
problems, and the SciPy library is robust and easy to use, the approach
here is not terribly different from how one would do the same fit in C or
Fortran. There are several practical challenges to using this approach,
including:

  a) The user has to keep track of the order of the variables, and their
     meaning -- ``variables[0]`` is the ``amplitude``, ``variables[2]`` is the
     ``frequency``, and so on, although there is no intrinsic meaning to this
     order.
  b) If the user wants to fix a particular variable (*not* vary it in the fit),
     the residual function has to be altered to have fewer variables, and have
     the corresponding constant value passed in some other way. While
     reasonable for simple cases, this quickly becomes a significant work for
     more complex models, and greatly complicates modeling for people not
     intimately familiar with the details of the fitting code.
  c) There is no simple, robust way to put bounds on values for the variables,
     or enforce mathematical relationships between the variables. While some
     optimization methods in SciPy do provide bounds, they require bounds to
     be set for all variables with separate arrays that are in the same
     arbitrary order as variable values. Again, this is acceptable for small
     or one-off cases, but becomes painful if the fitting model needs to
     change.
  d) In some cases, constraints can be placed on Parameter values, but this is
     a pretty opaque and complex process.

While these shortcomings can be worked around with some work, they are all
essentially due to the use of arrays or lists to hold the variables.
This closely matches the implementation of the underlying Fortran code, but
does not fit very well with Python's rich selection of objects and data
structures. The key concept in lmfit is to define and use :class:`Parameter`
objects instead of plain floating point numbers as the variables for the
fit. Using :class:`Parameter` objects (or the closely related
:class:`Parameters` -- a dictionary of :class:`Parameter` objects), allows one
to do the following:

   a) forget about the order of variables and refer to Parameters
      by meaningful names.
   b) place bounds on Parameters as attributes, without worrying about
      preserving the order of arrays for variables and boundaries, and without
      relying on the solver to support bounds itself.
   c) fix Parameters, without having to rewrite the objective function.
   d) place algebraic constraints on Parameters.

To illustrate the value of this approach, we can rewrite the above example
for the decaying sine wave as:

.. jupyter-execute::

    from numpy import exp, sin

    from lmfit import minimize, Parameters


    def residual(params, x, data, uncertainty):
        amp = params['amp']
        phaseshift = params['phase']
        freq = params['frequency']
        decay = params['decay']

        model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

        return (data-model) / uncertainty


    params = Parameters()
    params.add('amp', value=10)
    params.add('decay', value=0.007)
    params.add('phase', value=0.2)
    params.add('frequency', value=3.0)

    out = minimize(residual, params, args=(x, data, uncertainty))


At first look, we simply replaced a list of values with a dictionary, so that
we can access Parameters by name. Just by itself, this is better as it allows
separation of the objective function from the code using it.

Note that creation of Parameters here could also be done as:

.. versionadded:: 1.2.0

.. jupyter-execute::

    from lmfit import create_params

    params = create_params(amp=10, decay=0.007, phase=0.2, frequency=3.0)


where keyword/value pairs set Parameter names and their initial values.

Either when using :func:`create_param` or :class:`Parameters`, the resulting
``params`` object is an instance of :class:`Parameters`, which acts like a
dictionary, with keys being the Parameter name and values being individual
:class:`Parameter` objects. These :class:`Parameter` objects hold the value
and several other attributes that control how a Parameter acts. For example,
Parameters can be fixed or bounded; setting attributes to control this
behavior can be done during definition, as with:


.. jupyter-execute::

   params = Parameters()
   params.add('amp', value=10, vary=False)
   params.add('decay', value=0.007, min=0.0)
   params.add('phase', value=0.2)
   params.add('frequency', value=3.0, max=10)


Here ``vary=False`` will prevent the value from changing in the fit, and
``min=0.0`` will set a lower bound on that parameter's value. The same thing
can be accomplished by providing a dictionary of attribute values to
:func:`create_params`:

.. versionadded:: 1.2.0

.. jupyter-execute::

   params = create_params(amp={'value': 10, 'vary': False},
                          decay={'value': 0.007, 'min': 0},
                          phase=0.2,
                          frequency={'value': 3.0, 'max':10})

Parameter attributes can also be modified after they have been created:

.. jupyter-execute::

    params['amp'].vary = False
    params['decay'].min = 0.10

Importantly, our objective function remains unchanged. This means the objective
function can simply express the parametrized phenomenon to be calculated,
accessing Parameter values by name and separating the choice of parameters to
be varied in the fit.

The ``params`` object can be copied and modified to make many user-level
changes to the model and fitting process. Of course, most of the information
about how your data is modeled goes into the objective function, but the
approach here allows some external control; that is, control by the **user**
performing the fit, instead of by the author of the objective function.

Finally, in addition to the :class:`Parameters` approach to fitting data, lmfit
allows switching optimization methods without changing the objective function,
provides tools for generating fitting reports, and provides a better
determination of Parameters confidence levels.
