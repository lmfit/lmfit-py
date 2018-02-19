.. _intro_chapter:

===========================================================
Getting started with Non-Linear Least-Squares Fitting
===========================================================

The lmfit package provides simple tools to help you build complex fitting
models for non-linear least-squares problems and apply these models to real
data.  This section gives an overview of the concepts and describes how to
set up and perform simple fits.  Some basic knowledge of Python, NumPy, and
modeling data are assumed -- this is not a tutorial on why or how to
perform a minimization or fit data, but is rather aimed at explaining how
to use lmfit to do these things.

In order to do a non-linear least-squares fit of a model to data or for any
other optimization problem, the main task is to write an *objective
function* that takes the values of the fitting variables and calculates
either a scalar value to be minimized or an array of values that are to be
minimized, typically in the least-squares sense.  For many data fitting
processes, the latter approach is used, and the objective function should
return an array of (data-model), perhaps scaled by some weighting factor
such as the inverse of the uncertainty in the data.  For such a problem,
the chi-square (:math:`\chi^2`) statistic is often defined as:

.. math::

 \chi^2 =  \sum_i^{N} \frac{[y^{\rm meas}_i - y_i^{\rm model}({\bf{v}})]^2}{\epsilon_i^2}

where :math:`y_i^{\rm meas}` is the set of measured data, :math:`y_i^{\rm
model}({\bf{v}})` is the model calculation, :math:`{\bf{v}}` is the set of
variables in the model to be optimized in the fit, and :math:`\epsilon_i`
is the estimated uncertainty in the data.

In a traditional non-linear fit, one writes an objective function that
takes the variable values and calculates the residual array :math:`y^{\rm
meas}_i - y_i^{\rm model}({\bf{v}})`, or the residual array scaled by the
data uncertainties, :math:`[y^{\rm meas}_i - y_i^{\rm
model}({\bf{v}})]/{\epsilon_i}`, or some other weighting factor.

As a simple concrete example, one might want to model data with a decaying
sine wave, and so write an objective function like this::

    from numpy import exp, sin


    def residual(vars, x, data, eps_data):
        amp = vars[0]
        phaseshift = vars[1]
        freq = vars[2]
        decay = vars[3]

        model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

        return (data-model) / eps_data

To perform the minimization with :mod:`scipy.optimize`, one would do this::

    from scipy.optimize import leastsq

    vars = [10.0, 0.2, 3.0, 0.007]
    out = leastsq(residual, vars, args=(x, data, eps_data))

Though it is wonderful to be able to use Python for such optimization
problems, and the SciPy library is robust and easy to use, the approach
here is not terribly different from how one would do the same fit in C or
Fortran.  There are several practical challenges to using this approach,
including:

  a) The user has to keep track of the order of the variables, and their
     meaning -- vars[0] is the amplitude, vars[2] is the frequency, and so
     on, although there is no intrinsic meaning to this order.

  b) If the user wants to fix a particular variable (*not* vary it in the
     fit), the residual function has to be altered to have fewer variables,
     and have the corresponding constant value passed in some other way.
     While reasonable for simple cases, this quickly becomes a significant
     work for more complex models, and greatly complicates modeling for
     people not intimately familiar with the details of the fitting code.

  c) There is no simple, robust way to put bounds on values for the
     variables, or enforce mathematical relationships between the
     variables.  In fact, the optimization methods that do provide
     bounds, require bounds to be set for all variables with separate
     arrays that are in the same arbitrary order as variable values.
     Again, this is acceptable for small or one-off cases, but becomes
     painful if the fitting model needs to change.

These shortcomings are due to the use of traditional arrays to hold the
variables, which matches closely the implementation of the underlying
Fortran code, but does not fit very well with Python's rich selection of
objects and data structures.  The key concept in lmfit is to define and use
:class:`Parameter` objects instead of plain floating point numbers as the
variables for the fit.  Using :class:`Parameter` objects (or the closely
related :class:`Parameters` -- a dictionary of :class:`Parameter` objects),
allows one to:

   a) forget about the order of variables and refer to Parameters
      by meaningful names.
   b) place bounds on Parameters as attributes, without worrying about
      preserving the order of arrays for variables and boundaries.
   c) fix Parameters, without having to rewrite the objective function.
   d) place algebraic constraints on Parameters.

To illustrate the value of this approach, we can rewrite the above example
for the decaying sine wave as::

    from numpy import exp, sin

    from lmfit import minimize, Parameters


    def residual(params, x, data, eps_data):
        amp = params['amp']
        phaseshift = params['phase']
        freq = params['frequency']
        decay = params['decay']

        model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

        return (data-model) / eps_data


    params = Parameters()
    params.add('amp', value=10)
    params.add('decay', value=0.007)
    params.add('phase', value=0.2)
    params.add('frequency', value=3.0)

    out = minimize(residual, params, args=(x, data, eps_data))


At first look, we simply replaced a list of values with a dictionary,
accessed by name -- not a huge improvement.  But each of the named
:class:`Parameter` in the :class:`Parameters` object holds additional
attributes to modify the value during the fit.  For example, Parameters can
be fixed or bounded.  This can be done during definition::

    params = Parameters()
    params.add('amp', value=10, vary=False)
    params.add('decay', value=0.007, min=0.0)
    params.add('phase', value=0.2)
    params.add('frequency', value=3.0, max=10)

where ``vary=False`` will prevent the value from changing in the fit, and
``min=0.0`` will set a lower bound on that parameter's value. It can also be done
later by setting the corresponding attributes after they have been
created::

    params['amp'].vary = False
    params['decay'].min = 0.10

Importantly, our objective function remains unchanged. This means the
objective function can simply express the parameterized phenomenon to be
modeled, and is separate from the choice of parameters to be varied in the
fit.


The `params` object can be copied and modified to make many user-level
changes to the model and fitting process.  Of course, most of the
information about how your data is modeled goes into the objective
function, but the approach here allows some external control; that is,
control by the **user** performing the fit, instead of by the author of the
objective function.

Finally, in addition to the :class:`Parameters` approach to fitting data,
lmfit allows switching optimization methods without changing
the objective function, provides tools for generating fitting reports, and
provides a better determination of Parameters confidence levels.
