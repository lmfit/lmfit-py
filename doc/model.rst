.. _model_chapter:

===============================
Modeling Data and Curve Fitting
===============================

.. module:: lmfit.model

A common use of least-squares minimization is *curve fitting*, where one
has a parametrized model function meant to explain some phenomena and wants
to adjust the numerical values for the model so that it most closely
matches some data. With :mod:`scipy`, such problems are typically solved
with :scipydoc:`optimize.curve_fit`, which is a wrapper around
:scipydoc:`optimize.leastsq`. Since lmfit's
:func:`~lmfit.minimizer.minimize` is also a high-level wrapper around
:scipydoc:`optimize.leastsq` it can be used for curve-fitting problems.
While it offers many benefits over :scipydoc:`optimize.leastsq`, using
:func:`~lmfit.minimizer.minimize` for many curve-fitting problems still
requires more effort than using :scipydoc:`optimize.curve_fit`.

The :class:`Model` class in lmfit provides a simple and flexible approach
to curve-fitting problems. Like :scipydoc:`optimize.curve_fit`, a
:class:`Model` uses a *model function* -- a function that is meant to
calculate a model for some phenomenon -- and then uses that to best match
an array of supplied data. Beyond that similarity, its interface is rather
different from :scipydoc:`optimize.curve_fit`, for example in that it uses
:class:`~lmfit.parameter.Parameters`, but also offers several other
important advantages.

In addition to allowing you to turn any model function into a curve-fitting
method, lmfit also provides canonical definitions for many known lineshapes
such as Gaussian or Lorentzian peaks and Exponential decays that are widely
used in many scientific domains. These are available in the :mod:`models`
module that will be discussed in more detail in the next chapter
(:ref:`builtin_models_chapter`). We mention it here as you may want to
consult that list before writing your own model. For now, we focus on
turning Python functions into high-level fitting models with the
:class:`Model` class, and using these to fit data.


Motivation and simple example: Fit data to Gaussian profile
===========================================================

Let's start with a simple and common example of fitting data to a Gaussian
peak. As we will see, there is a built-in :class:`GaussianModel` class that
can help do this, but here we'll build our own. We start with a simple
definition of the model function:

.. jupyter-execute::
    :hide-code:

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 150
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'

.. jupyter-execute::

    from numpy import exp, linspace, random


    def gaussian(x, amp, cen, wid):
        return amp * exp(-(x-cen)**2 / wid)

We want to use this function to fit to data :math:`y(x)` represented by the
arrays ``y`` and ``x``.  With :scipydoc:`optimize.curve_fit`, this would be:

.. jupyter-execute::
    :hide-output:

    from scipy.optimize import curve_fit

    x = linspace(-10, 10, 101)
    y = gaussian(x, 2.33, 0.21, 1.51) + random.normal(0, 0.2, x.size)

    init_vals = [1, 0, 1]  # for [amp, cen, wid]
    best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)

That is, we create data, make an initial guess of the model values, and run
:scipydoc:`optimize.curve_fit` with the model function, data arrays, and
initial guesses. The results returned are the optimal values for the
parameters and the covariance matrix. It's simple and useful, but it
misses the benefits of lmfit.

With lmfit, we create a :class:`Model` that wraps the ``gaussian`` model
function, which automatically generates the appropriate residual function,
and determines the corresponding parameter names from the function
signature itself:

.. jupyter-execute::

    from lmfit import Model

    gmodel = Model(gaussian)
    print(f'parameter names: {gmodel.param_names}')
    print(f'independent variables: {gmodel.independent_vars}')

As you can see, the Model ``gmodel`` determined the names of the parameters
and the independent variables. By default, the first argument of the
function is taken as the independent variable, held in
:attr:`independent_vars`, and the rest of the functions positional
arguments (and, in certain cases, keyword arguments -- see below) are used
for Parameter names. Thus, for the ``gaussian`` function above, the
independent variable is ``x``, and the parameters are named ``amp``,
``cen``, and ``wid``, and -- all taken directly from the signature of the
model function. As we will see below, you can modify the default
assignment of independent variable / arguments and specify yourself what
the independent variable is and which function arguments should be identified
as parameter names.

:class:`~lmfit.parameter.Parameters` are *not* created when the model is
created. The model knows what the parameters should be named, but nothing about
the scale and range of your data. To help you create Parameters for a Model,
each model has a :meth:`make_params` method that will generate parameters with
the expected names. You will have to do this, or make Parameters some other way
(say, with :func:`~lmfit.parameter.create_params`), and assign initial values
for all Parameters. You can also assign other attributes when doing this:

.. jupyter-execute::

    params = gmodel.make_params()

This creates the :class:`~lmfit.parameter.Parameters` but does not
automatically give them initial values since it has no idea what the scale
should be. If left unspecified, the initial values will be ``-Inf``, which will
generally fail to give useful results. You can set initial values for
parameters with keyword arguments to :meth:`make_params`:

.. jupyter-execute::

    params = gmodel.make_params(cen=0.3, amp=3, wid=1.25)

or assign them (and other parameter properties) after the
:class:`~lmfit.parameter.Parameters` class has been created.

A :class:`Model` has several methods associated with it. For example, one
can use the :meth:`eval` method to evaluate the model or the :meth:`fit`
method to fit data to this model with a :class:`Parameter` object. Both of
these methods can take explicit keyword arguments for the parameter values.
For example, one could use :meth:`eval` to calculate the predicted
function:

.. jupyter-execute::

    x_eval = linspace(0, 10, 201)
    y_eval = gmodel.eval(params, x=x_eval)

or with:

.. jupyter-execute::

    y_eval = gmodel.eval(x=x_eval, cen=6.5, amp=100, wid=2.0)

Admittedly, this a slightly long-winded way to calculate a Gaussian
function, given that you could have called your ``gaussian`` function
directly. But now that the model is set up, we can use its :meth:`fit`
method to fit this model to data, as with:

.. jupyter-execute::

    result = gmodel.fit(y, params, x=x)

or with:

.. jupyter-execute::

    result = gmodel.fit(y, x=x, cen=0.5, amp=10, wid=2.0)

Putting everything together, included in the ``examples`` folder with the
source code, is:

.. jupyter-execute:: ../examples/doc_model_gaussian.py
    :hide-output:

which is pretty compact and to the point. The returned ``result`` will be
a :class:`ModelResult` object. As we will see below, this has many
components, including a :meth:`fit_report` method, which will show:

.. jupyter-execute::
    :hide-code:

    print(result.fit_report())

As the script shows, the result will also have :attr:`init_fit` for the fit
with the initial parameter values and a :attr:`best_fit` for the fit with
the best fit parameter values. These can be used to generate the following
plot:

.. jupyter-execute::
    :hide-code:

    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.legend()
    plt.show()

which shows the data in blue dots, the best fit as a solid green line, and
the initial fit as a dashed orange line.

Note that the model fitting was really performed with:

.. jupyter-execute::

    gmodel = Model(gaussian)
    result = gmodel.fit(y, params, x=x, amp=5, cen=5, wid=1)

These lines clearly express that we want to turn the ``gaussian`` function
into a fitting model, and then fit the :math:`y(x)` data to this model,
starting with values of 5 for ``amp``, 5 for ``cen`` and 1 for ``wid``. In
addition, all the other features of lmfit are included:
:class:`~lmfit.parameter.Parameters` can have bounds and constraints and
the result is a rich object that can be reused to explore the model fit in
detail.


The :class:`Model` class
========================

The :class:`Model` class provides a general way to wrap a pre-defined
function as a fitting model.

.. autoclass::  Model


:class:`Model` class Methods
----------------------------

.. automethod:: Model.eval

.. automethod:: Model.fit

.. automethod:: Model.guess

.. automethod:: Model.make_params

.. automethod:: Model.set_param_hint

   See :ref:`model_param_hints_section`.

.. automethod:: Model.print_param_hints

   See :ref:`model_param_hints_section`.

..  automethod:: Model.post_fit

   See :ref:`modelresult_uvars_postfit_section`.


:class:`Model` class Attributes
-------------------------------

.. attribute:: func

   The model function used to calculate the model.

.. attribute:: independent_vars

   List of strings for names of the independent variables.

.. attribute:: nan_policy

   Describes what to do for NaNs that indicate missing values in the data.
   The choices are:

    * ``'raise'``: Raise a ``ValueError`` (default)
    * ``'propagate'``: Do not check for NaNs or missing values. The fit will
      try to ignore them.
    * ``'omit'``: Remove NaNs or missing observations in data. If pandas is
      installed, :func:`pandas.isnull` is used, otherwise
      :func:`numpy.isnan` is used.

.. attribute:: name

   Name of the model, used only in the string representation of the
   model. By default this will be taken from the model function.

.. attribute:: opts

   Extra keyword arguments to pass to model function. Normally this will
   be determined internally and should not be changed.

.. attribute:: param_hints

   Dictionary of parameter hints. See :ref:`model_param_hints_section`.

.. attribute:: param_names

   List of strings of parameter names.

.. attribute:: prefix

   Prefix used for name-mangling of parameter names. The default is ``''``.
   If a particular :class:`Model` has arguments ``amplitude``,
   ``center``, and ``sigma``, these would become the parameter names.
   Using a prefix of ``'g1_'`` would convert these parameter names to
   ``g1_amplitude``, ``g1_center``, and ``g1_sigma``. This can be
   essential to avoid name collision in composite models.


Determining parameter names and independent variables for a function
--------------------------------------------------------------------

The :class:`Model` created from the supplied function ``func`` will create a
:class:`~lmfit.parameter.Parameters` object, and names are inferred from the
function` arguments, and a residual function is automatically constructed.

By default, the independent variable is taken as the first argument to the
function. You can, of course, explicitly set this, and will need to do so
if the independent variable is not first in the list, or if there is actually
more than one independent variable.

If not specified, Parameters are constructed from all positional arguments
and all keyword arguments that have a default value that is numerical, except
the independent variable, of course. Importantly, the Parameters can be
modified after creation. In fact, you will have to do this because none of the
parameters have valid initial values. In addition, one can place bounds and
constraints on Parameters, or fix their values.


Explicitly specifying ``independent_vars``
------------------------------------------

As we saw for the Gaussian example above, creating a :class:`Model` from a
function is fairly easy. Let's try another one:

.. jupyter-execute::

    import numpy as np
    from lmfit import Model


    def decay(t, tau, N):
       return N*np.exp(-t/tau)


    decay_model = Model(decay)
    print(f'independent variables: {decay_model.independent_vars}')

    params = decay_model.make_params()
    print('\nParameters:')
    for pname, par in params.items():
        print(pname, par)

Here, ``t`` is assumed to be the independent variable because it is the
first argument to the function. The other function arguments are used to
create parameters for the model.

If you want ``tau`` to be the independent variable in the above example,
you can say so:

.. jupyter-execute::

    decay_model = Model(decay, independent_vars=['tau'])
    print(f'independent variables: {decay_model.independent_vars}')

    params = decay_model.make_params()
    print('\nParameters:')
    for pname, par in params.items():
        print(pname, par)

You can also supply multiple values for multi-dimensional functions with
multiple independent variables. In fact, the meaning of *independent
variable* here is simple, and based on how it treats arguments of the
function you are modeling:

independent variable
    A function argument that is not a parameter or otherwise part of the
    model, and that will be required to be explicitly provided as a
    keyword argument for each fit with :meth:`Model.fit` or evaluation
    with :meth:`Model.eval`.

Note that independent variables are not required to be arrays, or even
floating point numbers.


Functions with keyword arguments
--------------------------------

If the model function had keyword parameters, these would be turned into
Parameters if the supplied default value was a valid number (but not
``None``, ``True``, or ``False``).

.. jupyter-execute::

    def decay2(t, tau, N=10, check_positive=False):
        if check_positive:
            arg = abs(t)/max(1.e-9, abs(tau))
        else:
            arg = t/tau
        return N*np.exp(arg)


    mod = Model(decay2)
    params = mod.make_params()
    print('Parameters:')
    for pname, par in params.items():
        print(pname, par)

Here, even though ``N`` is a keyword argument to the function, it is turned
into a parameter, with the default numerical value as its initial value.
By default, it is permitted to be varied in the fit -- the 10 is taken as
an initial value, not a fixed value. On the other hand, the
``check_positive`` keyword argument, was not converted to a parameter
because it has a boolean default value. In some sense,
``check_positive`` becomes like an independent variable to the model.
However, because it has a default value it is not required to be given for
each model evaluation or fit, as independent variables are.

Defining a ``prefix`` for the Parameters
----------------------------------------

As we will see in the next chapter when combining models, it is sometimes
necessary to decorate the parameter names in the model, but still have them
be correctly used in the underlying model function. This would be
necessary, for example, if two parameters in a composite model (see
:ref:`composite_models_section` or examples in the next chapter) would have
the same name. To avoid this, we can add a ``prefix`` to the
:class:`Model` which will automatically do this mapping for us.

.. jupyter-execute::

    def myfunc(x, amplitude=1, center=0, sigma=1):
        # function definition, for now just ``pass``
        pass


    mod = Model(myfunc, prefix='f1_')
    params = mod.make_params()
    print('Parameters:')
    for pname, par in params.items():
        print(pname, par)

You would refer to these parameters as ``f1_amplitude`` and so forth, and
the model will know to map these to the ``amplitude`` argument of ``myfunc``.


Initializing model parameter values
-----------------------------------

As mentioned above, creating a model does not automatically create the
corresponding :class:`~lmfit.parameter.Parameters`. These can be created with
either the :func:`create_params` function, or the :meth:`Model.make_params`
method of the corresponding instance of :class:`Model`.

When creating Parameters, each parameter is created with invalid initial value
of ``-Inf`` if it is not set explicitly. That is to say, parameter values
**must** be initialized in order for the model to evaluate a finite result or
used in a fit. There are a few different ways to do this:

  1. You can supply initial values in the definition of the model function.
  2. You can initialize the parameters when creating parameters with :meth:`Model.make_params`.
  3. You can create a Parameters object with :class:`Parameters` or :func:`create_params`.
  4. You can supply initial values for the parameters when calling
     :meth:`Model.eval` or :meth:`Model.fit` methods.

Generally, using the :meth:`Model.make_params` method is recommended. The methods
described above can be mixed, allowing you to overwrite initial values at any point
in the process of defining and using the model.


Initializing values in the function definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To supply initial values for parameters in the definition of the model
function, you can simply supply a default value:

.. jupyter-execute::

    def myfunc(x, a=1, b=0):
        return a*x + 10*a - b

instead of using:

.. jupyter-execute::

    def myfunc(x, a, b):
        return a*x + 10*a - b

This has the advantage of working at the function level -- all parameters
with keywords can be treated as options. It also means that some default
initial value will always be available for the parameter.


Initializing values with :meth:`Model.make_params`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating parameters with :meth:`Model.make_params` you can specify initial
values. To do this, use keyword arguments for the parameter names. You can
either set initial values as numbers (floats or ints) or as dictionaries with
keywords of (``value``, ``vary``, ``min``, ``max``, ``expr``, ``brute_step``,
and ``is_init_value``) to specify these parameter attributes.

.. jupyter-execute::

    mod = Model(myfunc)

    # simply supply initial values
    pars = mod.make_params(a=3, b=0.5)

    # supply initial values, attributes for bounds, etcetera:
    pars_bounded = mod.make_params(a=dict(value=3, min=0),
                                   b=dict(value=0.5, vary=False))


Creating a :class:`Parameters` object directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create your own Parameters directly using :func:`create_params`.
This is independent of using the :class:`Model` class, but is essentially
equivalent to :meth:`Model.make_params` except with less checking of errors for
model prefixes and so on.

.. jupyter-execute::

    from lmfit import create_params

    mod = Model(myfunc)

    # simply supply initial values
    pars = create_params(a=3, b=0.5)

    # supply initial values and attributes for bounds, etc:
    pars_bounded = create_params(a=dict(value=3, min=0),
                                 b=dict(value=0.5, vary=False))

Because less error checking is done, :meth:`Model.make_params` should probably
be preferred when using Models.


Initializing parameter values for a model with keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, you can explicitly supply initial values when using a model. That
is, as with :meth:`Model.make_params`, you can include values as keyword
arguments to either the :meth:`Model.eval` or :meth:`Model.fit` methods:

.. jupyter-execute::

    x = linspace(0, 10, 100)
    y_eval = mod.eval(x=x, a=7.0, b=-2.0)
    y_sim = y_eval + random.normal(0, 0.2, x.size)
    out = mod.fit(y_sim, pars, x=x, a=3.0, b=0.0)

These approaches to initialization provide many opportunities for setting
initial values for parameters. The methods can be combined, so that you
can set parameter hints but then change the initial value explicitly with
:meth:`Model.fit`.

.. _model_param_hints_section:

Using parameter hints
---------------------

After a model has been created, but prior to creating parameters with
:meth:`Model.make_params`, you can define parameter hints for that model. This
allows you to set other parameter attributes for bounds, whether it is varied in
the fit, or set a default constraint expression for a parameter. You can also
set the initial value, but that is not really the intention of the method,
which is to really to let you say that about the idealized Model, for example
that some values may not make sense for some parameters, or that some parameters
might be a small change from another parameter and so be fixed or constrained
by default.

To set a parameter hint, you can use :meth:`Model.set_param_hint`,
as with:

.. jupyter-execute::

    mod = Model(myfunc)
    mod.set_param_hint('bounded_parameter', min=0, max=1.0)
    pars = mod.make_params()

Parameter hints are discussed in more detail in section
:ref:`model_param_hints_section`.

Parameter hints are stored in a model's :attr:`param_hints` attribute,
which is simply a nested dictionary:

.. jupyter-execute::

    print('Parameter hints:')
    for pname, par in mod.param_hints.items():
        print(pname, par)

You can change this dictionary directly or use the :meth:`Model.set_param_hint`
method. Either way, these parameter hints are used by :meth:`Model.make_params`
when making parameters.

Parameter hints also allow you to create new parameters. This can be useful to
make derived parameters with constraint expressions. For example to get the
full-width at half maximum of a Gaussian model, one could use a parameter hint
of:

.. jupyter-execute::

    mod = Model(gaussian)
    mod.set_param_hint('wid', min=0)
    mod.set_param_hint('fwhm', expr='2.3548*wid')
    params = mod.make_params(amp={'value': 10, 'min':0.1, 'max':2000},
                             cen=5.5, wid=1.25)
    params.pretty_print()

With that definition, the value (and uncertainty) of the ``fwhm`` parameter
will be reported in the output of any fit done with that model.

.. _model_data_coercion_section:

Data Types for data  and independent data with ``Model``
-------------------------------------------------------------

The model as defined by your model function will use the independent
variable(s) you specify to best match the data you provide.  The model is meant
to be an abstract representation for data, but when you do a fit with
:meth:`Model.fit`, you really need to pass in values for the data to be modeled
and the independent data used to calculate that data.


As discussed in :ref:`fit-data-label`, the mathematical solvers used by
``lmfit`` all work exclusively with 1-dimensional numpy arrays of datatype
(dtype) "float64".  The value of the calculation ``(model-data)*weights`` using
the calculation of your model function, and the data and weights you pass in
**will always be coerced** to an 1-dimensional ndarray with dtype "float64"
when it is passed to the solver.  If it cannot be coerced, an error will occur
and the fit will be aborted.

That coercion will usually work for "array like" data that is not already a
float64 ndarray.  But, depending on the model function, the calculations within
the model function may not always work well for some "array like" data types
- especially independent data that are in list of numbers and ndarrays of type
"float32" or "int16" or less precision.


To be clear, independent data for models using ``Model`` are meant to be truly
independent, and not **not** required to be strictly numerical or objects that
are easily converted to arrays of numbers.  The could, for example, be a
dictionary, an instance of a user-defined class, or other type of structured
data.  You can use independent data any way you want in your model function.
But, as with almost all the examples given here, independent data is often also
a 1-dimensional array of values, say ``x``, and a simple view of the fit would
be to plot the data as ``y`` as a function of ``x``.  Again, this is not
required, but it is very common, especially for novice users.

By default, all data and independent data passed to :meth:`Model.fit` that is
"array like" - a list or tuple of numbers, a ``pandas.Series``, and
``h5py.Dataset``, or any object that has an ``__array__()`` method -- will be
converted to a "float64" ndarray before the fit begins.  If the array-like data
is complex, it will be converted to a "complex128" ndarray, which will always
work too.  This conversion before the fit begins ensures that the model
function sees only "float64 ndarrays", and nearly guarantees that data type
conversion will not cause problems for the fit.  But it also means that if you
have passed a ``pandas.Series`` as data or independent data, not all of the
methods or attributes of that ``Series`` will be available by default within
the model function.

.. versionadded:: 1.2.2

This coercion can be turned of with the ``coerce_farray`` option to
:meth:`Model.fit`.  When set to ``False``, neither the data nor the independent
data will be coerced from their original data type, and the user will be
responsible to arrange for the calculation and return value from the model
function to be allow a proper and accurate conversion to a "float64" ndarray.

See also :ref:`fit-data-label` for general advise and recommendations on
types of data to use when fitting data.

.. _model_saveload_sec:

Saving and Loading Models
-------------------------


It is sometimes desirable to save a :class:`Model` for later use outside of
the code used to define the model. Lmfit provides a :func:`save_model`
function that will save a :class:`Model` to a file. There is also a
companion :func:`load_model` function that can read this file and
reconstruct a :class:`Model` from it.

Saving a model turns out to be somewhat challenging. The main issue is that
Python is not normally able to *serialize* a function (such as the model
function making up the heart of the Model) in a way that can be
reconstructed into a callable Python object. The ``dill`` package can
sometimes serialize functions, but with the limitation that it can be used
only in the same version of Python. In addition, class methods used as
model functions will not retain the rest of the class attributes and
methods, and so may not be usable. With all those warnings, it should be
emphasized that if you are willing to save or reuse the definition of the
model function as Python code, then saving the Parameters and rest of the
components that make up a model presents no problem.

If the ``dill`` package is installed, the model function will also be saved
using it. But because saving the model function is not always reliable,
saving a model will always save the *name* of the model function. The
:func:`load_model` takes an optional :attr:`funcdefs` argument that can
contain a dictionary of function definitions with the function names as
keys and function objects as values. If one of the dictionary keys matches
the saved name, the corresponding function object will be used as the model
function. If it is not found by name, and if ``dill`` was used to save
the model, and if ``dill`` is available at run-time, the ``dill``-encoded
function will try to be used.  Note that this approach will generally allow
you to save a model that can be used by another installation of the
same version of Python, but may not work across Python versions.  For preserving
fits for extended periods of time (say, archiving for documentation of
scientific results), we strongly encourage you to save the full Python code
used for the model function and fit process.


.. autofunction:: save_model

.. autofunction:: load_model

As a simple example, one can save a model as:

.. jupyter-execute:: ../examples/doc_model_savemodel.py

To load that later, one might do:

.. jupyter-execute:: ../examples/doc_model_loadmodel.py
    :hide-output:

See also :ref:`modelresult_saveload_sec`.

The :class:`ModelResult` class
==============================

A :class:`ModelResult` (which had been called ``ModelFit`` prior to version
0.9) is the object returned by :meth:`Model.fit`. It is a subclass of
:class:`~lmfit.minimizer.Minimizer`, and so contains many of the fit results.
Of course, it knows the :class:`Model` and the set of
:class:`~lmfit.parameter.Parameters` used in the fit, and it has methods to
evaluate the model, to fit the data (or re-fit the data with changes to
the parameters, or fit with different or modified data) and to print out a
report for that fit.

While a :class:`Model` encapsulates your model function, it is fairly
abstract and does not contain the parameters or data used in a particular
fit. A :class:`ModelResult` *does* contain parameters and data as well as
methods to alter and re-do fits. Thus the :class:`Model` is the idealized
model while the :class:`ModelResult` is the messier, more complex (but perhaps
more useful) object that represents a fit with a set of parameters to data
with a model.


A :class:`ModelResult` has several attributes holding values for fit
results, and several methods for working with fits. These include
statistics inherited from :class:`~lmfit.minimizer.Minimizer` useful for
comparing different models, including ``chisqr``, ``redchi``, ``aic``,
and ``bic``.

.. autoclass:: ModelResult


:class:`ModelResult` methods
----------------------------

.. automethod:: ModelResult.eval

.. automethod:: ModelResult.eval_components

.. automethod:: ModelResult.fit

.. automethod:: ModelResult.fit_report

.. automethod:: ModelResult.summary

.. automethod:: ModelResult.conf_interval

.. automethod:: ModelResult.ci_report

.. automethod:: ModelResult.eval_uncertainty

.. automethod:: ModelResult.plot

.. automethod:: ModelResult.plot_fit

.. automethod:: ModelResult.plot_residuals


.. method:: ModelResult.iter_cb

   Optional callable function, to be called at each fit iteration. This
   must take take arguments of ``(params, iter, resid, *args, **kws)``, where
   ``params`` will have the current parameter values, ``iter`` the
   iteration, ``resid`` the current residual array, and ``*args`` and
   ``**kws`` as passed to the objective function. See :ref:`fit-itercb-label`.

.. method:: ModelResult.jacfcn

   Optional callable function, to be called to calculate Jacobian array.


:class:`ModelResult` attributes
-------------------------------

A :class:`ModelResult` will take all of the attributes of
:class:`MinimizerResult`, and several more. Here, we arrange them into
categories.


Parameters and Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: best_values

   Dictionary with parameter names as keys, and best-fit values as values.

.. attribute:: init_params

   Initial parameters, as passed to :meth:`Model.fit`.

.. attribute:: init_values

   Dictionary with parameter names as keys, and initial values as values.

.. attribute:: init_vals

   list of values for the variable parameters.

.. attribute::  params

   Parameters used in fit; will contain the best-fit values.

.. attribute:: uvars

   Dictionary of ``uncertainties`` ufloats from Parameters.

.. attribute::   var_names

   List of variable Parameter names used in optimization in the
   same order as the values in :attr:`init_vals` and :attr:`covar`.

Fit Arrays and Model
~~~~~~~~~~~~~~~~~~~~~~~

.. attribute:: best_fit

   numpy.ndarray result of model function, evaluated at provided
   independent variables and with best-fit parameters.


.. attribute:: covar

   numpy.ndarray (square) covariance matrix returned from fit.


.. attribute:: data

   numpy.ndarray of data to compare to model.

.. attribute:: dely

   numpy.ndarray of estimated uncertainties in the ``y`` values of the model
   from :meth:`ModelResult.eval_uncertainty`  (see :ref:`eval_uncertainty_sec`).

.. attribute:: dely_comps

   a dictionary of estimated uncertainties in the ``y`` values of the model
   components, from :meth:`ModelResult.eval_uncertainty` (see
   :ref:`eval_uncertainty_sec`).

.. attribute:: init_fit

   numpy.ndarray result of model function, evaluated at provided
   independent variables and with initial parameters.

.. attribute::  residual

   numpy.ndarray for residual.

.. attribute:: weights

   numpy.ndarray (or ``None``) of weighting values to be used in fit. If not
   ``None``, it will be used as a multiplicative factor of the residual
   array, so that ``weights*(data - fit)`` is minimized in the
   least-squares sense.

.. attribute:: components

   List of components of the :class:`Model`.



Fit Status
~~~~~~~~~~~~~~~~~~~

.. attribute:: aborted

   Whether the fit was aborted.

.. attribute:: errorbars

   Boolean for whether error bars were estimated by fit.

.. attribute:: flatchain

   A ``pandas.DataFrame`` view of the sampling chain if the ``emcee`` method is uses.

.. attribute::  ier

   Integer returned code from :scipydoc:`optimize.leastsq`.

.. attribute::  lmdif_message

   String message returned from :scipydoc:`optimize.leastsq`.

.. attribute::  message

   String message returned from :func:`~lmfit.minimizer.minimize`.

.. attribute::  method

   String naming fitting method for :func:`~lmfit.minimizer.minimize`.

.. attribute::  call_kws

   Dict of keyword arguments actually send to underlying solver with
   :func:`~lmfit.minimizer.minimize`.

.. attribute::  model

   Instance of :class:`Model` used for model.

.. attribute::  scale_covar

   Boolean flag for whether to automatically scale covariance matrix.


.. attribute:: userargs

   positional arguments passed to :meth:`Model.fit`, a tuple of (``y``, ``weights``)

.. attribute:: userkws

   keyword arguments passed to :meth:`Model.fit`, a dict, which will have independent data arrays such as ``x``.



Fit Statistics
~~~~~~~~~~~~~~~~~~~

.. attribute:: aic

   Floating point best-fit Akaike Information Criterion statistic
   (see :ref:`fit-results-label`).

.. attribute:: bic

   Floating point best-fit Bayesian Information Criterion statistic
   (see :ref:`fit-results-label`).

.. attribute:: chisqr

   Floating point best-fit chi-square statistic (see :ref:`fit-results-label`).

.. attribute:: ci_out

   Confidence interval data (see :ref:`confidence_chapter`) or ``None`` if
   the confidence intervals have not been calculated.

.. attribute::  ndata

   Integer number of data points.

.. attribute::  nfev

   Integer number of function evaluations used for fit.

.. attribute::  nfree

   Integer number of free parameters in fit.

.. attribute::  nvarys

   Integer number of independent, freely varying variables in fit.


.. attribute::  redchi

   Floating point reduced chi-square statistic (see :ref:`fit-results-label`).

.. attribute:: rsquared

   Floating point :math:`R^2` statistic, defined for data :math:`y` and best-fit model :math:`f` as

.. math::
   :nowrap:

   \begin{eqnarray*}
     R^2 &=&  1 - \frac{\sum_i (y_i - f_i)^2}{\sum_i (y_i - \bar{y})^2}
    \end{eqnarray*}

.. attribute:: success

   Boolean value of whether fit succeeded. This is an optimistic
   view of success, meaning that the method finished without error.



.. _eval_uncertainty_sec:

Calculating uncertainties in the model function
-----------------------------------------------

We return to the first example above and ask not only for the
uncertainties in the fitted parameters but for the range of values that
those uncertainties mean for the model function itself. We can use the
:meth:`ModelResult.eval_uncertainty` method of the model result object to
evaluate the uncertainty in the model with a specified level for
:math:`\sigma`.

That is, adding:

.. jupyter-execute:: ../examples/doc_model_gaussian.py
    :hide-output:
    :hide-code:

.. jupyter-execute::
    :hide-output:

    dely = result.eval_uncertainty(sigma=3)
    plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB",
                     label='3-$\sigma$ uncertainty band')

to the example fit to the Gaussian at the beginning of this chapter will
give 3-:math:`\sigma` bands for the best-fit Gaussian, and produce the
figure below.

.. jupyter-execute::
    :hide-code:

    plt.plot(x, y, 'o')
    plt.plot(x, result.init_fit, '--', label='initial fit')
    plt.plot(x, result.best_fit, '-', label='best fit')
    plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB",
                     label='3-$\sigma$ uncertainty band')
    plt.legend()
    plt.show()


.. versionadded:: 1.0.4

If the model is a composite built from multiple components, the
:meth:`ModelResult.eval_uncertainty` method will evaluate the uncertainty of
both the full model (often the sum of multiple components) as well as the
uncertainty in each component.  The uncertainty of the full model will be held in
``result.dely``, and the uncertainties for each component will be held in the dictionary
``result.dely_comps``, with keys that are the component prefixes.

An example script shows how the uncertainties in components of a composite
model can be calculated and used:

.. jupyter-execute:: ../examples/doc_model_uncertainty2.py



.. _modelresult_uvars_postfit_section:

Using uncertainties in the fitted parameters for post-fit calculations
--------------------------------------------------------------------------

.. versionadded:: 1.2.2

.. _uncertainties package:   https://pythonhosted.org/uncertainties/

As with the previous section, after a fit is complete, you may want to do some
further calculations with the resulting Parameter values.  Since these
Parameters will have not only best-fit values but also usually have
uncertainties, it is desirable for subsequent calculations to be able to
propagate those uncertainties to any resulting calculated value.  In addition,
it is common for Parameters to have finite - and sometimes large -
correlations which should be taken into account in such calculations.

The :attr:`ModelResult.uvars` will be a dictionary with keys for all variable
Parameters and values that are ``uvalues`` from the `uncertainties package`_.
When used in mathematical calculations with basic Python operators or numpy
functions, these ``uvalues`` will automatically propagate their uncertainties
to the resulting calculation, and taking into account the full covariance
matrix describing the correlation between values.

This readily allows "derived Parameters" to be evaluated just after the fit.
In fact, it might be useful to have a Model always do such a calculation just
after the fit.  The :meth:`Model.post_fit` method allows exactly that: you can
overwrite this otherwise empty method for any Model.  It takes one argument:
the :class:`ModelResult` instance just after the actual fit has run (and before
:meth:`Model.fit` returns) and can be used to add Parameters or do other
post-fit processing.

The following example script shows two different methods for calculating a centroid
value for two peaks, either by doing the calculation directly after the fit
with the ``result.uvars`` or by capturing this in a :meth:`Model.post_fit`
method that would be run for all instances of that model.  It also demonstrates
that taking correlations between Parameters into account when performing
calculations can have a noticeable influence on the resulting uncertainties.


.. jupyter-execute:: ../examples/doc_uvars_params.py


Note that the :meth:`Model.post_fit` does not need to be limited to this
use case of adding derived Parameters.


.. _modelresult_saveload_sec:

Saving and Loading ModelResults
-------------------------------

As with saving models (see section :ref:`model_saveload_sec`), it is
sometimes desirable to save a :class:`ModelResult`, either for later use or
to organize and compare different fit results. Lmfit provides a
:func:`save_modelresult` function that will save a :class:`ModelResult` to
a file. There is also a companion :func:`load_modelresult` function that
can read this file and reconstruct a :class:`ModelResult` from it.

As discussed in section :ref:`model_saveload_sec`, there are challenges to
saving model functions that may make it difficult to restore a saved a
:class:`ModelResult` in a way that can be used to perform a fit.
Use of the optional :attr:`funcdefs` argument is generally the most
reliable way to ensure that a loaded :class:`ModelResult` can be used to
evaluate the model function or redo the fit.

.. autofunction:: save_modelresult

.. autofunction:: load_modelresult

An example of saving a :class:`ModelResult` is:

.. jupyter-execute:: ../examples/doc_model_savemodelresult.py
    :hide-output:

To load that later, one might do:

.. jupyter-execute:: ../examples/doc_model_loadmodelresult.py
    :hide-output:

.. index:: Composite models

.. _composite_models_section:

Composite Models : adding (or multiplying) Models
=================================================

One of the more interesting features of the :class:`Model` class is that
Models can be added together or combined with basic algebraic operations
(add, subtract, multiply, and divide) to give a composite model. The
composite model will have parameters from each of the component models,
with all parameters being available to influence the whole model. This
ability to combine models will become even more useful in the next chapter,
when pre-built subclasses of :class:`Model` are discussed. For now, we'll
consider a simple example, and build a model of a Gaussian plus a line, as
to model a peak with a background. For such a simple problem, we could just
build a model that included both components:

.. jupyter-execute::

    def gaussian_plus_line(x, amp, cen, wid, slope, intercept):
        """line + 1-d gaussian"""

        gauss = (amp / (sqrt(2*pi) * wid)) * exp(-(x-cen)**2 / (2*wid**2))
        line = slope*x + intercept
        return gauss + line

and use that with:

.. jupyter-execute::

    mod = Model(gaussian_plus_line)

But we already had a function for a gaussian function, and maybe we'll
discover that a linear background isn't sufficient which would mean the
model function would have to be changed.

Instead, lmfit allows models to be combined into a :class:`CompositeModel`.
As an alternative to including a linear background in our model function,
we could define a linear function:

.. jupyter-execute::

    def line(x, slope, intercept):
        """a line"""
        return slope*x + intercept

and build a composite model with just:

.. jupyter-execute::

    mod = Model(gaussian) + Model(line)

This model has parameters for both component models, and can be used as:

.. jupyter-execute:: ../examples/doc_model_two_components.py
    :hide-output:

which prints out the results:

.. jupyter-execute::
    :hide-code:

     print(result.fit_report())

and shows the plot on the left.

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'o')
    axes[0].plot(x, result.init_fit, '--', label='initial fit')
    axes[0].plot(x, result.best_fit, '-', label='best fit')
    axes[0].legend()

    comps = result.eval_components()
    axes[1].plot(x, y, 'o')
    axes[1].plot(x, comps['gaussian'], '--', label='Gaussian component')
    axes[1].plot(x, comps['line'], '--', label='Line component')
    axes[1].legend()
    plt.show()

On the left, data is shown in blue dots, the total fit is shown in solid
green line, and the initial fit is shown as a orange dashed line. The figure
on the right shows again the data in blue dots, the Gaussian component as
a orange dashed line and the linear component as a green dashed line. It is
created using the following code:

.. jupyter-execute::
    :hide-output:

    comps = result.eval_components()
    plt.plot(x, y, 'o')
    plt.plot(x, comps['gaussian'], '--', label='Gaussian component')
    plt.plot(x, comps['line'], '--', label='Line component')

The components were generated after the fit using the
:meth:`ModelResult.eval_components` method of the ``result``, which returns
a dictionary of the components, using keys of the model name
(or ``prefix`` if that is set). This will use the parameter values in
``result.params`` and the independent variables (``x``) used during the
fit. Note that while the :class:`ModelResult` held in ``result`` does store the
best parameters and the best estimate of the model in ``result.best_fit``,
the original model and parameters in ``pars`` are left unaltered.

You can apply this composite model to other data sets, or evaluate the
model at other values of ``x``. You may want to do this to give a finer or
coarser spacing of data point, or to extrapolate the model outside the
fitting range. This can be done with:

.. jupyter-execute::

    xwide = linspace(-5, 25, 3001)
    predicted = mod.eval(result.params, x=xwide)

In this example, the argument names for the model functions do not overlap.
If they had, the ``prefix`` argument to :class:`Model` would have allowed
us to identify which parameter went with which component model. As we will
see in the next chapter, using composite models with the built-in models
provides a simple way to build up complex models.

.. autoclass::  CompositeModel(left, right, op[, **kws])

Note that when using built-in Python binary operators, a
:class:`CompositeModel` will automatically be constructed for you. That is,
doing:

.. jupyter-execute::
    :hide-code:

    def fcn1(x, a):
       pass

    def fcn2(x, b):
       pass

    def fcn3(x, c):
       pass

.. jupyter-execute::

     mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

will create a :class:`CompositeModel`. Here, ``left`` will be ``Model(fcn1)``,
``op`` will be :meth:`operator.add`, and ``right`` will be another
CompositeModel that has a ``left`` attribute of ``Model(fcn2)``, an ``op`` of
:meth:`operator.mul`, and a ``right`` of ``Model(fcn3)``.

To use a binary operator other than ``+``, ``-``, ``*``, or ``/`` you can
explicitly create a :class:`CompositeModel` with the appropriate binary
operator. For example, to convolve two models, you could define a simple
convolution function, perhaps as:

.. jupyter-execute::

    import numpy as np

    def convolve(dat, kernel):
        """simple convolution of two arrays"""
        npts = min(len(dat), len(kernel))
        pad = np.ones(npts)
        tmp = np.concatenate((pad*dat[0], dat, pad*dat[-1]))
        out = np.convolve(tmp, kernel, mode='valid')
        noff = int((len(out) - npts) / 2)
        return (out[noff:])[:npts]

which extends the data in both directions so that the convolving kernel
function gives a valid result over the data range. Because this function
takes two array arguments and returns an array, it can be used as the
binary operator. A full script using this technique is here:

.. jupyter-execute:: ../examples/doc_model_composite.py
    :hide-output:

which prints out the results:

.. jupyter-execute::
    :hide-code:

     print(result.fit_report())

and shows the plots:

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'o')
    axes[0].plot(x, result.init_fit, '--', label='initial fit')
    axes[0].plot(x, result.best_fit, '-', label='best fit')
    axes[0].legend()
    axes[1].plot(x, y, 'o')
    axes[1].plot(x, 10*comps['jump'], '--', label='Jump component')
    axes[1].plot(x, 10*comps['gaussian'], '-', label='Gaussian component')
    axes[1].legend()
    plt.show()

Using composite models with built-in or custom operators allows you to
build complex models from testable sub-components.
