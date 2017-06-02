.. _model_chapter:

=================================================
Modeling Data and Curve Fitting
=================================================

.. module:: lmfit.model

A common use of least-squares minimization is *curve fitting*, where one
has a parametrized model function meant to explain some phenomena and wants
to adjust the numerical values for the model so that it most closely
matches some data.  With :mod:`scipy`, such problems are typically solved
with :scipydoc:`optimize.curve_fit`, which is a wrapper around
:scipydoc:`optimize.leastsq`.  Since lmfit's
:func:`~lmfit.minimizer.minimize` is also a high-level wrapper around
:scipydoc:`optimize.leastsq` it can be used for curve-fitting problems.
While it offers many benefits over :scipydoc:`optimize.leastsq`, using
:func:`~lmfit.minimizer.minimize` for many curve-fitting problems still
requires more effort than using :scipydoc:`optimize.curve_fit`.

The :class:`Model` class in lmfit provides a simple and flexible approach
to curve-fitting problems.  Like :scipydoc:`optimize.curve_fit`, a
:class:`Model` uses a *model function* -- a function that is meant to
calculate a model for some phenomenon -- and then uses that to best match
an array of supplied data.  Beyond that similarity, its interface is rather
different from :scipydoc:`optimize.curve_fit`, for example in that it uses
:class:`~lmfit.parameter.Parameters`, but also offers several other
important advantages.

In addition to allowing you to turn any model function into a curve-fitting
method, lmfit also provides canonical definitions for many known line shapes
such as Gaussian or Lorentzian peaks and Exponential decays that are widely
used in many scientific domains.  These are available in the :mod:`models`
module that will be discussed in more detail in the next chapter
(:ref:`builtin_models_chapter`).  We mention it here as you may want to
consult that list before writing your own model.  For now, we focus on
turning Python functions into high-level fitting models with the
:class:`Model` class, and using these to fit data.


Motivation and simple example: Fit data to Gaussian profile
=============================================================

Let's start with a simple and common example of fitting data to a Gaussian
peak.  As we will see, there is a buit-in :class:`GaussianModel` class that
can help do this, but here we'll build our own.  We start with a simple
definition of the model function:

    >>> from numpy import sqrt, pi, exp, linspace, random
    >>>
    >>> def gaussian(x, amp, cen, wid):
    ...    return amp * exp(-(x-cen)**2 /wid)

We want to use this function to fit to data :math:`y(x)` represented by the
arrays `y` and `x`.  With :scipydoc:`optimize.curve_fit`, this would be::

    >>> from scipy.optimize import curve_fit
    >>>
    >>> x = linspace(-10,10, 101)
    >>> y = gaussian(x, 2.33, 0.21, 1.51) + random.normal(0, 0.2, len(x))
    >>>
    >>> init_vals = [1, 0, 1]     # for [amp, cen, wid]
    >>> best_vals, covar = curve_fit(gaussian, x, y, p0=init_vals)
    >>> print best_vals


That is, we create data, make an initial guess of the model values, and run
:scipydoc:`optimize.curve_fit` with the model function, data arrays, and
initial guesses.  The results returned are the optimal values for the
parameters and the covariance matrix.  It's simple and useful, but it
misses the benefits of lmfit.

With lmfit, we create a :class:`Model` that wraps the `gaussian` model
function, which automatically generates the appropriate residual function,
and determines the corresponding parameter names from the function
signature itself::

    >>> from lmfit import Model
    >>> gmodel = Model(gaussian)
    >>> gmodel.param_names
    set(['amp', 'wid', 'cen'])
    >>> gmodel.independent_vars
    ['x']

As you can see, the Model `gmodel` determined the names of the parameters
and the independent variables.  By default, the first argument of the
function is taken as the independent variable, held in
:attr:`independent_vars`, and the rest of the functions positional
arguments (and, in certain cases, keyword arguments -- see below) are used
for Parameter names.  Thus, for the `gaussian` function above, the
independent variable is `x`, and the parameters are named `amp`,
`cen`, and `wid`, and -- all taken directly from the signature of the
model function. As we will see below, you can modify the default
assignment of independent variable / arguments and specify yourself what
the independent variable is and which function arguments should be identified
as parameter names.

The Parameters are *not* created when the model is created. The model knows
what the parameters should be named, but not anything about the scale and
range of your data.  You will normally have to make these parameters and
assign initial values and other attributes.  To help you do this, each
model has a :meth:`make_params` method that will generate parameters with
the expected names:

    >>> params = gmod.make_params()

This creates the :class:`~lmfit.parameter.Parameters` but does not
automaticaly give them initial values since it has no idea what the scale
should be.  You can set initial values for parameters with keyword
arguments to :meth:`make_params`:

    >>> params = gmod.make_params(cen=5, amp=200, wid=1)

or assign them (and other parameter properties) after the
:class:`~lmfit.parameter.Parameters` class has been created.

A :class:`Model` has several methods associated with it.  For example, one
can use the :meth:`eval` method to evaluate the model or the :meth:`fit`
method to fit data to this model with a :class:`Parameter` object.  Both of
these methods can take explicit keyword arguments for the parameter values.
For example, one could use :meth:`eval` to calculate the predicted
function::

    >>> x = linspace(0, 10, 201)
    >>> y = gmod.eval(params, x=x)

or with::

    >>> y = gmod.eval(x=x, cen=6.5, amp=100, wid=2.0)

Admittedly, this a slightly long-winded way to calculate a Gaussian
function, given that you could have called your `gaussian` function
directly.  But now that the model is set up, we can use its
:meth:`fit` method to fit this model to data, as with::

    >>> result = gmod.fit(y, params)

or with::

    >>> result = gmod.fit(y, cen=6.5, amp=100, wid=2.0)

Putting everything together,  (included in the
``examples`` folder with the source code) is:

.. literalinclude:: ../examples/doc_model1.py

which is pretty compact and to the point.  The returned `result` will be
a :class:`ModelResult` object.  As we will see below, this has many
components, including a :meth:`fit_report` method, which will show::

    [[Model]]
        Model(gaussian)
    [[Fit Statistics]]
        # function evals   = 31
        # data points      = 101
        # variables        = 3
        chi-square         = 3.409
        reduced chi-square = 0.035
        Akaike info crit   = -336.264
        Bayesian info crit = -328.418
    [[Variables]]
        amp:   5.07800631 +/- 0.064957 (1.28%) (init= 5)
        cen:   5.65866112 +/- 0.010304 (0.18%) (init= 5)
        wid:   0.97344373 +/- 0.028756 (2.95%) (init= 1)
    [[Correlations]] (unreported correlations are <  0.100)
        C(amp, wid)                  = -0.577


As the script shows, the result will also have :attr:`init_fit` for the fit
with the initial parameter values and a :attr:`best_fit` for the fit with
the best fit parameter values.  These can be used to generate the following
plot:

.. image:: _images/model_fit1.png
   :target: _images/model_fit1.png
   :width: 50%

which shows the data in blue dots, the best fit as a solid red line, and
the initial fit as a dashed black line.

Note that the model fitting was really performed with::

    gmodel = Model(gaussian)
    result = gmodel.fit(y, params, x=x, amp=5, cen=5, wid=1)

These lines clearly express that we want to turn the `gaussian` function
into a fitting model, and then fit the :math:`y(x)` data to this model,
starting with values of 5 for `amp`, 5 for `cen` and 1 for `wid`.  In
addition, all the other features of lmfit are included:
:class:`~lmfit.parameter.Parameters` can have bounds and constraints and
the result is a rich object that can be reused to explore the model fit in
detail.


The :class:`Model` class
=======================================

The :class:`Model` class provides a general way to wrap a pre-defined
function as a fitting model.

.. autoclass::  Model


:class:`Model` class Methods
---------------------------------

.. automethod:: Model.eval

.. automethod:: Model.fit

.. automethod:: Model.guess

.. automethod:: Model.make_params


.. automethod:: Model.set_param_hint

   See :ref:`model_param_hints_section`.


.. automethod:: Model.print_param_hints


:class:`Model` class Attributes
---------------------------------

.. attribute:: func

   The model function used to calculate the model.

.. attribute:: independent_vars

   List of strings for names of the independent variables.

.. attribute:: missing

   Describes what to do for missing values.  The choices are:

    * None: Do not check for null or missing values (default).
    * 'none': Do not check for null or missing values.
    * 'drop': Drop null or missing observations in data.  If pandas is installed, :func:`pandas.isnull` is used, otherwise :func:`numpy.isnan` is used.
    * 'raise': Raise a (more helpful) exception when data contains null or missing values.

.. attribute:: name

   Name of the model, used only in the string representation of the
   model. By default this will be taken from the model function.

.. attribute:: opts

   Extra keyword arguments to pass to model function.  Normally this will
   be determined internally and should not be changed.

.. attribute:: param_hints

   Dictionary of parameter hints.  See :ref:`model_param_hints_section`.

.. attribute:: param_names

   List of strings of parameter names.

.. attribute:: prefix

   Prefix used for name-mangling of parameter names.  The default is ''.
   If a particular :class:`Model` has arguments `amplitude`,
   `center`, and `sigma`, these would become the parameter names.
   Using a prefix of `'g1_'` would convert these parameter names to
   `g1_amplitude`, `g1_center`, and `g1_sigma`.   This can be
   essential to avoid name collision in composite models.


Determining parameter names and independent variables for a function
-----------------------------------------------------------------------

The :class:`Model` created from the supplied function `func` will create
a :class:`~lmfit.parameter.Parameters` object, and names are inferred from the function
arguments, and a residual function is automatically constructed.


By default, the independent variable is take as the first argument to the
function.  You can, of course, explicitly set this, and will need to do so
if the independent variable is not first in the list, or if there are actually
more than one independent variables.

If not specified, Parameters are constructed from all positional arguments
and all keyword arguments that have a default value that is numerical, except
the independent variable, of course.   Importantly, the Parameters can be
modified after creation.  In fact, you will have to do this because none of the
parameters have valid initial values. In addition, one can place bounds and
constraints on Parameters, or fix their values.


Explicitly specifying ``independent_vars``
-------------------------------------------------

As we saw for the Gaussian example above, creating a :class:`Model` from a
function is fairly easy. Let's try another one::

    >>> from lmfit import Model
    >>> import numpy as np
    >>> def decay(t, tau, N):
    ...    return N*np.exp(-t/tau)
    ...
    >>> decay_model = Model(decay)
    >>> print decay_model.independent_vars
    ['t']
    >>> for pname, par in decay_model.params.items():
    ...     print pname, par
    ...
    tau <Parameter 'tau', None, bounds=[None:None]>
    N <Parameter 'N', None, bounds=[None:None]>

Here, `t` is assumed to be the independent variable because it is the
first argument to the function.  The other function arguments are used to
create parameters for the model.

If you want `tau` to be the independent variable in the above example,
you can say so::

    >>> decay_model = Model(decay, independent_vars=['tau'])
    >>> print decay_model.independent_vars
    ['tau']
    >>> for pname, par in decay_model.params.items():
    ...     print pname, par
    ...
    t <Parameter 't', None, bounds=[None:None]>
    N <Parameter 'N', None, bounds=[None:None]>


You can also supply multiple values for multi-dimensional functions with
multiple independent variables.  In fact, the meaning of *independent
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
-----------------------------------------

If the model function had keyword parameters, these would be turned into
Parameters if the supplied default value was a valid number (but not
None, True, or False).

    >>> def decay2(t, tau, N=10, check_positive=False):
    ...    if check_small:
    ...        arg = abs(t)/max(1.e-9, abs(tau))
    ...    else:
    ...        arg = t/tau
    ...    return N*np.exp(arg)
    ...
    >>> mod = Model(decay2)
    >>> for pname, par in mod.params.items():
    ...     print pname, par
    ...
    t <Parameter 't', None, bounds=[None:None]>
    N <Parameter 'N', 10, bounds=[None:None]>

Here, even though `N` is a keyword argument to the function, it is turned
into a parameter, with the default numerical value as its initial value.
By default, it is permitted to be varied in the fit -- the 10 is taken as
an initial value, not a fixed value.  On the other hand, the
`check_positive` keyword argument, was not converted to a parameter
because it has a boolean default value.    In some sense,
`check_positive` becomes like an independent variable to the model.
However, because it has a default value it is not required to be given for
each model evaluation or fit, as independent variables are.

Defining a `prefix` for the Parameters
--------------------------------------------

As we will see in the next chapter when combining models, it is sometimes
necessary to decorate the parameter names in the model, but still have them
be correctly used in the underlying model function.  This would be
necessary, for example, if two parameters in a composite model (see
:ref:`composite_models_section` or examples in the next chapter) would have
the same name.  To avoid this, we can add a `prefix` to the
:class:`Model` which will automatically do this mapping for us.

    >>> def myfunc(x, amplitude=1, center=0, sigma=1):
    ...

    >>> mod = Model(myfunc, prefix='f1_')
    >>> for pname, par in mod.params.items():
    ...     print pname, par
    ...
    f1_amplitude <Parameter 'f1_amplitude', None, bounds=[None:None]>
    f1_center <Parameter 'f1_center', None, bounds=[None:None]>
    f1_sigma <Parameter 'f1_sigma', None, bounds=[None:None]>

You would refer to these parameters as `f1_amplitude` and so forth, and
the model will know to map these to the `amplitude` argument of `myfunc`.


Initializing model parameters
--------------------------------

As mentioned above, the parameters created by :meth:`Model.make_params` are
generally created with invalid initial values of None.  These values
**must** be initialized in order for the model to be evaluated or used in a
fit.  There are four different ways to do this initialization that can be
used in any combination:

  1. You can supply initial values in the definition of the model function.
  2. You can initialize the parameters when creating parameters with :meth:`Model.make_params`.
  3. You can give parameter hints with :meth:`Model.set_param_hint`.
  4. You can supply initial values for the parameters when you use the
     :meth:`Model.eval` or :meth:`Model.fit` methods.

Of course these methods can be mixed, allowing you to overwrite initial
values at any point in the process of defining and using the model.

Initializing values in the function definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To supply initial values for parameters in the definition of the model
function, you can simply supply a default value::

    >>> def myfunc(x, a=1, b=0):
    >>>     ...

instead of using::

    >>> def myfunc(x, a, b):
    >>>     ...

This has the advantage of working at the function level -- all parameters
with keywords can be treated as options.  It also means that some default
initial value will always be available for the parameter.


Initializing values with :meth:`Model.make_params`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating parameters with :meth:`Model.make_params` you can specify initial
values.  To do this, use keyword arguments for the parameter names and
initial values::

    >>> mod = Model(myfunc)
    >>> pars = mod.make_params(a=3, b=0.5)


Initializing values by setting parameter hints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a model has been created, but prior to creating parameters with
:meth:`Model.make_params`, you can set parameter hints.  These allows you to set
not only a default initial value but also to set other parameter attributes
controlling bounds, whether it is varied in the fit, or a constraint
expression.  To set a parameter hint, you can use :meth:`Model.set_param_hint`,
as with::

    >>> mod = Model(myfunc)
    >>> mod.set_param_hint('a', value = 1.0)
    >>> mod.set_param_hint('b', value = 0.3, min=0, max=1.0)
    >>> pars = mod.make_params()

Parameter hints are discussed in more detail in section
:ref:`model_param_hints_section`.


Initializing values when using a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, you can explicitly supply initial values when using a model.  That
is, as with :meth:`Model.make_params`, you can include values
as keyword arguments to either the :meth:`Model.eval` or :meth:`Model.fit` methods::

   >>> y1 = mod.eval(x=x, a=7.0, b=-2.0)

   >>> out = mod.fit(x=x, pars, a=3.0, b=-0.0)

These approaches to initialization provide many opportunities for setting
initial values for parameters.  The methods can be combined, so that you
can set parameter hints but then change the initial value explicitly with
:meth:`Model.fit`.

.. _model_param_hints_section:

Using parameter hints
--------------------------------

After a model has been created, you can give it hints for how to create
parameters with :meth:`Model.make_params`.  This allows you to set not only a
default initial value but also to set other parameter attributes
controlling bounds, whether it is varied in the fit, or a constraint


expression.   To set a parameter hint, you can use :meth:`Model.set_param_hint`,
as with::

    >>> mod = Model(myfunc)
    >>> mod.set_param_hint('a', value = 1.0)
    >>> mod.set_param_hint('b', value = 0.3, min=0, max=1.0)

Parameter hints are stored in a model's :attr:`param_hints` attribute,
which is simply a nested dictionary::

    >>> print mod.param_hints
    {'a': {'value': 1}, 'b': {'max': 1.0, 'value': 0.3, 'min': 0}}


You can change this dictionary directly, or with the :meth:`Model.set_param_hint`
method.  Either way, these parameter hints are used by :meth:`Model.make_params`
when making parameters.

An important feature of parameter hints is that you can force the creation
of new parameters with parameter hints.  This can be useful to make derived
parameters with constraint expressions.  For example to get the full-width
at half maximum of a Gaussian model, one could use a parameter hint of::

    >>> mod = Model(gaussian)
    >>> mod.set_param_hint('fwhm', expr='2.3548*sigma')


The :class:`ModelResult` class
=======================================

A :class:`ModelResult` (which had been called `ModelFit` prior to version
0.9) is the object returned by :meth:`Model.fit`.  It is a subclass of
:class:`~lmfit.minimizer.Minimizer`, and so contains many of the fit results.
Of course, it knows the :class:`Model` and the set of
:class:`~lmfit.parameter.Parameters` used in the fit, and it has methods to
evaluate the model, to fit the data (or re-fit the data with changes to
the parameters, or fit with different or modified data) and to print out a
report for that fit.

While a :class:`Model` encapsulates your model function, it is fairly
abstract and does not contain the parameters or data used in a particular
fit.  A :class:`ModelResult` *does* contain parameters and data as well as
methods to alter and re-do fits.  Thus the :class:`Model` is the idealized
model while the :class:`ModelResult` is the messier, more complex (but perhaps
more useful) object that represents a fit with a set of parameters to data
with a model.


A :class:`ModelResult` has several attributes holding values for fit
results, and several methods for working with fits.  These include
statistics inherited from :class:`~lmfit.minimizer.Minimizer` useful for
comparing different models, including `chisqr`, `redchi`, `aic`, and `bic`.

.. autoclass:: ModelResult


:class:`ModelResult` methods
---------------------------------

.. automethod:: ModelResult.eval


.. automethod:: ModelResult.eval_components

.. automethod:: ModelResult.fit


.. automethod:: ModelResult.fit_report

.. automethod:: ModelResult.conf_interval

.. automethod:: ModelResult.ci_report

.. automethod:: ModelResult.eval_uncertainty

.. automethod:: ModelResult.plot

.. automethod:: ModelResult.plot_fit

.. automethod:: ModelResult.plot_residuals


:class:`ModelResult` attributes
---------------------------------

.. attribute:: aic

   Floating point best-fit Akaike Information Criterion statistic (see :ref:`fit-results-label`).

.. attribute:: best_fit

   numpy.ndarray result of model function, evaluated at provided
   independent variables and with best-fit parameters.

.. attribute:: best_values

   Dictionary with parameter names as keys, and best-fit values as values.

.. attribute:: bic

   Floating point best-fit Bayesian Information Criterion statistic (see :ref:`fit-results-label`).

.. attribute:: chisqr

   Floating point best-fit chi-square statistic (see :ref:`fit-results-label`).

.. attribute:: ci_out

   Confidence interval data (see :ref:`confidence_chapter`) or None if
   the confidence intervals have not been calculated.

.. attribute:: covar

   numpy.ndarray (square) covariance matrix returned from fit.

.. attribute:: data

   numpy.ndarray of data to compare to model.

.. attribute:: errorbars

   Boolean for whether error bars were estimated by fit.

.. attribute::  ier

   Integer returned code from :scipydoc:`optimize.leastsq`.

.. attribute:: init_fit

   numpy.ndarray result of model function, evaluated at provided
   independent variables and with initial parameters.

.. attribute:: init_params

   Initial parameters.

.. attribute:: init_values

   Dictionary with parameter names as keys, and initial values as values.

.. attribute:: iter_cb

   Optional callable function, to be called at each fit iteration.  This
   must take take arguments of ``(params, iter, resid, *args, **kws)``, where
   `params` will have the current parameter values, `iter` the
   iteration, `resid` the current residual array, and `*args` and
   `**kws` as passed to the objective function.  See :ref:`fit-itercb-label`.

.. attribute:: jacfcn

   Optional callable function, to be called to calculate Jacobian array.

.. attribute::  lmdif_message

   String message returned from :scipydoc:`optimize.leastsq`.

.. attribute::  message

   String message returned from :func:`~lmfit.minimizer.minimize`.

.. attribute::  method

   String naming fitting method for :func:`~lmfit.minimizer.minimize`.

.. attribute::  model

   Instance of :class:`Model` used for model.

.. attribute::  ndata

   Integer number of data points.

.. attribute::  nfev

   Integer number of function evaluations used for fit.

.. attribute::  nfree

   Integer number of free parameters in fit.

.. attribute::  nvarys

   Integer number of independent, freely varying variables in fit.

.. attribute::  params

   Parameters used in fit.  Will have best-fit values.

.. attribute::  redchi

   Floating point reduced chi-square statistic (see :ref:`fit-results-label`).

.. attribute::  residual

   numpy.ndarray for residual.

.. attribute::  scale_covar

   Boolean flag for whether to automatically scale covariance matrix.

.. attribute:: success

   Boolean value of whether fit succeeded.

.. attribute:: weights

   numpy.ndarray (or None) of weighting values to be used in fit.  If not
   None, it will be used as a multiplicative factor of the residual
   array, so that ``weights*(data - fit)`` is minimized in the
   least-squares sense.


Calculating uncertainties in the model function
-------------------------------------------------

We return to the first example above and ask not only for the
uncertainties in the fitted parameters but for the range of values that
those uncertainties mean for the model function itself.  We can use the
:meth:`ModelResult.eval_uncertainty` method of the model result object to
evaluate the uncertainty in the model with a specified level for
:math:`sigma`.

That is, adding::

    dely = result.eval_uncertainty(sigma=3)
    plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")

to the example fit to the Gaussian at the beginning of this chapter will
give :math:`3-sigma` bands for the best-fit Gaussian, and produce the
figure below.

.. _figModel4:

  .. image:: _images/model_fit4.png
     :target: _images/model_fit4.png
     :width: 50%



.. index:: Composite models

.. _composite_models_section:


Composite Models : adding (or multiplying) Models
==============================================================

One of the more interesting features of the :class:`Model` class is that
Models can be added together or combined with basic algebraic operations
(add, subtract, multiply, and divide) to give a composite model.  The
composite model will have parameters from each of the component models,
with all parameters being available to influence the whole model.  This
ability to combine models will become even more useful in the next chapter,
when pre-built subclasses of :class:`Model` are discussed.  For now, we'll
consider a simple example, and build a model of a Gaussian plus a line, as
to model a peak with a background. For such a simple problem, we could just
build a model that included both components::

    def gaussian_plus_line(x, amp, cen, wid, slope, intercept):
        "line + 1-d gaussian"

        gauss = (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))
        line = slope * x + intercept
        return gauss + line

and use that with::

    mod = Model(gaussian_plus_line)

But we already had a function for a gaussian function, and maybe we'll
discover that a linear background isn't sufficient which would mean the
model function would have to be changed.

Instead, lmfit allows models to be combined into a :class:`CompositeModel`.
As an alternative to including a linear background in our model function,
we could define a linear function::

    def line(x, slope, intercept):
        "a line"
        return slope * x + intercept

and build a composite model with just::

    mod = Model(gaussian) + Model(line)

This model has parameters for both component models, and can be used as:

.. literalinclude:: ../examples/doc_model2.py

which prints out the results::

    [[Model]]
        (Model(gaussian) + Model(line))
    [[Fit Statistics]]
        # function evals   = 44
        # data points      = 101
        # variables        = 5
        chi-square         = 2.579
        reduced chi-square = 0.027
        Akaike info crit   = -360.457
        Bayesian info crit = -347.381
    [[Variables]]
        amp:         8.45931061 +/- 0.124145 (1.47%) (init= 5)
        cen:         5.65547872 +/- 0.009176 (0.16%) (init= 5)
        intercept:  -0.96860201 +/- 0.033522 (3.46%) (init= 1)
        slope:       0.26484403 +/- 0.005748 (2.17%) (init= 0)
        wid:         0.67545523 +/- 0.009916 (1.47%) (init= 1)
    [[Correlations]] (unreported correlations are <  0.100)
        C(amp, wid)                  =  0.666
        C(cen, intercept)            =  0.129

and shows the plot on the left.

.. _figModel2:

  .. image:: _images/model_fit2.png
     :target: _images/model_fit2.png
     :width: 48%
  .. image:: _images/model_fit2a.png
     :target: _images/model_fit2a.png
     :width: 48%


On the left, data is shown in blue dots, the total fit is shown in solid
red line, and the initial fit is shown as a black dashed line.  In the
figure on the right, the data is again shown in blue dots, and the Gaussian
component shown as a black dashed line, and the linear component shown as a
red dashed line.  These components were generated after the fit using the
Models :meth:`ModelResult.eval_components` method of the `result`::

    comps = result.eval_components()

which returns a dictionary of the components, using keys of the model name
(or `prefix` if that is set).  This will use the parameter values in
`result.params` and the independent variables (`x`) used during the
fit.  Note that while the :class:`ModelResult` held in `result` does store the
best parameters and the best estimate of the model in `result.best_fit`,
the original model and parameters in `pars` are left unaltered.

You can apply this composite model to other data sets, or evaluate the
model at other values of `x`.  You may want to do this to give a finer or
coarser spacing of data point, or to extrapolate the model outside the
fitting range.  This can be done with::

    xwide = np.linspace(-5, 25, 3001)
    predicted = mod.eval(x=xwide)

In this example, the argument names for the model functions do not overlap.
If they had, the `prefix` argument to :class:`Model` would have allowed
us to identify which parameter went with which component model.  As we will
see in the next chapter, using composite models with the built-in models
provides a simple way to build up complex models.

.. autoclass::  CompositeModel(left, right, op[, **kws])

Note that when using builtin Python binary operators, a
:class:`CompositeModel` will automatically be constructed for you. That is,
doing::

     mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

will create a :class:`CompositeModel`.  Here, `left` will be `Model(fcn1)`,
`op` will be :meth:`operator.add`, and `right` will be another
CompositeModel that has a `left` attribute of `Model(fcn2)`, an `op` of
:meth:`operator.mul`, and a `right` of `Model(fcn3)`.

To use a binary operator other than '+', '-', '*', or '/' you can
explicitly create a :class:`CompositeModel` with the appropriate binary
operator.  For example, to convolve two models, you could define a simple
convolution function, perhaps as::

    import numpy as np
    def convolve(dat, kernel):
        # simple convolution
        npts = min(len(dat), len(kernel))
        pad  = np.ones(npts)
        tmp  = np.concatenate((pad*dat[0], dat, pad*dat[-1]))
        out  = np.convolve(tmp, kernel, mode='valid')
        noff = int((len(out) - npts)/2)
        return (out[noff:])[:npts]

which extends the data in both directions so that the convolving kernel
function gives a valid result over the data range.  Because this function
takes two array arguments and returns an array, it can be used as the
binary operator.  A full script using this technique is here:

.. literalinclude:: ../examples/doc_model3.py

which prints out the results::

    [[Model]]
        (Model(jump) <function convolve at 0x109ee4488> Model(gaussian))
    [[Fit Statistics]]
        # function evals   = 27
        # data points      = 201
        # variables        = 3
        chi-square         = 22.091
        reduced chi-square = 0.112
        Akaike info crit   = -437.837
        Bayesian info crit = -427.927
    [[Variables]]
        mid:         5 (fixed)
        sigma:       0.64118585 +/- 0.013233 (2.06%) (init= 1.5)
        center:      4.51633608 +/- 0.009567 (0.21%) (init= 3.5)
        amplitude:   0.62654849 +/- 0.001813 (0.29%) (init= 1)
    [[Correlations]] (unreported correlations are <  0.100)
        C(center, amplitude)         =  0.344
        C(sigma, amplitude)          =  0.280


and shows the plots:

.. _figModel3:

  .. image:: _images/model_fit3a.png
     :target: _images/model_fit3a.png
     :width: 48%
  .. image:: _images/model_fit3b.png
     :target: _images/model_fit3b.png
     :width: 48%

Using composite models with built-in or custom operators allows you to
build complex models from testable sub-components.
