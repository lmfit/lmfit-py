.. _builtin_models_chapter:

=====================================================
Built-in Fitting Models in the :mod:`models` module
=====================================================

.. module:: lmfit.models

Lmfit provides several built-in fitting models in the :mod:`models` module.
These pre-defined models each subclass from the :class:`~lmfit.model.Model` class of the
previous chapter and wrap relatively well-known functional forms, such as
Gaussians, Lorentzian, and Exponentials that are used in a wide range of
scientific domains.  In fact, all the models are based on simple, plain
Python functions defined in the :mod:`~lmfit.lineshapes` module.  In addition to
wrapping a function into a :class:`~lmfit.model.Model`, these models also provide a
:meth:`~lmfit.model.Model.guess` method that is intended to give a reasonable
set of starting values from a data array that closely approximates the
data to be fit.

As shown in the previous chapter, a key feature of the :class:`~lmfit.model.Model` class
is that models can easily be combined to give a composite
:class:`~lmfit.model.CompositeModel`. Thus, while some of the models listed here may
seem pretty trivial (notably, :class:`ConstantModel` and :class:`LinearModel`),
the main point of having these is to be able to use them in composite models. For
example, a Lorentzian plus a linear background might be represented as::

    from lmfit.models import LinearModel, LorentzianModel

    peak = LorentzianModel()
    background = LinearModel()
    model = peak + background

All the models listed below are one-dimensional, with an independent
variable named ``x``.  Many of these models represent a function with a
distinct peak, and so share common features.  To maintain uniformity,
common parameter names are used whenever possible.  Thus, most models have
a parameter called ``amplitude`` that represents the overall intensity (or
area of) a peak or function and a ``sigma`` parameter that gives a
characteristic width.

After a list of built-in models, a few examples of their use are given.

Peak-like models
-------------------

There are many peak-like models available.  These include
:class:`GaussianModel`, :class:`LorentzianModel`, :class:`VoigtModel`,
:class:`PseudoVoigtModel`, and some less commonly used variations.  Most of
these models are *unit-normalized* and share the same parameter names so
that you can easily switch between models and interpret the results.  The
``amplitude`` parameter is the multiplicative factor for the
unit-normalized peak lineshape, and so will represent the strength of that
peak or the area under that curve.  The ``center`` parameter will be the
centroid ``x`` value.  The ``sigma`` parameter is the characteristic width
of the peak, with many functions using :math:`(x-\mu)/\sigma` where
:math:`\mu` is the centroid value.  Most of these peak functions will have
two additional parameters derived from and constrained by the other
parameters. The first of these is ``fwhm`` which will hold the estimated
"Full Width at Half Max" for the peak, which is often easier to compare
between different models than ``sigma``.  The second of these is ``height``
which will contain the maximum value of the peak, typically the value at
:math:`x = \mu`.  Finally, each of these models has a :meth:`guess` method
that uses data to make a fairly crude but usually sufficient guess for the
value of ``amplitude``, ``center``, and ``sigma``, and sets a lower bound
of 0 on the value of ``sigma``.

:class:`GaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianModel

:class:`LorentzianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LorentzianModel

:class:`SplitLorentzianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SplitLorentzianModel

:class:`VoigtModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: VoigtModel

:class:`PseudoVoigtModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PseudoVoigtModel

:class:`MoffatModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MoffatModel

:class:`Pearson7Model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Pearson7Model

:class:`StudentsTModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StudentsTModel

:class:`BreitWignerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BreitWignerModel

:class:`LognormalModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LognormalModel

:class:`DampedOcsillatorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DampedOscillatorModel

:class:`DampedHarmonicOcsillatorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DampedHarmonicOscillatorModel

:class:`ExponentialGaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialGaussianModel

:class:`SkewedGaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SkewedGaussianModel

:class:`SkewedVoigtModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SkewedVoigtModel

:class:`ThermalDistributionModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ThermalDistributionModel

:class:`DonaichModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: DonaichModel


Linear and Polynomial Models
------------------------------------

These models correspond to polynomials of some degree.  Of course, lmfit is
a very inefficient way to do linear regression (see :numpydoc:`polyfit`
or :scipydoc:`stats.linregress`), but these models may be useful as one
of many components of a composite model.

:class:`ConstantModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConstantModel

:class:`LinearModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearModel

:class:`QuadraticModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: QuadraticModel

:class:`PolynomialModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PolynomialModel


Step-like models
-----------------------------------------------

Two models represent step-like functions, and share many characteristics.

:class:`StepModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: StepModel

:class:`RectangleModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RectangleModel


Exponential and Power law models
-----------------------------------------------

:class:`ExponentialModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialModel

:class:`PowerLawModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PowerLawModel


User-defined Models
----------------------------

.. _asteval: https://newville.github.io/asteval/

As shown in the previous chapter (:ref:`model_chapter`), it is fairly
straightforward to build fitting models from parametrized Python functions.
The number of model classes listed so far in the present chapter should
make it clear that this process is not too difficult.  Still, it is
sometimes desirable to build models from a user-supplied function.  This
may be especially true if model-building is built-in to some larger library
or application for fitting in which the user may not be able to easily
build and use a new model from Python code.


The :class:`ExpressionModel` allows a model to be built from a
user-supplied expression.  This uses the `asteval`_ module also used for
mathematical constraints as discussed in :ref:`constraints_chapter`.


:class:`ExpressionModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExpressionModel

Since the point of this model is that an arbitrary expression will be
supplied, the determination of what are the parameter names for the model
happens when the model is created.  To do this, the expression is parsed,
and all symbol names are found.  Names that are already known (there are
over 500 function and value names in the asteval namespace, including most
Python built-ins, more than 200 functions inherited from NumPy, and more
than 20 common lineshapes defined in the :mod:`lineshapes` module) are not
converted to parameters.  Unrecognized names are expected to be names of either
parameters or independent variables.  If ``independent_vars`` is the
default value of None, and if the expression contains a variable named
``x``, that will be used as the independent variable.  Otherwise,
``independent_vars`` must be given.

For example, if one creates an :class:`ExpressionModel` as::

    mod = ExpressionModel('off + amp * exp(-x/x0) * sin(x*phase)')

The name ``exp`` will be recognized as the exponent function, so the model
will be interpreted to have parameters named ``off``, ``amp``, ``x0`` and
``phase``. In addition, ``x`` will be assumed to be the sole independent variable.
In general, there is no obvious way to set default parameter values or
parameter hints for bounds, so this will have to be handled explicitly.

To evaluate this model, you might do the following::

    x = numpy.linspace(0, 10, 501)
    params = mod.make_params(off=0.25, amp=1.0, x0=2.0, phase=0.04)
    y = mod.eval(params, x=x)

While many custom models can be built with a single line expression
(especially since the names of the lineshapes like ``gaussian``, ``lorentzian``
and so on, as well as many NumPy functions, are available), more complex
models will inevitably require multiple line functions.  You can include
such Python code with the ``init_script`` argument.  The text of this script
is evaluated when the model is initialized (and before the actual
expression is parsed), so that you can define functions to be used
in your expression.

As a probably unphysical example, to make a model that is the derivative of
a Gaussian function times the logarithm of a Lorentzian function you may
could to define this in a script::

    script = """
    def mycurve(x, amp, cen, sig):
        loren = lorentzian(x, amplitude=amp, center=cen, sigma=sig)
        gauss = gaussian(x, amplitude=amp, center=cen, sigma=sig)
        return log(loren) * gradient(gauss) / gradient(x)
    """

and then use this with :class:`ExpressionModel` as::

    mod = ExpressionModel('mycurve(x, height, mid, wid)', init_script=script,
                          independent_vars=['x'])

As above, this will interpret the parameter names to be ``height``, ``mid``,
and ``wid``, and build a model that can be used to fit data.



Example 1: Fit Peak data to Gaussian, Lorentzian, and  Voigt profiles
------------------------------------------------------------------------

Here, we will fit data to three similar line shapes, in order to decide which
might be the better model.  We will start with a Gaussian profile, as in
the previous chapter, but use the built-in :class:`GaussianModel` instead
of writing one ourselves.  This is a slightly different version from the
one in previous example in that the parameter names are different, and have
built-in default values.  We will simply use:

.. jupyter-execute::
    :hide-output:

    from numpy import loadtxt

    from lmfit.models import GaussianModel

    data = loadtxt('test_peak.dat')
    x = data[:, 0]
    y = data[:, 1]

    mod = GaussianModel()

    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)

    print(out.fit_report(min_correl=0.25))

which prints out the results:

.. jupyter-execute::
    :hide-code:

    print(out.fit_report(min_correl=0.25))

We see a few interesting differences from the results of the previous
chapter. First, the parameter names are longer. Second, there are ``fwhm``
and ``height`` parameters, to give the full-width-at-half-maximum and
maximum peak height, respectively. And third, the automated initial guesses
are pretty good. A plot of the fit:

.. jupyter-execute::
    :hide-code:

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 150
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'

    import matplotlib.pyplot as plt
    plt.plot(x, y, 'b-')
    plt.plot(x, out.best_fit, 'r-', label='Gaussian Model')
    plt.legend(loc='best')
    plt.show()

shows a decent match to the data -- the fit worked with no explicit setting
of initial parameter values.  Looking more closely, the fit is not perfect,
especially in the tails of the peak, suggesting that a different peak
shape, with longer tails, should be used.  Perhaps a Lorentzian would be
better?  To do this, we simply replace ``GaussianModel`` with
``LorentzianModel`` to get a :class:`LorentzianModel`:

.. jupyter-execute::

    from lmfit.models import LorentzianModel
    mod = LorentzianModel()

with the rest of the script as above.  Perhaps predictably, the first thing
we try gives results that are worse by comaparing the fit statistics:

.. jupyter-execute::
    :hide-code:

    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))

and also by visual inspection of the fit to the data (figure below).

.. jupyter-execute::
    :hide-code:

     plt.plot(x, y, 'b-')
     plt.plot(x, out.best_fit, 'r-', label='Lorentzian Model')
     plt.legend(loc='best')
     plt.show()

The tails are now too big, and the value for :math:`\chi^2` almost doubled.
A Voigt model does a better job. Using :class:`VoigtModel`, this is as simple as using:

.. jupyter-execute::

    from lmfit.models import VoigtModel
    mod = VoigtModel()

with all the rest of the script as above.  This gives:

.. jupyter-execute::
    :hide-code:

    pars = mod.guess(y, x=x)
    out = mod.fit(y, pars, x=x)
    print(out.fit_report(min_correl=0.25))

which has a much better value for :math:`\chi^2` and the other
goodness-of-fit measures, and an obviously better match to the data as seen
in the figure below (left).

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'b-')
    axes[0].plot(x, out.best_fit, 'r-', label='Voigt Model\ngamma constrained')
    axes[0].legend(loc='best')
    # free gamma parameter
    pars['gamma'].set(value=0.7, vary=True, expr='')
    out_gamma = mod.fit(y, pars, x=x)
    axes[1].plot(x, y, 'b-')
    axes[1].plot(x, out_gamma.best_fit, 'r-', label='Voigt Model\ngamma unconstrained')
    axes[1].legend(loc='best')
    plt.show()

Fit to peak with Voigt model (left) and Voigt model with ``gamma``
varying independently of ``sigma`` (right).

Can we do better? The Voigt function has a :math:`\gamma` parameter
(``gamma``) that can be distinct from ``sigma``.  The default behavior used
above constrains ``gamma`` to have exactly the same value as ``sigma``.  If
we allow these to vary separately, does the fit improve?  To do this, we
have to change the ``gamma`` parameter from a constrained expression and
give it a starting value using something like::

   mod = VoigtModel()
   pars = mod.guess(y, x=x)
   pars['gamma'].set(value=0.7, vary=True, expr='')

which gives:

.. jupyter-execute::
    :hide-code:

    print(out_gamma.fit_report(min_correl=0.25))

and the fit shown on the right above.

Comparing the two fits with the Voigt function, we see that :math:`\chi^2`
is definitely improved with a separately varying ``gamma`` parameter.  In
addition, the two values for ``gamma`` and ``sigma`` differ significantly
-- well outside the estimated uncertainties.  More compelling, reduced
:math:`\chi^2` is improved even though a fourth variable has been added to
the fit.  In the simplest statistical sense, this suggests that ``gamma``
is a significant variable in the model.  In addition, we can use both the
Akaike or Bayesian Information Criteria (see
:ref:`information_criteria_label`) to assess how likely the model with
variable ``gamma`` is to explain the data than the model with ``gamma``
fixed to the value of ``sigma``.  According to theory,
:math:`\exp(-(\rm{AIC1}-\rm{AIC0})/2)` gives the probability that a model with
AIC1 is more likely than a model with AIC0.  For the two models here, with
AIC values of -1436 and -1324 (Note: if we had more carefully set the value
for ``weights`` based on the noise in the data, these values might be
positive, but there difference would be roughly the same), this says that
the model with ``gamma`` fixed to ``sigma`` has a probability less than 5.e-25
of being the better model.


Example 2: Fit data to a Composite Model with pre-defined models
------------------------------------------------------------------

Here, we repeat the point made at the end of the last chapter that
instances of :class:`~lmfit.model.Model` class can be added together to make a
*composite model*.  By using the large number of built-in models available,
it is therefore very simple to build models that contain multiple peaks and
various backgrounds.  An example of a simple fit to a noisy step function
plus a constant:

.. jupyter-execute:: ../examples/doc_builtinmodels_stepmodel.py
    :hide-output:

After constructing step-like data, we first create a :class:`StepModel`
telling it to use the ``erf`` form (see details above), and a
:class:`ConstantModel`.  We set initial values, in one case using the data
and :meth:`guess` method for the initial step function paramaters, and
:meth:`make_params` arguments for the linear component.
After making a composite model, we run :meth:`fit` and report the
results, which gives:

.. jupyter-execute::
    :hide-code:

    print(out.fit_report())

with a plot of

.. jupyter-execute::
    :hide-code:

    plt.plot(x, y, 'b')
    plt.plot(x, out.init_fit, 'k--', label='initial fit')
    plt.plot(x, out.best_fit, 'r-', label='best fit')
    plt.legend(loc='best')
    plt.show()


Example 3: Fitting Multiple Peaks -- and using Prefixes
------------------------------------------------------------------

.. _NIST StRD: https://itl.nist.gov/div898/strd/nls/nls_main.shtml

As shown above, many of the models have similar parameter names.  For
composite models, this could lead to a problem of having parameters for
different parts of the model having the same name.  To overcome this, each
:class:`~lmfit.model.Model` can have a ``prefix`` attribute (normally set to a blank
string) that will be put at the beginning of each parameter name.  To
illustrate, we fit one of the classic datasets from the `NIST StRD`_ suite
involving a decaying exponential and two gaussians.

.. jupyter-execute:: ../examples/doc_builtinmodels_nistgauss.py
    :hide-output:

where we give a separate prefix to each model (they all have an
``amplitude`` parameter).  The ``prefix`` values are attached transparently
to the models.

Note that the calls to :meth:`make_param` used the bare name, without the
prefix.  We could have used the prefixes, but because we used the
individual model ``gauss1`` and ``gauss2``, there was no need.

Note also in the example here that we explicitly set bounds on many of the
parameter values.

The fit results printed out are:

.. jupyter-execute::
    :hide-code:

    print(out.fit_report())

We get a very good fit to this problem (described at the NIST site as of
average difficulty, but the tests there are generally deliberately challenging) by
applying reasonable initial guesses and putting modest but explicit bounds
on the parameter values.  The overall fit is shown on the left, with its individual
components displayed on the right:

.. jupyter-execute::
    :hide-code:

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].plot(x, y, 'b')
    axes[0].plot(x, init, 'k--', label='initial fit')
    axes[0].plot(x, out.best_fit, 'r-', label='best fit')
    axes[0].legend(loc='best')

    comps = out.eval_components(x=x)
    axes[1].plot(x, y, 'b')
    axes[1].plot(x, comps['g1_'], 'g--', label='Gaussian component 1')
    axes[1].plot(x, comps['g2_'], 'm--', label='Gaussian component 2')
    axes[1].plot(x, comps['exp_'], 'k--', label='Exponential component')
    axes[1].legend(loc='best')

    plt.show()

One final point on setting initial values.  From looking at the data
itself, we can see the two Gaussian peaks are reasonably well separated but
do overlap. Furthermore, we can tell that the initial guess for the
decaying exponential component was poorly estimated because we used the
full data range.  We can simplify the initial parameter values by using
this, and by defining an :func:`index_of` function to limit the data range.
That is, with::

    def index_of(arrval, value):
        """Return index of array *at or below* value."""
        if value < min(arrval):
            return 0
        return max(np.where(arrval <= value)[0])

    ix1 = index_of(x, 75)
    ix2 = index_of(x, 135)
    ix3 = index_of(x, 175)

    exp_mod.guess(y[:ix1], x=x[:ix1])
    gauss1.guess(y[ix1:ix2], x=x[ix1:ix2])
    gauss2.guess(y[ix2:ix3], x=x[ix2:ix3])


.. jupyter-execute:: ../examples/doc_builtinmodels_nistgauss2.py
    :hide-code:
    :hide-output:

we can get a better initial estimate (see below).

.. jupyter-execute::
    :hide-code:

    plt.plot(x, y, 'b')
    plt.plot(x, out.init_fit, 'k--', label='initial fit')
    plt.plot(x, out.best_fit, 'r-', label='best fit')
    plt.legend(loc='best')

    plt.show()

The fit converges to the same answer, giving to identical values
(to the precision printed out in the report), but in fewer steps,
and without any bounds on parameters at all:

.. jupyter-execute::
    :hide-code:

    print(out.fit_report())

This script is in the file ``doc_builtinmodels_nistgauss2.py`` in the examples folder,
and the figure above shows an improved initial estimate of the data.
