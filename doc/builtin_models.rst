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

    >>> from lmfit.models import LinearModel, LorentzianModel
    >>> peak = LorentzianModel()
    >>> background = LinearModel()
    >>> model = peak + background

All the models listed below are one dimensional, with an independent
variable named ``x``.  Many of these models represent a function with a
distinct peak, and so share common features.  To maintain uniformity,
common parameter names are used whenever possible.  Thus, most models have
a parameter called ``amplitude`` that represents the overall height (or
area of) a peak or function, a ``center`` parameter that represents a peak
centroid position, and a ``sigma`` parameter that gives a characteristic
width.  Many peak shapes also have a parameter ``fwhm`` (constrained by
``sigma``) giving the full width at half maximum and a parameter ``height``
(constrained by ``sigma`` and ``amplitude``) to give the maximum peak
height.

After a list of built-in models, a few examples of their use are given.

Peak-like models
-------------------

There are many peak-like models available.  These include
:class:`GaussianModel`, :class:`LorentzianModel`, :class:`VoigtModel` and
some less commonly used variations.  The :meth:`guess`
methods for all of these make a fairly crude guess for the value of
``amplitude``, but also set a lower bound of 0 on the value of ``sigma``.

:class:`GaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GaussianModel

:class:`LorentzianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LorentzianModel


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
parameters or independent variables.  If `independent_vars` is the
default value of None, and if the expression contains a variable named
`x`, that will be used as the independent variable.  Otherwise,
`independent_vars` must be given.

For example, if one creates an :class:`ExpressionModel` as::

    >>> mod = ExpressionModel('off + amp * exp(-x/x0) * sin(x*phase)')

The name `exp` will be recognized as the exponent function, so the model
will be interpreted to have parameters named `off`, `amp`, `x0` and
`phase`. In addition, `x` will be assumed to be the sole independent variable.
In general, there is no obvious way to set default parameter values or
parameter hints for bounds, so this will have to be handled explicitly.

To evaluate this model, you might do the following::

    >>> x = numpy.linspace(0, 10, 501)
    >>> params = mod.make_params(off=0.25, amp=1.0, x0=2.0, phase=0.04)
    >>> y = mod.eval(params, x=x)

While many custom models can be built with a single line expression
(especially since the names of the lineshapes like `gaussian`, `lorentzian`
and so on, as well as many NumPy functions, are available), more complex
models will inevitably require multiple line functions.  You can include
such Python code with the `init_script` argument.  The text of this script
is evaluated when the model is initialized (and before the actual
expression is parsed), so that you can define functions to be used
in your expression.

As a probably unphysical example, to make a model that is the derivative of
a Gaussian function times the logarithm of a Lorentzian function you may
could to define this in a script::

    >>> script = """
    def mycurve(x, amp, cen, sig):
        loren = lorentzian(x, amplitude=amp, center=cen, sigma=sig)
        gauss = gaussian(x, amplitude=amp, center=cen, sigma=sig)
        return log(loren) * gradient(gauss) / gradient(x)
    """

and then use this with :class:`ExpressionModel` as::

    >>> mod = ExpressionModel('mycurve(x, height, mid, wid)',
                              init_script=script,
                              independent_vars=['x'])

As above, this will interpret the parameter names to be `height`, `mid`,
and `wid`, and build a model that can be used to fit data.



Example 1: Fit Peak data to Gaussian, Lorentzian, and  Voigt profiles
------------------------------------------------------------------------

Here, we will fit data to three similar line shapes, in order to decide which
might be the better model.  We will start with a Gaussian profile, as in
the previous chapter, but use the built-in :class:`GaussianModel` instead
of writing one ourselves.  This is a slightly different version from the
one in previous example in that the parameter names are different, and have
built-in default values.  We will simply use::

     from numpy import loadtxt

     from lmfit.models import GaussianModel

     data = loadtxt('test_peak.dat')
     x = data[:, 0]
     y = data[:, 1]

     mod = GaussianModel()

     pars = mod.guess(y, x=x)
     out = mod.fit(y, pars, x=x)
     print(out.fit_report(min_correl=0.25))


which prints out the results::

    [[Model]]
        Model(gaussian)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 27
        # data points      = 401
        # variables        = 3
        chi-square         = 29.9943157
        reduced chi-square = 0.07536260
        Akaike info crit   = -1033.77437
        Bayesian info crit = -1021.79248
    [[Variables]]
        sigma:      1.23218359 +/- 0.00737496 (0.60%) (init = 1.35)
        amplitude:  30.3135620 +/- 0.15712686 (0.52%) (init = 43.62238)
        center:     9.24277047 +/- 0.00737496 (0.08%) (init = 9.25)
        fwhm:       2.90157056 +/- 0.01736670 (0.60%) == '2.3548200*sigma'
        height:     9.81457817 +/- 0.05087283 (0.52%) == '0.3989423*amplitude/max(1.e-15, sigma)'
    [[Correlations]] (unreported correlations are < 0.250)
        C(sigma, amplitude) =  0.577

We see a few interesting differences from the results of the previous
chapter. First, the parameter names are longer. Second, there are ``fwhm``
and ``height`` parameters, to give the full width at half maximum and
maximum peak height.  And third, the automated initial guesses are pretty
good. A plot of the fit:

.. _figA1:

  .. image::  _images/models_peak1.png
     :target: _images/models_peak1.png
     :width: 48 %
  .. image::  _images/models_peak2.png
     :target: _images/models_peak2.png
     :width: 48 %

  Fit to peak with Gaussian (left) and Lorentzian (right) models.

shows a decent match to the data -- the fit worked with no explicit setting
of initial parameter values.  Looking more closely, the fit is not perfect,
especially in the tails of the peak, suggesting that a different peak
shape, with longer tails, should be used.  Perhaps a Lorentzian would be
better?  To do this, we simply replace ``GaussianModel`` with
``LorentzianModel`` to get a :class:`LorentzianModel`::

    from lmfit.models import LorentzianModel
    mod = LorentzianModel()

with the rest of the script as above.  Perhaps predictably, the first thing
we try gives results that are worse::

    [[Model]]
        Model(lorentzian)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 23
        # data points      = 401
        # variables        = 3
        chi-square         = 53.7535387
        reduced chi-square = 0.13505914
        Akaike info crit   = -799.830322
        Bayesian info crit = -787.848438
    [[Variables]]
        sigma:      1.15483925 +/- 0.01315659 (1.14%) (init = 1.35)
        center:     9.24438944 +/- 0.00927619 (0.10%) (init = 9.25)
        amplitude:  38.9727645 +/- 0.31386183 (0.81%) (init = 54.52798)
        fwhm:       2.30967850 +/- 0.02631318 (1.14%) == '2.0000000*sigma'
        height:     10.7421156 +/- 0.08633945 (0.80%) == '0.3183099*amplitude/max(1.e-15, sigma)'
    [[Correlations]] (unreported correlations are < 0.250)
        C(sigma, amplitude) =  0.709

with the plot shown on the right in the figure above.  The tails are now
too big, and the value for :math:`\chi^2` almost doubled.  A Voigt model
does a better job.  Using :class:`VoigtModel`, this is as simple as using::

    from lmfit.models import VoigtModel
    mod = VoigtModel()

with all the rest of the script as above.  This gives::

    [[Model]]
        Model(voigt)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 23
        # data points      = 401
        # variables        = 3
        chi-square         = 14.5448627
        reduced chi-square = 0.03654488
        Akaike info crit   = -1324.00615
        Bayesian info crit = -1312.02427
    [[Variables]]
        center:     9.24411150 +/- 0.00505482 (0.05%) (init = 9.25)
        sigma:      0.73015627 +/- 0.00368460 (0.50%) (init = 0.8775)
        amplitude:  35.7554146 +/- 0.13861321 (0.39%) (init = 65.43358)
        gamma:      0.73015627 +/- 0.00368460 (0.50%) == 'sigma'
        fwhm:       2.62951907 +/- 0.01326940 (0.50%) == '3.6013100*sigma'
        height:     10.2203969 +/- 0.03009415 (0.29%) == 'amplitude*wofz((1j*gamma)/(sigma*sqrt(2))).real/(sigma*sqrt(2*pi))'
    [[Correlations]] (unreported correlations are < 0.250)
        C(sigma, amplitude) =  0.651

which has a much better value for :math:`\chi^2` and the other
goodness-of-fit measures, and an obviously better match to the data as seen
in the figure below (left).

.. _figA2:

  .. image::  _images/models_peak3.png
     :target: _images/models_peak3.png
     :width: 48 %
  .. image::  _images/models_peak4.png
     :target: _images/models_peak4.png
     :width: 48 %

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


which gives::

    [[Model]]
        Model(voigt)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 23
        # data points      = 401
        # variables        = 4
        chi-square         = 10.9301767
        reduced chi-square = 0.02753193
        Akaike info crit   = -1436.57602
        Bayesian info crit = -1420.60017
    [[Variables]]
        sigma:      0.89518909 +/- 0.01415450 (1.58%) (init = 0.8775)
        amplitude:  34.1914737 +/- 0.17946860 (0.52%) (init = 65.43358)
        center:     9.24374847 +/- 0.00441903 (0.05%) (init = 9.25)
        gamma:      0.52540199 +/- 0.01857955 (3.54%) (init = 0.7)
        fwhm:       3.22385342 +/- 0.05097475 (1.58%) == '3.6013100*sigma'
        height:     10.0872204 +/- 0.03482129 (0.35%) == 'amplitude*wofz((1j*gamma)/(sigma*sqrt(2))).real/(sigma*sqrt(2*pi))'
    [[Correlations]] (unreported correlations are < 0.250)
        C(sigma, gamma)     = -0.928
        C(amplitude, gamma) =  0.821
        C(sigma, amplitude) = -0.651

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

.. literalinclude:: ../examples/doc_builtinmodels_stepmodel.py

After constructing step-like data, we first create a :class:`StepModel`
telling it to use the ``erf`` form (see details above), and a
:class:`ConstantModel`.  We set initial values, in one case using the data
and :meth:`guess` method for the initial step function paramaters, and
:meth:`make_params` arguments for the linear component.
After making a composite model, we run :meth:`fit` and report the
results, which gives::

    [[Model]]
        (Model(step, prefix='step_', form='erf') + Model(linear, prefix='line_'))
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 49
        # data points      = 201
        # variables        = 5
        chi-square         = 593.709622
        reduced chi-square = 3.02913072
        Akaike info crit   = 227.700173
        Bayesian info crit = 244.216698
    [[Variables]]
        line_intercept:  12.0964833 +/- 0.27606235 (2.28%) (init = 11.58574)
        line_slope:      1.87164655 +/- 0.09318714 (4.98%) (init = 0)
        step_sigma:      0.67392841 +/- 0.01091168 (1.62%) (init = 1.428571)
        step_center:     3.13494792 +/- 0.00516615 (0.16%) (init = 2.5)
        step_amplitude:  112.858376 +/- 0.65392949 (0.58%) (init = 134.7378)
    [[Correlations]] (unreported correlations are < 0.100)
        C(line_slope, step_amplitude)     = -0.879
        C(step_sigma, step_amplitude)     =  0.564
        C(line_slope, step_sigma)         = -0.457
        C(line_intercept, step_center)    =  0.427
        C(line_intercept, line_slope)     = -0.309
        C(line_slope, step_center)        = -0.234
        C(line_intercept, step_sigma)     = -0.137
        C(line_intercept, step_amplitude) = -0.117
        C(step_center, step_amplitude)    =  0.109

with a plot of

.. image::  _images/models_stepfit.png
   :target: _images/models_stepfit.png
   :width: 50 %


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

.. literalinclude:: ../examples/doc_builtinmodels_nistgauss.py

where we give a separate prefix to each model (they all have an
``amplitude`` parameter).  The ``prefix`` values are attached transparently
to the models.

Note that the calls to :meth:`make_param` used the bare name, without the
prefix.  We could have used the prefixes, but because we used the
individual model ``gauss1`` and ``gauss2``, there was no need.

Note also in the example here that we explicitly set bounds on many of the
parameter values.

The fit results printed out are::

    [[Model]]
        ((Model(gaussian, prefix='g1_') + Model(gaussian, prefix='g2_')) + Model(exponential, prefix='exp_'))
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 48
        # data points      = 250
        # variables        = 8
        chi-square         = 1247.52821
        reduced chi-square = 5.15507524
        Akaike info crit   = 417.864631
        Bayesian info crit = 446.036318
    [[Variables]]
        exp_decay:      90.9508860 +/- 1.10310509 (1.21%) (init = 93.24905)
        exp_amplitude:  99.0183283 +/- 0.53748735 (0.54%) (init = 162.2102)
        g1_center:      107.030954 +/- 0.15006786 (0.14%) (init = 105)
        g1_sigma:       16.6725753 +/- 0.16048161 (0.96%) (init = 15)
        g1_amplitude:   4257.77319 +/- 42.3833645 (1.00%) (init = 2000)
        g1_fwhm:        39.2609138 +/- 0.37790530 (0.96%) == '2.3548200*g1_sigma'
        g1_height:      101.880231 +/- 0.59217100 (0.58%) == '0.3989423*g1_amplitude/max(1.e-15, g1_sigma)'
        g2_center:      153.270101 +/- 0.19466743 (0.13%) (init = 155)
        g2_sigma:       13.8069484 +/- 0.18679415 (1.35%) (init = 15)
        g2_amplitude:   2493.41771 +/- 36.1694731 (1.45%) (init = 2000)
        g2_fwhm:        32.5128783 +/- 0.43986659 (1.35%) == '2.3548200*g2_sigma'
        g2_height:      72.0455934 +/- 0.61722094 (0.86%) == '0.3989423*g2_amplitude/max(1.e-15, g2_sigma)'
    [[Correlations]] (unreported correlations are < 0.500)
        C(g1_sigma, g1_amplitude)   =  0.824
        C(g2_sigma, g2_amplitude)   =  0.815
        C(exp_decay, exp_amplitude) = -0.695
        C(g1_sigma, g2_center)      =  0.684
        C(g1_center, g2_amplitude)  = -0.669
        C(g1_center, g2_sigma)      = -0.652
        C(g1_amplitude, g2_center)  =  0.648
        C(g1_center, g2_center)     =  0.621
        C(g1_center, g1_sigma)      =  0.507
        C(exp_decay, g1_amplitude)  = -0.507

We get a very good fit to this problem (described at the NIST site as of
average difficulty, but the tests there are generally deliberately challenging) by
applying reasonable initial guesses and putting modest but explicit bounds
on the parameter values.  This fit is shown on the left:

.. _figA3:

  .. image::  _images/models_nistgauss.png
     :target: _images/models_nistgauss.png
     :width: 48 %
  .. image::  _images/models_nistgauss2.png
     :target: _images/models_nistgauss2.png
     :width: 48 %


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

we can get a better initial estimate.  The fit converges to the same answer,
giving to identical values (to the precision printed out in the report),
but in few steps, and without any bounds on parameters at all::

    [[Model]]
        ((Model(gaussian, prefix='g1_') + Model(gaussian, prefix='g2_')) + Model(exponential, prefix='exp_'))
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 39
        # data points      = 250
        # variables        = 8
        chi-square         = 1247.52821
        reduced chi-square = 5.15507524
        Akaike info crit   = 417.864631
        Bayesian info crit = 446.036318
    [[Variables]]
        exp_decay:      90.9508890 +/- 1.10310483 (1.21%) (init = 111.1985)
        exp_amplitude:  99.0183270 +/- 0.53748905 (0.54%) (init = 94.53724)
        g1_sigma:       16.6725765 +/- 0.16048227 (0.96%) (init = 14.5)
        g1_amplitude:   4257.77343 +/- 42.3836432 (1.00%) (init = 3189.648)
        g1_center:      107.030956 +/- 0.15006873 (0.14%) (init = 106.5)
        g1_fwhm:        39.2609166 +/- 0.37790686 (0.96%) == '2.3548200*g1_sigma'
        g1_height:      101.880230 +/- 0.59217233 (0.58%) == '0.3989423*g1_amplitude/max(1.e-15, g1_sigma)'
        g2_sigma:       13.8069461 +/- 0.18679534 (1.35%) (init = 15)
        g2_amplitude:   2493.41733 +/- 36.1696911 (1.45%) (init = 2818.337)
        g2_center:      153.270101 +/- 0.19466905 (0.13%) (init = 150)
        g2_fwhm:        32.5128728 +/- 0.43986940 (1.35%) == '2.3548200*g2_sigma'
        g2_height:      72.0455948 +/- 0.61722329 (0.86%) == '0.3989423*g2_amplitude/max(1.e-15, g2_sigma)'
    [[Correlations]] (unreported correlations are < 0.500)
        C(g1_sigma, g1_amplitude)   =  0.824
        C(g2_sigma, g2_amplitude)   =  0.815
        C(exp_decay, exp_amplitude) = -0.695
        C(g1_sigma, g2_center)      =  0.684
        C(g1_center, g2_amplitude)  = -0.669
        C(g1_center, g2_sigma)      = -0.652
        C(g1_amplitude, g2_center)  =  0.648
        C(g1_center, g2_center)     =  0.621
        C(g1_sigma, g1_center)      =  0.507
        C(exp_decay, g1_amplitude)  = -0.507

This script is in the file ``doc_builtinmodels_nistgauss2.py`` in the examples folder,
and the fit result shown on the right above shows an improved initial
estimate of the data.
