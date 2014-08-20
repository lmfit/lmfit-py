.. _builtin_models_chapter:

=================================================
Built-in Fitting Models in the :mod:`models`
=================================================

Lmfit provides several builtin fitting models in the :mod:`models` module.
These pre-defined models each subclass from the :class:`Model` class of the
previous chapter and wrap relatively well-known functional forms, such as
Gaussians, Lorentzian, and Exponentials that are used in a wide range of
scientific domains.  In fact, all the models are all based on simple, plain
python functions defined in the :mod:`lineshapes` module.  In addition to
wrapping a function into a :class:`Model`, these models also provide a
:meth:`guess_starting_values` method that is intended to give a reasonable
set of starting values from a data array that closely approximates the
data to be fit.

.. module:: models

As shown in the previous chapter, a key feature of the :class:`Model` class
is that models can easily be combined to give a composite
:class:`Model`. Thus while some of the models listed here may seem pretty
trivial (notably, :class:`ConstantModel` and :class:`LinearModel`), the
main point of having these is to be able to used in composite models.  For
example,  a Lorentzian plus a linear background might be represented as::

    >>> from lmfit.models import LinearModel, LorentzianModel
    >>> peak = LorentzianModel()
    >>> background  = LinearModel()
    >>> model = peak + background




All the models listed below are one dimensional, with an independent
variable named ``x``.  Many of these models represent a function with a
distinct peak, and so share common features.  To maintain uniformity,
common parameter names are used whenever possible.  Thus, most models have
a parameter called ``amplitude`` that represents the overall height (or
area of) a peak or function, a ``center`` parameter that represents a peak
centroid position, and a ``sigma`` parameter that gives a characteristic
width.   Some peak shapes also have a parameter ``fwhm``, typically
constrained by ``sigma`` to give the full width at half maximum.

After a list of builtin models, a few examples of their use is given.

Peak-like models
-------------------

There are many peak-like models available.  These include
:class:`GaussianModel`, :class:`LorentzianModel`, :class:`VoigtModel` and
some less commonly used variations.

:class:`GaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. class:: GaussianModel()

A model based on a `Gaussian or normal distribution lineshape
<http://en.wikipedia.org/wiki/Normal_distribution>`_.  Parameter names:
``amplitude``, ``center``, and ``sigma``.  In addition, a constrained
parameter ``fwhm`` is included.

.. math::

  f(x; A, \mu, \sigma) = \frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}

where the parameter ``amplitude`` corresponds to :math:`A`, ``center`` to
:math:`\mu`, and ``sigma`` to :math:`\sigma`.  The Full-Width at
Half-Maximum is :math:`2\sigma\sqrt{2\ln{2}}`, approximately
:math:`2.3548\sigma`


:class:`LorentzianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: LorentzianModel()

A model based on a `Lorentzian or Cauchy-Lorentz distribution function
<http://en.wikipedia.org/wiki/Cauchy_distribution>`_.  Parameter names:
``amplitude``, ``center``, and ``sigma``.  In addition, a constrained
parameter ``fwhm`` is included.

.. math::

  f(x; A, \mu, \sigma) = \frac{A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]

where the parameter ``amplitude`` corresponds to :math:`A`, ``center`` to
:math:`\mu`, and ``sigma`` to :math:`\sigma`.  The Full-Width at
Half-Maximum is :math:`2\sigma`.


:class:`VoigtModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: VoigtModel()

A model based on a `Voigt distribution function
<http://en.wikipedia.org/wiki/Voigt_profile>`_.  Parameter names:
``amplitude``, ``center``, and ``sigma``.  A ``gamma`` parameter is also
available.  By default, it is constrained to have value equal to ``sigma``,
though this can be varied independently.  In addition, a constrained
parameter ``fwhm`` is included.  The definition for the Voigt function used
here is

.. math::

    f(x; A, \mu, \sigma, \gamma) = \frac{A \textrm{Re}[w(z)]}{\sigma\sqrt{2 \pi}}

where

.. math::
   :nowrap:

   \begin{eqnarray*}
     z &=& \frac{x-\mu +i\gamma}{\sigma\sqrt{2}} \\
     w(z) &=& e^{-z^2}{\operatorname{erfc}}(-iz)
   \end{eqnarray*}

and :func:`erfc` is the complimentary error function.  As above,
``amplitude`` corresponds to :math:`A`, ``center`` to
:math:`\mu`, and ``sigma`` to :math:`\sigma`. The parameter ``gamma``
corresponds  to :math:`\gamma`.
If ``gamma`` is kept at the default value (constrained to ``sigma``),
the full width at half maximum is approximately :math:`3.6013\sigma`.


:class:`PseudoVoigtModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: PseudoVoigtModel()

a model based on a `pseudo-Voigt distribution function
<http://en.wikipedia.org/wiki/Voigt_profile#Pseudo-Voigt_Approximation>`_,
which is a weighted sum of a Gaussian and Lorentzian distribution functions
with the same values for ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`)
and ``sigma`` (:math:`\sigma`), and a parameter ``fraction`` (:math:`\alpha`)
in

.. math::

  f(x; A, \mu, \sigma, \alpha) = (1-\alpha)\frac{A}{\pi}
  \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big] + \frac{\alpha A}{\pi} \big[\frac{\sigma}{(x - \mu)^2 + \sigma^2}\big]


The :meth:`guess_starting_values` function always gives a starting
value for ``fraction`` of 0.5

:class:`Pearson7Model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: Pearson7Model()

A model based on a `Pearson VII distribution
<http://en.wikipedia.org/wiki/Pearson_distribution#The_Pearson_type_VII_distribution>`_.
This is another Voigt-like distribution function.  It has the usual
parameters ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and
``sigma`` (:math:`\sigma`), and also ``exponent`` (:math:`p`) in

.. math::

    f(x; A, \mu, \sigma, p) = \frac{sA}{\big\{[1 + (\frac{x-\mu}{\sigma})^2] (2^{1/p} -1)  \big\}^p}

where

.. math::

    s = \frac{\Gamma(p) \sqrt{2^{1/p} -1}}{ \sigma\sqrt{\pi}\,\Gamma(p-1/2)}

where :math:`\Gamma(x)` is the gamma function.

The :meth:`guess_starting_values` function always gives a starting
value for ``exponent`` of 0.5.

:class:`StudentsTModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: StudentsTModel()

A model based on a `Student's t distribution function
<http://en.wikipedia.org/wiki/Student%27s_t-distribution>`_, with the usual
parameters ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and
``sigma`` (:math:`\sigma`) in

.. math::

    f(x; A, \mu, \sigma) = \frac{A \Gamma(\frac{\sigma+1}{2})} {\sqrt{\sigma\pi}\,\Gamma(\frac{\sigma}{2})} \Bigl[1+\frac{(x-\mu)^2}{\sigma}\Bigr]^{-\frac{\sigma+1}{2}}


where :math:`\Gamma(x)` is the gamma function.


:class:`BreitWignerModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: BreitWignerModel()

A model based on a `Breit-Wigner-Fano function
<http://en.wikipedia.org/wiki/Fano_resonance>`_.  It has the usual
parameters ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and
``sigma`` (:math:`\sigma`), plus ``q`` (:math:`q`) in

.. math::

    f(x; A, \mu, \sigma, q) = \frac{A (q\sigma/2 + x - \mu)^2}{(\sigma/2)^2 + (x - \mu)^2}


:class:`LognormalModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: LognormalModel()

A model based on the `Log-normal distribution function
<http://en.wikipedia.org/wiki/Lognormal>`_.
It has the usual parameters
``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and ``sigma``
(:math:`\sigma`) in

.. math::

    f(x; A, \mu, \sigma) = \frac{A e^{-(\ln(x) - \mu)/ 2\sigma^2}}{x}


:class:`DampedOcsillatorModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: DampedOcsillatorModel()

A model based on the `Damped Harmonic Oscillator Amplitude
<http://en.wikipedia.org/wiki/Harmonic_oscillator#Amplitude_part>`_.
It has the usual parameters ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and
``sigma`` (:math:`\sigma`) in

.. math::

    f(x; A, \mu, \sigma) = \frac{A}{\sqrt{ [1 - (x/\mu)^2]^2 + (2\sigma x/\mu)^2}}


:class:`ExponentialGaussianModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ExponentialGaussianModel()

A model of an `Exponentially modified Gaussian distribution
<http://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution>`_.
It has the usual parameters ``amplitude`` (:math:`A`), ``center`` (:math:`\mu`) and
``sigma`` (:math:`\sigma`), and also ``gamma`` (:math:`\gamma`) in

.. math::

    f(x; A, \mu, \sigma, \gamma) = \frac{A\gamma}{2}
    \exp\bigl[\gamma({\mu - x  + \sigma^2/2})\bigr]
    {\operatorname{erfc}}\bigl[\frac{\mu + \gamma\sigma^2 - x}{\sqrt{2}\sigma}\bigr]


where :func:`erfc` is the complimentary error function.


:class:`DonaichModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: DonaichModel()

A model of an `Doniach Sunjic asymmetric lineshape
<http://www.casaxps.com/help_manual/line_shapes.htm>`_, used in
photo-emission. With the usual parameters ``amplitude`` (:math:`A`),
``center`` (:math:`\mu`) and ``sigma`` (:math:`\sigma`), and also ``gamma``
(:math:`\gamma`) in

.. math::

    f(x; A, \mu, \sigma, \gamma) = A\frac{\cos\bigl[\pi\gamma/2 + (1-\gamma)
    \arctan{(x - \mu)}/\sigma\bigr]} {\bigr[1 + (x-\mu)/\sigma\bigl]^{(1-\gamma)/2}}


Linear and Polynomial Models
------------------------------------

These models correspond to polynomials of some degree.  Of course, lmfit is
a very inefficient way to do linear regression (see :func:`numpy.polyfit`
or :func:`scipy.stats.linregress`), but these models may be useful as one
of many components of composite model.

:class:`ConstantModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ConstantModel()

   a class that consists of a single value, ``c``.  This is constant in the
   sense of having no dependence on the independent variable ``x``, not in
   the sense of being non-varying.  To be clear, ``c`` will be a variable
   Parameter.

:class:`LinearModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: LinearModel()

   a class that gives a linear model:

.. math::

    f(x; m, b) = m x + b

with parameters ``slope`` for :math:`m` and  ``intercept`` for :math:`b`.


:class:`QuadraticModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: QuadraticModel()


   a class that gives a quadratic model:

.. math::

    f(x; a, b, c) = a x^2 + b x + c

with parameters ``a``, ``b``, and ``c``.


:class:`ParabolicModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ParabolicModel()

   same as :class:`QuadraticModel`.


:class:`PolynomialModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. class:: PolynomialModel(degree)

   a class that gives a polynomial model up to ``degree`` (with maximum
   value of 7).

.. math::

    f(x; c_0, c_1, \ldots, c_7) = \sum_{i=0, 7} c_i  x^i

with parameters ``c0``, ``c1``, ..., ``c7``.  The supplied ``degree``
will specify how many of these are actual variable parameters.



Step-like models
-----------------------------------------------


:class:`StepModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: StepModel(form='linear')

A model based on a Step function, with four choices for functional form.
The step function starts with a value 0, and ends with a value of :math:`A`
(``amplitude``), rising to :math:`A/2` at :math:`\mu` (``center``),
with :math:`\sigma` (``sigma``) setting the characteristic width. The
supported functional forms are ``linear`` (the default), ``atan`` or
``arctan`` for an arc-tangent function,  ``erf`` for an error function, or
``logistic`` for a `logistic function <http://en.wikipedia.org/wiki/Logistic_function>`_.
The forms are

.. math::
   :nowrap:

   \begin{eqnarray*}
   & f(x; A, \mu, \sigma, {\mathrm{form={}'linear{}'}})  & = A \min{[1, \max{(0,  \alpha)}]} \\
   & f(x; A, \mu, \sigma, {\mathrm{form={}'arctan{}'}})  & = A [1/2 + \arctan{(\alpha)}/{\pi}] \\
   & f(x; A, \mu, \sigma, {\mathrm{form={}'erf{}'}})     & = A [1 + {\operatorname{erf}}(\alpha)]/2 \\
   & f(x; A, \mu, \sigma, {\mathrm{form={}'logistic{}'}})& = A [1 - \frac{1}{1 +  e^{\alpha}} ]
   \end{eqnarray*}

where :math:`\alpha  = (x - \mu)/{\sigma}`.

:class:`RectangleModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. class:: RectangleModel(form='linear')

A model based on a Step-up and Step-down function of the same form.  The
same choices for functional form as for :class:`StepModel` are supported,
with ``linear`` as the default.  The function starts with a value 0, and
ends with a value of :math:`A` (``amplitude``), rising to :math:`A/2` at
:math:`\mu_1` (``center1``), with :math:`\sigma_1` (``sigma1``) setting the
characteristic width.  It drops to rising to :math:`A/2` at :math:`\mu_2`
(``center2``), with characteristic width :math:`\sigma_2` (``sigma2``).

.. math::
   :nowrap:

   \begin{eqnarray*}
   &f(x; A, \mu, \sigma, {\mathrm{form={}'linear{}'}})   &= A \{ \min{[1, \max{(0, \alpha_1)}]} + \min{[-1, \max{(0,  \alpha_2)}]} \} \\
   &f(x; A, \mu, \sigma, {\mathrm{form={}'arctan{}'}})   &= A [\arctan{(\alpha_1)} + \arctan{(\alpha_2)}]/{\pi} \\
   &f(x; A, \mu, \sigma, {\mathrm{form={}'erf{}'}})      &= A [{\operatorname{erf}}(\alpha_1) + {\operatorname{erf}}(\alpha_2)]/2 \\
   &f(x; A, \mu, \sigma, {\mathrm{form={}'logistic{}'}}) &= A [1 - \frac{1}{1 + e^{\alpha_1}} - \frac{1}{1 +  e^{\alpha_2}} ]
   \end{eqnarray*}


where :math:`\alpha_1  = (x - \mu_1)/{\sigma_1}` and :math:`\alpha_2  = -(x - \mu_2)/{\sigma_2}`.


Exponential and Power law models
-----------------------------------------------

:class:`ExponentialModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: ExponentialModel()

A model based on an `exponential decay function
<http://en.wikipedia.org/wiki/Exponential_decay>`_. With parameters named
``amplitude`` (:math:`A`), and ``decay`` (:math:`\tau`), this has the form:

.. math::

   f(x; A, \tau) = A e^{-x/\tau}


:class:`PowerLawModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: PowerLawModel()

A model based on a `Power Law <http://en.wikipedia.org/wiki/Power_law>`_.
With parameters
named ``amplitude`` (:math:`A`), and ``exponent`` (:math:`k`), this has the
form:

.. math::

   f(x; A, k) = A x^k



Example 1: Fit Peaked data to Gaussian or Voigt profiles
------------------------------------------------------------------

Here, we will fit data to two similar lineshapes, in order to decide which
might be the better model.  We will start with a Gaussian profile, as in
the previous chapter, but use the built-in :class:`GaussianModel` instead
of one we write ourselves.  This is a slightly different version from the
one in previous example in that the parameter names are different, and have
built-in default values.  So, we'll simply use::

    from numpy import loadtxt
    from lmfit.models import GaussianModel

    data = loadtxt('test_peak.dat')
    x = data[:, 0]
    y = data[:, 1]

    mod = GaussianModel()
    mod.guess_starting_values(y, x=x)
    out  = mod.fit(y, x=x)
    print(mod.fit_report(min_correl=0.25))

which prints out the results::

    [[Fit Statistics]]
        # function evals   = 25
        # data points      = 401
        # variables        = 3
        chi-square         = 29.994
        reduced chi-square = 0.075
    [[Variables]]
        amplitude:     30.31352 +/- 0.1571252 (0.52%) initial =  21.54192
        center:        9.242771 +/- 0.00737481 (0.08%) initial =  9.25
        fwhm:          2.901562 +/- 0.01736635 (0.60%) == '2.354820*sigma'
        sigma:         1.23218 +/- 0.00737481 (0.60%) initial =  1.35
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.577

We see a few interesting differences from the results of the previous
chapter.  First, the parameter names are longer.  Second, there is a
``fwhm``, defined as :math:`\sim 2.355\sigma`.  And third, the automated
initial guesses are pretty good.  A plot of the fit shows not such a great
fit:

.. _figA1:

  .. image::  _images/models_peak1.png
     :target: _images/models_peak1.png
     :width: 48 %
  .. image::  _images/models_peak2.png
     :target: _images/models_peak2.png
     :width: 48 %

  Fit to peak with Gaussian (left) and Lorentzian (right) models.

suggesting that a different peak shape, with longer tails, should be used.
Perhaps a Lorentzian would be better?  To do this, we simply replace
``GaussianModel`` with ``LorentzianModel`` to get a
:class:`LorentzianModel`::

    from lmfit.models import LorentzianModel
    mod = LorentzianModel()
    mod.guess_starting_values(y, x=x)
    out  = mod.fit(y, x=x)
    print(mod.fit_report(min_correl=0.25))

The results, or course, are worse::

    [[Fit Statistics]]
        # function evals   = 29
        # data points      = 401
        # variables        = 3
        chi-square         = 53.754
        reduced chi-square = 0.135
    [[Variables]]
        amplitude:     38.97278 +/- 0.3138612 (0.81%) initial =  21.54192
        center:        9.244389 +/- 0.009276152 (0.10%) initial =  9.25
        fwhm:          2.30968 +/- 0.02631297 (1.14%) == '2.0000000*sigma'
        sigma:         1.15484 +/- 0.01315648 (1.14%) initial =  1.35
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.709


with the plot shown in the figure above.

A Voigt model does a better job.  Using :class:`VoigtModel`, this is
as simple as::

    from lmfit.models import LorentzianModel
    mod = LorentzianModel()
    mod.guess_starting_values(y, x=x)
    out  = mod.fit(y, x=x)
    print(mod.fit_report(min_correl=0.25))

which gives::

    [[Fit Statistics]]
        # function evals   = 30
        # data points      = 401
        # variables        = 3
        chi-square         = 14.545
        reduced chi-square = 0.037
    [[Variables]]
        amplitude:     35.75536 +/- 0.1386167 (0.39%) initial =  21.54192
        center:        9.244111 +/- 0.005055079 (0.05%) initial =  9.25
        fwhm:          2.629512 +/- 0.01326999 (0.50%) == '3.6013100*sigma'
        gamma:         0.7301542 +/- 0.003684769 (0.50%) == 'sigma'
        sigma:         0.7301542 +/- 0.003684769 (0.50%) initial =  1.35
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.651

with the much better value for :math:`\chi^2` and the obviously better
match to the data as seen in the figure below (left).

.. _figA2:

  .. image::  _images/models_peak3.png
     :target: _images/models_peak3.png
     :width: 48 %
  .. image::  _images/models_peak4.png
     :target: _images/models_peak4.png
     :width: 48 %

  Fit to peak with Voigt model (left) and Voigt model with ``gamma``
  varying independently of ``sigma`` (right).

The Voigt function has a :math:`\gamma` parameter (``gamma``) that can be
distinct from ``sigma``.  The default behavior used above constrains
``gamma`` to have exactly the same value as ``sigma``.  If we allow these
to vary separately, does the fit improve?  To do this, we have to change
the ``gamma`` parameter from a constrained expression and give it a
starting value::

    mod = VoigtModel()
    mod.guess_starting_values(y, x=x)
    mod.params['gamma'].expr  = None
    mod.params['gamma'].value = 0.7

    out  = mod.fit(y, x=x)
    print(mod.fit_report(min_correl=0.25))

which gives::

    [[Fit Statistics]]
        # function evals   = 32
        # data points      = 401
        # variables        = 4
        chi-square         = 10.930
        reduced chi-square = 0.028
    [[Variables]]
        amplitude:     34.19147 +/- 0.1794683 (0.52%) initial =  21.54192
        center:        9.243748 +/- 0.00441902 (0.05%) initial =  9.25
        fwhm:          3.223856 +/- 0.05097446 (1.58%) == '3.6013100*sigma'
        gamma:         0.5254013 +/- 0.01857953 (3.54%) initial =  0.7
        sigma:         0.8951898 +/- 0.01415442 (1.58%) initial =  1.35
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, gamma)          =  0.821

and the fit shown above (on the right).

Comparing the two fits with the Voigt function, we see that :math:`\chi^2`
is definitely better with a separately varying ``gamma`` parameter.  In
addition, the two values for ``gamma`` and ``sigma`` differ significantly
-- well outside the estimated uncertainties.  Even more compelling, reduced
:math:`\chi^2` is improved even though a fourth variable has been added to
the fit, justifying it as a significant variable in the model.


This example shows how easy it can be to alter and compare fitting models
for simple problems.


Example 2: Fit data to a Composite Model with pre-defined models
------------------------------------------------------------------


Here, we repeat the point made at the end of the last chapter that instances
of :class:`Model` class can be added them together to make a *composite
model*.  But using the large number of built-in models available, this is
very simple.  An example of a simple fit to a noisy step function plus a
constant:

.. literalinclude:: ../examples/doc_stepmodel.py

After constructing step-like data, we first create a :class:`StepModel`
telling it to use the ``erf`` form (see details below), and a
:class:`ConstantModel`.  We set initial values, in one case using the data
and :meth:`guess_starting_values` method, and using the explicit
:meth:`set_paramval` for the initial constant value.    Making a composite
model, we run :meth:`fit` and report the results, which give::

    [[Fit Statistics]]
        # function evals   = 52
        # data points      = 201
        # variables        = 4
        chi-square         = 600.191
        reduced chi-square = 3.047
    [[Variables]]
        amplitude:     111.1106 +/- 0.3122441 (0.28%) initial =  115.3431
        c:             11.31151 +/- 0.2631688 (2.33%) initial =  9.278188
        center:        3.122191 +/- 0.00506929 (0.16%) initial =  5
        sigma:         0.6637199 +/- 0.009799607 (1.48%) initial =  1.428571
    [[Correlations]] (unreported correlations are <  0.100)
        C(c, center)                 =  0.381
        C(amplitude, sigma)          =  0.381

with a plot of

.. image::  _images/models_stepfit.png
   :target: _images/models_stepfit.png
   :width: 50 %


Example 3: Fitting Multiple Peaks -- and using Prefixes
------------------------------------------------------------------

.. _NIST StRD: http://itl.nist.gov/div898/strd/nls/nls_main.shtml

As shown above, many of the models have similar parameter names.  For
composite models, this could lead to a problem of having parameters for
different parts of the model having the same name.  To overcome this, each
:class:`Model` can have a ``prefix`` attribute (normally set to a blank
string) that will be put at the beginning of each parameter name.  To
illustrate, we fit one of the classic datasets from the `NIST StRD`_ suite
involving a decaying exponential and two gaussians.

.. literalinclude:: ../examples/doc_nistgauss.py


where we give a separate prefix to each model (they all have an
``amplitude`` parameter).  The ``prefix`` values are attached transparently
to the models.  Note that the calls to :meth:`set_paramval` used the bare
name, without the prefix.   We could have used them, but because we used
the individual model ``gauss1`` and ``gauss2``, there was no need.  Had we
used the composite model to set the initial parameter values, we would have
needed to, as with::

    ## WRONG
    mod.set_paramval('amplitude', 500, min=10)

    ## Raises KeyError: "'amplitude' not a parameter name"

    ## Correct
    mod.set_paramval('g1_amplitude', 501, min=10)


The fit results printed out are::

    [[Fit Statistics]]
        # function evals   = 66
        # data points      = 250
        # variables        = 8
        chi-square         = 1247.528
        reduced chi-square = 5.155
    [[Variables]]
        exp_amplitude:     99.01833 +/- 0.5374884 (0.54%) initial =  162.2102
        exp_decay:         90.95088 +/- 1.103105 (1.21%) initial =  93.24905
        g1_amplitude:      4257.774 +/- 42.38366 (1.00%) initial =  500
        g1_center:         107.031 +/- 0.1500691 (0.14%) initial =  105
        g1_fwhm:           39.26092 +/- 0.3779083 (0.96%) == '2.354820*g1_sigma'
        g1_sigma:          16.67258 +/- 0.1604829 (0.96%) initial =  12
        g2_amplitude:      2493.417 +/- 36.16923 (1.45%) initial =  500
        g2_center:         153.2701 +/- 0.194667 (0.13%) initial =  150
        g2_fwhm:           32.51287 +/- 0.4398624 (1.35%) == '2.354820*g2_sigma'
        g2_sigma:          13.80695 +/- 0.1867924 (1.35%) initial =  12
    [[Correlations]] (unreported correlations are <  0.100)
        C(g1_amplitude, g1_sigma)    =  0.824
        C(g2_amplitude, g2_sigma)    =  0.815
        C(g1_sigma, g2_center)       =  0.684
        C(g1_amplitude, g2_center)   =  0.648
        C(g1_center, g2_center)      =  0.621
        C(g1_center, g1_sigma)       =  0.507
        C(g1_amplitude, g1_center)   =  0.418
        C(exp_amplitude, g2_amplitude)  =  0.282
        C(exp_amplitude, g2_sigma)   =  0.171
        C(exp_amplitude, g1_amplitude)  =  0.148
        C(exp_decay, g1_center)      =  0.105

We get a very good fit to this challenging problem (described at the NIST
site as of average difficulty, but the tests there are generally hard) by
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
itself, we can see the two Gaussian peaks are reasonably well centered.  We
can simplify the initial parameter values by using this, and by defining an
:func:`index_of` function to limit the data range.  That is, with::

    def index_of(arrval, value):
        "return index of array *at or below* value "
        if value < min(arrval):  return 0
        return max(np.where(arrval<=value)[0])

    ix1 = index_of(x,  75)
    ix2 = index_of(x, 135)
    ix3 = index_of(x, 175)

    exp_mod.guess_starting_values(y[:ix1], x=x[:ix1])
    gauss1.guess_starting_values(y[ix1:ix2], x=x[ix1:ix2])
    gauss2.guess_starting_values(y[ix2:ix3], x=x[ix2:ix3])

we can get a better initial estimate, and the fit converges in fewer steps,
and without any bounds on parameters::

    [[Fit Statistics]]
        # function evals   = 46
        # data points      = 250
        # variables        = 8
        chi-square         = 1247.528
        reduced chi-square = 5.155
    [[Variables]]
        exp_amplitude:     99.01833 +/- 0.5374875 (0.54%) initial =  94.53724
        exp_decay:         90.95089 +/- 1.103105 (1.21%) initial =  111.1985
        g1_amplitude:      4257.773 +/- 42.38338 (1.00%) initial =  2126.432
        g1_center:         107.031 +/- 0.1500679 (0.14%) initial =  106.5
        g1_fwhm:           39.26091 +/- 0.3779053 (0.96%) == '2.354820*g1_sigma'
        g1_sigma:          16.67258 +/- 0.1604816 (0.96%) initial =  14.5
        g2_amplitude:      2493.418 +/- 36.16948 (1.45%) initial =  1878.892
        g2_center:         153.2701 +/- 0.1946675 (0.13%) initial =  150
        g2_fwhm:           32.51288 +/- 0.4398666 (1.35%) == '2.354820*g2_sigma'
        g2_sigma:          13.80695 +/- 0.1867942 (1.35%) initial =  15
    [[Correlations]] (unreported correlations are <  0.500)
        C(g1_amplitude, g1_sigma)    =  0.824
        C(g2_amplitude, g2_sigma)    =  0.815
        C(g1_sigma, g2_center)       =  0.684
        C(g1_amplitude, g2_center)   =  0.648
        C(g1_center, g2_center)      =  0.621
        C(g1_center, g1_sigma)       =  0.507


This example is in the file ``doc_nistgauss2.py`` in the examples folder,
and the fit result shown on the right above.
