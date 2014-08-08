.. _models1d-label:

=================================
Simple Builtin Fitting Models
=================================

It is common to want to fit some data to one of a well-known functional
form such as Gaussians, Lorentzian, and Voigt peaks, Exponential decays,
and so on.  These are used in a wide range of scientific domains and
spectroscopic techniques as well as in basic mathematical analysis.  In an
effort to make make simple things truly simple, the lmfit package provides
canonical definitions for many known lineshapes.  and a few pre-defined
high-level fitting models in the :mod:`models` module. In addition, there
is a :class:`model` class for building high-level fitting models from other
functions.  Here, we outline the existing models, and describe how to build
your own.


Example 1: Fit data to Gaussian profile
===========================================

Let's start with a simple and common example of fitting data to a Gaussian
peak.  Of course, we could define a model Gaussian function, define the
Parameters that go into and build a residual function.  But since many
people are likely to want to do such fits, it's useful xto set up such a
model once, especially if done in such away as to be easy to extend.

The :class:`GaussianModel` class provides a model function for a Gaussian
profile, the set of Parameters needed for this model.  It also has built-in
functions for guessing starting values for the Parameters based on some
data, and fitting the model to a set of data.  This will give a very simple
interface to fitting data to this well-known function.  Here's a script to
do this (included in the ``examples`` folder with the source code):


.. literalinclude:: ../examples/models_doc1.py

After some imports, we read in the data for ``x`` and ``y`` from a text
file. We then create a Gaussian model.  This will automatically contains
all the Parameters for a Gaussian line shape -- it has parameters named
``amplitude``, ``center``, and ``sigma``.  We then tell this model to guess
initial starting values for these parameters based on the data arrays.  We
then run the :meth:`fit` method of the model, and print out the results,
which will look like this::

    [[Variables]]
         amplitude:     8.880221 +/- 0.1135958 (1.28%) initial =  10.90974
         center:        5.658661 +/- 0.01030511 (0.18%) initial =  5.7
         fwhm:          1.642852 +/- 0.02426668 (1.48%) == '2.354820*sigma'
         sigma:         0.6976551 +/- 0.01030511 (1.48%) initial =  0.8
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.577

You can see here that the model created Parameters named ``amplitude``,
``center``, and ``sigma`` for the Gaussian model, and a parameter named
``fwhm``, constrained with ``sigma`` to report Full Width at Half Maximum.
Finally, we display the data, best fit and initial guess graphically --
note that both the initial and best fit are preserved in the ``result``
returned by :meth:`fit` method.

.. image:: _images/models_doc1.png
   :target: _images/models_doc1.png
   :width: 85%

which shows the data in blue dots, the best fit as a solid red line, and
the initial fit in black dashed line.  You can also see from the results
that the starting guess were a pretty good estimate for this simple data
set.

We emphasize here that the fit to this pre-built model really took 3
lines of code::

    model = GaussianModel()
    model.guess_starting_values(y, x=x)
    result = model.fit(y, x=x)

Of course, some models are necessarily mode complicated.


Example 2: Fit data to Gaussian profile + Line
=================================================

We can expand on the model by showing an important feature of the lmfit
Models derived from the powerful :class:`Model` class: you can add them
together.  That is, to fit data to a Gaussian plus a linear offset, we
could use this script (also included in the ``examples`` folder with the
source code):

.. literalinclude:: ../examples/models_doc2.py


This is only slightly more complicated than the script above.  Here, we
start with a :class:`GaussianModel` as before and use the built-in method
to guess starting values.  But then we create a :class:`LinearModel` (which
has parameters named ``slope`` and ``intercept``), and add this to the
:class:`GaussianModel` with the simple::

    total = gauss + line

and call the :meth:`fit` method of the combined model ``total``.  That will
fit all the parameters, reporting results of::

    [[Variables]]
         amplitude:     8.459308 +/- 0.1241455 (1.47%) initial =  11.96192
         center:        5.655479 +/- 0.009176806 (0.16%) initial =  5.7
         fwhm:          1.590575 +/- 0.02335249 (1.47%) == '2.354820*sigma'
         intercept:    -2.968602 +/- 0.03352202 (1.13%) initial = -1
         sigma:         0.6754549 +/- 0.009916889 (1.47%) initial =  0.9
         slope:         0.1148441 +/- 0.005748924 (5.01%) initial =  0
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.666

and give a plot like this:

.. image:: _images/models_doc2.png
   :target: _images/models_doc2.png
   :width: 85%

again showing (simulated) data shown in blue dots, with the best fit as a
solid red line, and the initial fit in black dashed line.

The emphasis here is that not only is fitting to a single pre-defined
function a simple matter, but that fitting to a model built up of several
pre-defined functions is not much more difficult.


The :class:`Model` class
=======================================

The :class:`Model` class is the most general way to wrap a pre-defined
function as a fitting model.  All the models described in this chapter are
derived from it.


.. class:: Model(func[, independent_vars=None[, param_names=None[, missing=None[, prefix=''[, components=None]]]]])

        Create a model based on the user-supplied function.  This uses a
        fair amount of introspection, automatically converting argument
        names to Parameter names.

        :param func: function to be wrapped
	:type func: callable
	:param independent_vars: list of argument names to ``func`` that are independent variables.
	:type independent_vars: ``None`` (default) or list of strings.
        :param param_names: list of argument names to ``func`` that should be made into Parameters.
        :type param_names: ``None`` (default) or list of strings
	:param missing: how to handle missing values.
	:type missing: one of ``None`` (default), 'drop', or 'raise'
	:param prefix: prefix to add to all parameter names to distinguish components.
	:type prefix: string
	:param components: list of model components for a composite fit (usually handled internally).
	:type components: ``None`` or default.


        Parameter names are inferred from the function arguments,
        and a residual function is automatically constructed.

        Example
        -------
        >>> def decay(t, tau, N):
        ...     return N*np.exp(-t/tau)
        ...
        >>> my_model = Model(decay, independent_vars=['t'])

            None: Do not check for null or missing values (default)
            'drop': Drop null or missing observations in data.
                if pandas is installed, pandas.isnull is used, otherwise
                numpy.isnan is used.
            'raise': Raise a (more helpful) exception when data contains null
                or missing values.


Available :class:`Model` subclasses in the :mod:`models` module
====================================================================

Several fitting models are pre-built and available in the :mod:`models` module.


.. class:: GaussianModel()

.. class:: LorentzianModel()

.. class:: VoigtModel()

.. class:: ExponentialModel()


.. class:: ExponentialModel()

.. class:: StepModel()


.. class:: RectangleModel()



Building a  :class:`Model` from your own function
=============================================================
