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


Example
===========

Let's start with a very simple example.  We'll read data from a text data
file, and fit it to a Gaussian peak.  A script to do this (included in the
``examples`` folder with the source code might look like this:

.. literalinclude:: ../examples/models_doc1.py

First, we read in the data for 'x' and 'y', then build a Gaussian model.
This `model` automatically contains all the Parameters for a Gaussian line
shape -- it has parameters named ``amplitude``, ``center``, and ``sigma``.
We then expliticly tell the model to guess initial starting values for
these parameters based on the data arrays.  We then , and save the model
predicted with these initial Parameter values.  We then perform a fit, and
print out the results, and display the data, best-fit and initial guess
graphically.  The printed output will be (approximately)::

    [[Variables]]
         amplitude:     8.880221 +/- 0.1135958 (1.28%) initial =  10.90974
         center:        5.658661 +/- 0.01030511 (0.18%) initial =  5.7
         fwhm:          1.642852 +/- 0.02426668 (1.48%) == '2.354820*sigma'
         sigma:         0.6976551 +/- 0.01030511 (1.48%) initial =  0.8
    [[Correlations]] (unreported correlations are <  0.250)
        C(amplitude, sigma)          =  0.577

and the resulting plot will look like:

.. image:: _images/models1d_doc1.png
   :target: _images/models1d_doc1.png
   :width: 85%

which shows a good fit (the data were simulated, and shown in blue dots,
with the best fit as a solid red line, and the initial fit in black dashed
line).

You can see here that the model created Parameters named ``amplitude``,
``center``, and ``sigma`` for the Gaussian model, and a parameter named
``fwhm``, constrained with ``sigma`` to report Full Width at Half Maximum.
You can also see from the results that the starting guess were a pretty
good estimate for this simple data set.


classes in the :mod:`models` module
=======================================

Several fitting models are available

.. class:: GaussianModel()

.. class:: LorentzianModel()

.. class:: VoigtModel()

.. class:: PeakModel()


.. class:: ExponentialModel()


.. class:: ExponentialModel()

.. class:: StepModel()


.. class:: RectangleModel()


