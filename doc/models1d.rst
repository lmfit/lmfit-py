.. _models1d-label:

=================================
Simple Builtin Fitting Models
=================================

It is common to want to fit some 1-dimensional data set to a simple
peak or line shape, such as Gaussians, Lorentzian, and Voigt peaks,
Exponential decays, and so on.  These are used in a wide range of
spectroscopic techniques as well as in basic mathematical analysis.
In an effort to make make simple things truly simple, the lmfit
provides a few simple wrappers for doing such fits in its `models1d`
module.


Example
===========

Let's start with a very simple example.  We'll read data from a simple
datafile, and fit it to a Gaussian peak.  A script to do this could be:

.. literalinclude:: ../tests/model1d_doc1.py

First, we read in the data for 'x' and 'y', then build a Gaussian
model.  This 'model' contains all the Parameters for a Gaussian line
shape.  We then tell the model to make initial guesses for the
Parameters based on the data arrays.  At this point, we save the
predicted data with the initial Parameter values.  We then perform the
actual fit, and print out and display the results.  The printed output
will be (approximately)::

  [[Variables]]
       amplitude:     12.558540 +/- 0.160655 (1.28%) initial =  27.274340
       center:        5.658661 +/- 0.010305 (0.18%) initial =  5.500000
       sigma:         0.493317 +/- 0.007288 (1.48%) initial =  1.666667
  [[Correlations]] (unreported correlations are <  0.100)
      C(amplitude, sigma)          =  0.577

and the resulting plot will look like:

.. image:: models1d_doc1.png
   :width: 85%

which shows a good fit (the data were simulated).

classes in the :mod:`models1d` module
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


