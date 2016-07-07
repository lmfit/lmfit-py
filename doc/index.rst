.. lmfit documentation master file,

Non-Linear Least-Square Minimization and Curve-Fitting for Python
===========================================================================

.. _Levenberg-Marquardt:     http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
.. _MINPACK-1:               http://en.wikipedia.org/wiki/MINPACK


.. warning::

  Upgrading scripts from version 0.8.3 to 0.9.0?  See  :ref:`whatsnew_090_label`

.. warning::

  Support for Python 2.6 and scipy 0.13 will be dropped with version 0.9.5.


Lmfit provides a high-level interface to non-linear optimization and curve
fitting problems for Python. Lmfit builds on and extends many of the
optimization algorithm of :mod:`scipy.optimize`, especially the
`Levenberg-Marquardt`_ method from :scipydoc:`optimize.leastsq`.

Lmfit provides a number of useful enhancements to optimization and data
fitting problems, including:

  * Using :class:`Parameter` objects instead of plain floats as variables.
    A :class:`Parameter` has a value that can be varied in the fit, have a
    fixed value, or have upper and/or lower bounds.  A Parameter can even
    have a value that is constrained by an algebraic expression of other
    Parameter values.

  * Ease of changing fitting algorithms.  Once a fitting model is set up,
    one can change the fitting algorithm used to find the optimal solution
    without changing the objective function.

  * Improved estimation of confidence intervals.  While
    :scipydoc:`optimize.leastsq` will automatically calculate
    uncertainties and correlations from the covariance matrix, the accuracy
    of these estimates are often questionable.  To help address this, lmfit
    has functions to explicitly explore parameter space to determine
    confidence levels even for the most difficult cases.

  * Improved curve-fitting with the :class:`Model` class.  This
    extends the capabilities of :scipydoc:`optimize.curve_fit`, allowing
    you to turn a function that models for your data into a python class
    that helps you parametrize and fit data with that model.

  * Many :ref:`pre-built models <builtin_models_chapter>` for common
    lineshapes are included and ready to use.

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py

The lmfit package is Free software, using an MIT license.  The software and
this document are works in progress.  If you are interested in
participating in this effort please use the `lmfit github repository`_.


.. toctree::
   :maxdepth: 2

   intro
   installation
   support
   faq
   parameters
   fitting
   model
   builtin_models
   confidence
   bounds
   constraints
