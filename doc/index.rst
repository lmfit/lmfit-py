.. lmfit documentation master file,

Non-Linear Least-Squares Minimization and Curve-Fitting for Python
==================================================================

.. _Levenberg-Marquardt:     https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
.. _scipy.optimize:      https://docs.scipy.org/doc/scipy/reference/optimize.html
.. _lmfit GitHub repository:   https://github.com/lmfit/lmfit-py

Lmfit provides a high-level interface to non-linear optimization and curve
fitting problems for Python. It builds on and extends many of the
optimization methods of `scipy.optimize`_. Initially inspired by (and
named for) extending the `Levenberg-Marquardt`_ method from
:scipydoc:`optimize.leastsq`, lmfit now provides a number of useful
enhancements to optimization and data fitting problems, including:

  * Using :class:`~lmfit.parameter.Parameter` objects instead of plain
    floats as variables. A :class:`~lmfit.parameter.Parameter` has a value
    that can be varied during the fit or kept at a fixed value. It can
    have upper and/or lower bounds. A Parameter can even have a value that
    is constrained by an algebraic expression of other Parameter values.
    As a Python object, a Parameter can also have attributes such as a
    standard error, after a fit that can estimate uncertainties.

  * Ease of changing fitting algorithms. Once a fitting model is set up,
    one can change the fitting algorithm used to find the optimal solution
    without changing the objective function.

  * Improved estimation of confidence intervals. While
    :scipydoc:`optimize.leastsq` will automatically calculate
    uncertainties and correlations from the covariance matrix, the accuracy
    of these estimates is sometimes questionable. To help address this,
    lmfit has functions to explicitly explore parameter space and determine
    confidence levels even for the most difficult cases. Additionally, lmfit
    will use the ``numdifftools`` package (if installed) to estimate parameter
    uncertainties and correlations for algorithms that do not natively
    support this in SciPy.

  * Improved curve-fitting with the :class:`~lmfit.model.Model` class. This
    extends the capabilities of :scipydoc:`optimize.curve_fit`, allowing
    you to turn a function that models your data into a Python class
    that helps you parametrize and fit data with that model.

  * Many :ref:`built-in models <builtin_models_chapter>` for common
    lineshapes are included and ready to use.

The lmfit package is Free software, using an Open Source license. The
software and this document are works in progress. If you are interested in
participating in this effort please use the `lmfit GitHub repository`_.

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
   whatsnew
   examples/index
