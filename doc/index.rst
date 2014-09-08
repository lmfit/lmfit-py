.. lmfit documentation master file,

Non-Linear Least-Square Minimization and Curve-Fitting for Python
===========================================================================

.. _Levenberg-Marquardt:     http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
.. _MINPACK-1:               http://en.wikipedia.org/wiki/MINPACK


The lmfit python package provides a simple and flexible interface to
non-linear optimization and curve fitting problems, extending the
capabilities of :mod:`scipy.optimize`.  Initially designed to extend the
the `Levenberg-Marquardt`_ algorithm in :func:`scipy.optimize.leastsq`,
lmfit supports most of the optimization methods from :mod:`scipy.optimize`.
It also provides a simple way to apply this extension to *curve fitting* or
*data modeling* problems.

The key concept in lmfit is the :class:`Parameter`, which is used as the
quantity to be optimized in all minimization problems in placee of plain
floating point value.  A :class:`Parameter` has a value that can be varied
or fixed, have upper and/or lower bounds placed on its value, or have
values that are evaluated from algebraic expressions of other Parameter
values.  Since :class:`Parameters` live outside the core optimization
routines, they can be used in **all** optimization routines from
:mod:`scipy.optimize`.  Furthermore, by using :class:`Parameter` objects
instead of plain variables, the objective function does not have to be
rewritten to reflect every change of what is varied in the fit, or if
constraints are placed on the Parameters.  This simplifies the writing of
models, and gives the user more flexibility in using and testing variations
of that model.

Lmfit supports several optimization methods from :mod:`scipy.optimize`.
The default and best tested optimization method (and the origin of the
name) is the `Levenberg-Marquardt`_ algorithm of
:func:`scipy.optimize.leastsq`.  An important advantage of this method is
that it is the only one automatically estimates uncertainties and
correlations between fitted variables from the covariance matrix used in
the fit. But, because this approach is sometimes questioned (and rightly
so), lmfit supports methods to do a brute force determiniation of the
confidence intervals for sets of parameters.

Lmfit provides high-level curve-fitting or data modeling functionality
through its :class:`Model` class, which extends the capabilities of
:func:`scipy.optimize.curve_fit`.  This allows you to turn a function that
generates a model for your data into a python class that helps you
parametrize and fit data with that model.  Many pre-built models for common
lineshapes are included and ready to use.

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py

The lmfit package is an open-source project, and the software and this document are works
in progress.  If you are interested in participating in this effort please
use the `lmfit github repository`_.


.. toctree::
   :maxdepth: 2

   intro
   installation
   parameters
   fitting
   model
   builtin_models
   confidence
   bounds
   constraints



