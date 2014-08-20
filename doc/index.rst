.. lmfit documentation master file,

Non-Linear Least-Square Minimization and Curve-Fitting for Python
===========================================================================

.. _Levenberg-Marquardt:     http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
.. _MINPACK-1:               http://en.wikipedia.org/wiki/MINPACK
.. _Nelder-Mead:             http://en.wikipedia.org/wiki/Nelder-Mead_method

The lmfit python package provides a simple and flexible interface to
non-linear optimization and curve fitting problems.  Lmfit extends the
optimization capabilities of :mod:`scipy.optimize`.  Initially designed to
extend the the `Levenberg-Marquardt`_ algorithm in
:func:`scipy.optimize.minimize.leastsq`, lmfit supports most of the
optimization methods from :mod:`scipy.optimize`.  It also provides a simple
way to apply this extension to *curve fitting* problems.

The key concept in lmfit is that instead of using plain floating pointing
values for the variables to be optimized (as all the optimization routines
in :mod:`scipy.optimize` use), optimizations are done using
:class:`Parameter` objects.  A :class:`Parameter` can have its value fixed
or varied, have upper and/or lower bounds placed on its value, or have
values that are evaluated from algebraic expressions of other Parameter
values.  This is all done outside the optimization routine, so that these
bounds and constraints can be applied to **all** optimization routines from
:mod:`scipy.optimize`, and with a more Pythonic interface than any of the
routines that do provide bounds.

By using :class:`Parameter` objects instead of plain variables, the
objective function does not have to be rewritten to reflect every change of
what is varied in the fit, or if relationships or constraints are placed on
the Parameters.  This simplifies the writing of models, and gives the user
more flexibility in using and testing variations of that model.


Lmfit supports several of the optimization methods from
:mod:`scipy.optimize`.  The default, and by far best tested optimization
method used (and the origin of the name) is the `Levenberg-Marquardt`_
algorithm of :func:`scipy.optimize.leastsq` and
:func:`scipy.optimize.curve_fit`.  Much of this document assumes this
algorithm is used unless explicitly stated.  An important point for many
scientific analysis is that this is only method that automatically
estimates uncertainties and correlations between fitted variables from the
covariance matrix calculated during the fit. Because the approach derived
from `MINPACK-1`_ using the covariance matrix to determine uncertainties is
sometimes questioned (and sometimes rightly so), lmfit supports methods to
do a brute force search of the confidence intervals and correlations for
sets of parameters.

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py

The lmfit package is an open-source project, and this document are a works
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



