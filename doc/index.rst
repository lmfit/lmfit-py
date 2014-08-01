.. lmfit documentation master file,

Non-Linear Least-Square Minimization for Python
================================================

.. _scipy.optimize:          http://docs.scipy.org/doc/scipy/reference/optimize.html
.. _scipy.optimize.leastsq:  http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
.. _scipy.optimize.minimize: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _Nelder-Mead:             http://en.wikipedia.org/wiki/Nelder-Mead_method
.. _MINPACK-1:               http://en.wikipedia.org/wiki/MINPACK
.. _Levenberg-Marquardt:     http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

The lmfit python package provides a simple and flexible interface to
non-linear optimization or curve fitting problems.  Lmfit extends the
optimization capabilities of `scipy.optimize`_.  Initially designed to
extend the the `Levenberg-Marquardt`_ algorithm from `MINPACK-1`_ as
implemented in `scipy.optimize.leastsq`_ , lmfit supports most of the
optimization methods from `scipy.optimize`_, including `Nelder-Mead` and
all the methods available from `scipy.optimize.minimize`_.

The key concept for lmfit is that instead of using plain floating pointing
values for the variables to be optimized (as all the optimization routines
in `scipy.optimize`_ use), optimizations are done using :class:`Parameter`
objects.  A :class:`Parameter` can have its value fixed or varied, have
upper and/or lower bounds placed on its value, or have values that are
evaluated from algebraic expressions of other Parameter values.  This is
all done outside the optimization routine, and means that lmfit provides
bounds and constraints on Parameters for **all** optimization routines from
`scipy.optimize`_, and with a more Pythonic interface than any of the
routines that do provide bounds.

By using :class:`Parameter` objects instead of plain variables, the
objective function does not have to be rewritten to reflect every change of
what is varied in the fit, or if relationships or constraints are placed on
the Parameters.  This means that the scientific programmer can write a
general model that encapsulates the phenomenon to be modeled.  The user of
that model can change what is varied, what is fixed, what ranges of values
are acceptable for Parameters, and what constraints are placed between
Parameters in the model without changing the objective function.  The ease
with which Parameters allow a model to be changed also allows better (and
automated) testing the significance of individual Parameters in a fitting
model.

As mentioned above, the lmfit package allows a choice of several
optimization methods available from `scipy.optimize`_.  The default, and by
far best tested optimization method used (and the origin of the name) is
the `Levenberg-Marquardt`_ algorithm of `scipy.optimize.leastsq`_.  Much of
this document assumes this algorithm is used unless explicitly stated.  An
important point for many scientific analysis is that this is only method
that automatically estimates uncertainties and correlations between fitted
variables from the covariance matrix calculated during the fit.

Because the approach derived from `MINPACK-1`_ using the covariance matrix to
determine uncertainties is sometimes questioned (and sometimes rightly so),
lmfit supports methods to do a brute force search of the confidence
intervals and correlations for sets of parameters.

lmfit and this document are a work in progress.  We are currently working
on providing easy-to-use models that for common fitting problems.

.. toctree::
   :maxdepth: 2

   installation
   parameters
   fitting
   confidence
   bounds
   constraints
   models1d
