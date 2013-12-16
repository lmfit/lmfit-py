.. lmfit documentation master file,

Non-Linear Least-Square Minimization for Python
================================================

.. _scipy.optimize.leastsq: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
.. _scipy.optimize.l_bfgs_b: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
.. _scipy.optimize.anneal: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.anneal.html
.. _scipy.optimize.fmin:   http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
.. _scipy.optimize.cobyla: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.cobyla.html
.. _scipy.optimize: http://docs.scipy.org/doc/scipy/reference/optimize.html

.. _Nelder-Mead: http://en.wikipedia.org/wiki/Nelder-Mead_method
.. _Levenberg-Marquardt: http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
.. _L-BFGS:  http://en.wikipedia.org/wiki/Limited-memory_BFGS

.. _MINPACK-1: http://en.wikipedia.org/wiki/MINPACK

The lmfit Python package provides a simple, flexible interface to
non-linear optimization or curve fitting problems.  The package extends the
optimization capabilities of `scipy.optimize`_ by replacing floating
pointing values for the variables to be optimized with Parameter objects.
These Parameters can be fixed or varied, have upper and/or lower bounds
placed on its value, or written as an algebraic expression of other
Parameters.

The principal advantage of using Parameters instead of simple variables is
that the objective function does not have to be rewritten to reflect every
change of what is varied in the fit, or what relationships or constraints
are placed on the Parameters.  This means a scientific programmer can write
a general model that encapsulates the phenomenon to be optimized, and then
allow user of that model to change what is varied and fixed, what range of
values is acceptable for Parameters, and what constraints are placed on the
model.  The ease with which the model can be changed also allows one to
easily test the significance of certain Parameters in a fitting model.

The lmfit package allows a choice of several optimization methods available
from `scipy.optimize`_.  The default, and by far best tested optimization
method used is the `Levenberg-Marquardt`_ algorithm from
from `MINPACK-1`_ as implemented in `scipy.optimize.leastsq`_.
This method is by far the most tested and best support method in lmfit, and
much of this document assumes this algorithm is used unless explicitly
stated. An important point for many scientific analysis is that this is
only method that automatically estimates uncertainties and correlations
between fitted variables from the covariance matrix calculated during the fit.

A few other optimization routines are also supported, including
`Nelder-Mead`_ simplex downhill, Powell's method, COBYLA, Sequential Least
Squares methods as implemented in `scipy.optimize.fmin`_, and several
others from `scipy.optimize`_.  In their native form, some of these methods
setting allow upper or lower bounds on parameter variables, or adding
constraints on fitted variables.  By using Parameter objects, lmfit allows
bounds and constraints for *all* of these methods, and makes it easy to
swap between methods without hanging the objective function or set of
Parameters.

Finally, because the approach derived from `MINPACK-1`_ usin the covariance
matrix to determine uncertainties is sometimes questioned (and sometimes
rightly so), lmfit supports methods to do a brute force search of the
confidence intervals and correlations for sets of parameters.

lmfit and this document are a work in progress.

.. toctree::
   :maxdepth: 2

   installation
   parameters
   fitting
   confidence
   bounds
   constraints
   models1d
