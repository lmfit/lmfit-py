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
.. _simulated annealing: http://en.wikipedia.org/wiki/Simulated_annealing

.. _MINPACK-1: http://en.wikipedia.org/wiki/MINPACK
.. _asteval: http://newville.github.com/asteval/

The lmfit Python package provides a simple, flexible interface to
non-linear optimization or curve fitting problems.  The default approach
uses the `Levenberg-Marquardt`_ algorithm from `scipy.optimize.leastsq`_,
but other optimization procedures such as Nelder-Mead downhill simplex,
Powell's method, COBYLA, Sequential Least Squares, and a few other
techniques available from `scipy.optimize`_ can also be used.

For any optimization problem, the programmer must provide an objective
function that takes a set of values for the variables in the fit, and
produces either a scalar value to be minimized or a residual array to be
minimized in the least-squares sense.  The lmfit package allows models to
be written in terms of a set of Parameters objects, which are extensions of
simple numerical variables with the following properties:

 * Parameters can be fixed or floated in the fit.
 * Parameters can be bounded with a minimum and/or maximum value.
 * Parameters can be written as simple mathematical expressions of
   other Parameters, using the `asteval`_ module (which is included with
   lmfit).  These values will be re-evaluated at each step in the fit,
   so that the expression is satisfied.  This gives a simple but flexible
   approach to constraining fit variables.

That is, a Parameter is an objects which not only has a value but also has
upper and lower bounds, a flag for whether it is to be varied or not, and
an optional mathematical expression to be used to determine its value.  The
principle advantage of using Parameters instead of fit variables is that
the objective function does not have to be rewritten for a change in what
is varied or what constraints are placed on the fit.  A programmer can
write a general model that encapsulates the phenomenon to be optimized, and
then allow a user of the model to change what is varied and what
constraints are placed on the model.  The ability to easily change whether
a Parameter is floated or fixed also allows one to easily test the
significance of certain Parameters to the fitting model.

By default, lmfit uses and builds upon the `Levenberg-Marquardt`_
minimization algorithm from `MINPACK-1`_ as implemented in
`scipy.optimize.leastsq`_.  A few other optimization routines are also
supported, including `Nelder-Mead`_ simplex downhill method as implemented
in `scipy.optimize.fmin`_, and several others from `scipy.optimize`_. Some
methods, including the `L-BFGS`_ (limited memory
Broyden-Fletcher-Goldfarb-Shanno) algorithm as implemented in
`scipy.optimize.l_bfgs_b`_ and the `simulated annealing`_ algorithm as
implemented in `scipy.optimize.anneal`_ are implemented, but appear to not
work very well.  In their native form, some of these methods setting upper
or lower bounds on parameters, or adding constraints on fitted variables.
By using Parameter objects, lmfit allows bounds and constraints for all of
these methods, and makes it easy to swap between methods.

To be clear, The Levenberg-Marquardt algorithm is by far the most tested
and best support method in lmfit, and much of this document assumes this
algorithm is used unless explicitly stated. An important point for many
scientific analysis is that only the Levenberg-Marquardt algorithm
calculates the estimated uncertainties and correlations between fitted
variables from the covariance matrix used in the fit.

Because this approach of using the covariance matrix to determine
uncertainties is sometimes questioned (and sometimes rightly so), lmfit
supports methods to do a brute force search of the confidence intervals and
correlations for sets of parameters.

lmfit and this document are a work in progress.

.. toctree::
   :maxdepth: 2

   installation
   parameters
   fitting
   confidence
   bounds
   constraints

