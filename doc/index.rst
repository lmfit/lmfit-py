.. lmfit documentation master file,

Non-Linear Least-Square Minimization for Python
================================================

.. _scipy_opt_link: http://example.com/

The lmfit Python package provides a simple, flexible interface to
non-linear least-squares fitting.  Currently, LMFIT uses the
Levenberg-Marquardt from MINPACK-1 as implemented in `Link
scipy.optimize.leastsq
http://docs.scipy.org/doc/scipy/reference/optimize.html`_, but the
intention is that will soon support other optimization routines.  While
these functions provide the core numerical algorithm for non-linear
least-squares minimization, the lmfit package adds a few simple
conveniences.

For any minimization problem, the programmer must provide a function that
takes a set of values for the variables in the fit, and produces the
residual function to be minimized in the least-squares sense.

The lmfit package allows models to be written in terms of a set of
Parameters, which are extensions of simple numerical variables with the
following properties:

 * Parameters can be fixed or floated in the fit.
 * Parameters can be bounded with a minimum and/or maximum value.
 * Parameters can be written as simple mathematical expressions of
   other Parameters.  These values will be re-evaluated at each
   step in the fit, so that the expression is statisfied.  This gives
   a simple but flexible approach to constraining fit variables.

The main advantage to using Parameters instead of fit variables is that the
model function does not have to be rewritten for a change in what is varied
or what constraints are placed on the fit.  The programmer can write a
fairly general model, and allow a user of the model to change what is
varied and what constraints are placed on the model.

In addition, lmfit calculates are reports the estimated uncertainties
and correlation between fitted variables.

.. toctree::
   :maxdepth: 2

   installation
   parameters
   fitting
   constraints

