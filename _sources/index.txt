.. lmfit documentation master file, 

Non-Linear Least-Square Minimization for Python
================================================

The lmfit Python package provides a simple, flexible interface to
non-linear least-squares fitting.  LMFIT uses the Levenberg-Marquardt
from MINPACK-1 as implemented in scipy.optimize.leastsq.  While that
function provides the core numerical routine for non-linear least-squares
minimization, the lmfit packaage adds a few simple conveniences.

For any least-squares minimization, the programmer must provide a function
that takes a set of values for the variables in the fit, and produces the
residual function to be minimized in the least-squares sense. 

The lmfit package allows models to be written in terms of Parameters,
which are extensions of simple numerical variables with the following
properties:

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
   simple
   outputs
   constraints

