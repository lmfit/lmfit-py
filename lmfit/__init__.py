"""
Lmfit provides a high-level interface to non-linear optimization and curve
fitting problems for Python. Lmfit builds on Levenberg-Marquardt algorithm of
scipy.optimize.leastsq(), but also supports most of the optimization methods
from scipy.optimize.  It has a number of useful enhancements, including:

  * Using Parameter objects instead of plain floats as variables.  A Parameter
    has a value that can be varied in the fit, fixed, have upper and/or lower
    bounds.  It can even have a value that is constrained by an algebraic
    expression of other Parameter values.

  * Ease of changing fitting algorithms.  Once a fitting model is set up, one
    can change the fitting algorithm without changing the objective function.

  * Improved estimation of confidence intervals.  While
    scipy.optimize.leastsq() will automatically calculate uncertainties and
    correlations from the covariance matrix, lmfit also has functions to
    explicitly explore parameter space to determine confidence levels even for
    the most difficult cases.

  * Improved curve-fitting with the Model class.  This which extends the
    capabilities of scipy.optimize.curve_fit(), allowing you to turn a function
    that models for your data into a python class that helps you parametrize
    and fit data with that model.

  * Many pre-built models for common lineshapes are included and ready to use.

   version: 0.8.0
   last update: 2014-Sep-21
   License: MIT
   Authors:  Matthew Newville, The University of Chicago
             Till Stensitzki, Freie Universitat Berlin
             Daniel B. Allen, Johns Hopkins University
             Antonino Ingargiola, University of California, Los Angeles
"""

from .minimizer import minimize, Minimizer, MinimizerException
from .parameter import Parameter, Parameters
from .confidence import conf_interval, conf_interval2d
from .printfuncs import (fit_report, ci_report,
                         report_fit, report_ci, report_errors)

from .model import Model, CompositeModel
from . import models

from . import uncertainties
from .uncertainties import ufloat, correlated_values


## versioneer code
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
