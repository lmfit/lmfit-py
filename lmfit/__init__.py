"""
LMFIT: Non-Linear Least-Squares Minimization and Curve-Fitting for Python.

Lmfit provides a high-level interface to non-linear optimization and
curve-fitting problems for Python. It builds on the Levenberg-Marquardt
algorithm of `scipy.optimize.leastsq`, but also supports most of the other
optimization methods present in `scipy.optimize`. It has a number of
useful enhancements, including:

  * Using Parameter objects instead of plain floats as variables. A
    Parameter has a value that can be varied in the fit, fixed, have
    upper and/or lower bounds. It can even have a value that is
    constrained by an algebraic expression of other Parameter values.

  * Ease of changing fitting algorithms. Once a fitting model is set
    up, one can change the fitting algorithm without changing the
    objective function.

  * Improved estimation of confidence intervals. While
    `scipy.optimize.leastsq` will automatically calculate uncertainties
    and correlations from the covariance matrix, lmfit also has functions
    to explicitly explore parameter space to determine confidence levels
    even for the most difficult cases.

  * Improved curve-fitting with the Model class. This extends the
    capabilities of `scipy.optimize.curve_fit`, allowing you to turn a
    function that models your data into a Python class that helps you
    parametrize and fit data with that model.

  * Many built-in models for common lineshapes are included and ready
    to use.

Copyright (c) 2023 Lmfit Developers ; BSD-3 license ; see LICENSE

"""
from asteval import Interpreter

from .confidence import conf_interval, conf_interval2d
from .minimizer import Minimizer, MinimizerException, minimize
from .parameter import Parameter, Parameters, create_params
from .printfuncs import ci_report, fit_report, report_ci, report_fit
from .model import Model, CompositeModel
from . import lineshapes, models

from lmfit.version import version as __version__
