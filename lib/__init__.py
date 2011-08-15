"""
   LMfit-py provides a Least-Squares Minimization routine and
   class with a simple, flexible approach to parameterizing a
   model for fitting to data.  Named Parameters can be held
   fixed or freely adjusted in the fit, or held between lower
   and upper bounds.  In addition, parameters can be constrained
   as a simple mathematical expression of other Parameters.

   version: 0.3
   last update:  15-Aug-2011
   License:  BSD
   Author:  Matthew Newville <newville@cars.uchicago.edu>
            Center for Advanced Radiation Sources,
            The University of Chicago

"""
__version__ = '0.3'

from .minimizer import minimize, Minimizer, Parameter, Parameters
from .asteval import Interpreter, NameFinder

__all__ = [Minimizer, minimize, Parameter, Parameters, Interpreter, NameFinder]
