"""
   LMfit-py provides a Least-Squares Minimization routine and
   class with a simple, flexible approach to parameterizing a
   model for fitting to data.  Named Parameters can be held
   fixed or freely adjusted in the fit, or held between lower
   and upper bounds.  If the separate asteval module has been
   installed, parameters can be constrained as a simple
   mathematical expression of other Parameters.

   version: 0.4
   last update:  8-Apr-2012
   License: BSD
   Author:  Matthew Newville <newville@cars.uchicago.edu>
            Center for Advanced Radiation Sources,
            The University of Chicago

"""
__version__ = '0.4'
from .minimizer import minimize, Minimizer, Parameter, Parameters

__all__ = [minimize, Minimizer, Parameter, Parameters]
