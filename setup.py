#!/usr/bin/env python
from setuptools import setup

import lib as lmfit
import numpy, scipy

long_desc = """A library for least-squares minimization and data fitting in
Python.  Built on top of scipy.optimize, lmfit provides a Parameter object
which can be set as fixed or free, can have upper and/or lower bounds, or
can be written in terms of algebraic constraints of other Parameters.  The
user writes a function to be minimized as a function of these Parameters,
and the scipy.optimize methods are used to find the optimal values for the
Parameters.  The Levenberg-Marquardt (leastsq) is the default minimization
algorithm, and provides estimated standard errors and correlations between
varied Parameters.  Other minimization methods, including Nelder-Mead's
downhill simplex, Powell's method, BFGS, Sequential Least Squares, and
others are also supported.  Bounds and contraints can be placed on
Parameters for all of these methods.

In addition, methods for explicitly calculating confidence intervals are
provided for exploring minmization problems where the approximation of
estimating Parameter uncertainties from the covariance matrix is
questionable. """


setup(name = 'lmfit',
      version = lmfit.__version__,
      author = 'Matthew Newville',
      author_email = 'newville@cars.uchicago.edu',
      url          = 'http://cars9.uchicago.edu/software/python/lmfit/',
      download_url = 'http://newville.github.com/lmfit-py/',
      requires = ('numpy', 'scipy'),      
      license = 'BSD',
      description = "Least-Squares Minimization with Bounds and Constraints",
      long_description = long_desc,
      platforms = ('Windows', 'Linux', 'Mac OS X'),
      classifiers=['Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   ],
      
      package_dir = {'lmfit': 'lib'},
      packages = ['lmfit'],
      )

