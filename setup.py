#!/usr/bin/env python
from distutils.core import setup
import os
import sys
import lib as lmfit

import numpy
import scipy

setup(name = 'lmfit',
      version = lmfit.__version__,
      author = 'Matthew Newville',
      author_email = 'newville@cars.uchicago.edu',
      url         = 'http://cars9.uchicago.edu/software/python/lmfit/',
      license = 'BSD',
      description = "Least-Squares Minimization with Constraints for Python",
      package_dir = {'lmfit': 'lib'},
      packages = ['lmfit'],
      )

