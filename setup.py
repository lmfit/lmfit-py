#!/usr/bin/env python
# from distutils.core import setup
from __future__ import print_function

from setuptools import setup
import versioneer

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

with open('requirements.txt', 'r') as f:
    install_reqs = f.read().splitlines()

setup(name='lmfit',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      author='LMFit Development Team',
      author_email='matt.newville@gmail.com',
      url='https://lmfit.github.io/lmfit-py/',
      download_url='https://lmfit.github.io//lmfit-py/',
      install_requires=install_reqs,
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*',
      license='BSD',
      description="Least-Squares Minimization with Bounds and Constraints",
      long_description=long_desc,
      platforms=['Windows', 'Linux', 'Mac OS X'],
      classifiers=['Intended Audience :: Science/Research',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                  ],
      # test_suite='nose.collector',
      # test_requires=['Nose'],
      package_dir={'lmfit': 'lmfit'},
      packages=['lmfit', 'lmfit.ui'],
     )
