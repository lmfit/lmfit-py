#!/usr/bin/env python

from setuptools import setup

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
others are also supported.  Bounds and constraints can be placed on
Parameters for all of these methods.

In addition, methods for explicitly calculating confidence intervals are
provided for exploring minmization problems where the approximation of
estimating Parameter uncertainties from the covariance matrix is
questionable. """

setup(name='lmfit',
      use_scm_version={
          'write_to': 'lmfit/version.py',
          'version_scheme': 'post-release'},
      author='LMFit Development Team',
      author_email='matt.newville@gmail.com',
      url='https://lmfit.github.io/lmfit-py/',
      download_url='https://lmfit.github.io//lmfit-py/',
      setup_requires=['setuptools_scm'],
      install_requires=['asteval>=0.9.22',
                        'numpy>=1.18',
                        'scipy>=1.4',
                        'uncertainties>=3.0.1'],
      python_requires='>=3.6',
      license='BSD-3',
      description="Least-Squares Minimization with Bounds and Constraints",
      long_description=long_desc,
      platforms=['Windows', 'Linux', 'Mac OS X'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Topic :: Scientific/Engineering',
                   ],
      keywords='curve-fitting, least-squares minimization',
      tests_require=['pytest'],
      package_dir={'lmfit': 'lmfit'},
      packages=['lmfit'],
      )
