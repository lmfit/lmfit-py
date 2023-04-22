LMfit-py
========

.. image:: https://dev.azure.com/lmfit/lmfit-py/_apis/build/status/lmfit.lmfit-py?branchName=master
    :target: https://dev.azure.com/lmfit/lmfit-py/_build/latest?definitionId=1&branchName=master

.. image:: https://codecov.io/gh/lmfit/lmfit-py/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/lmfit/lmfit-py

.. image:: https://img.shields.io/pypi/v/lmfit.svg
   :target: https://pypi.org/project/lmfit

.. image:: https://img.shields.io/pypi/dm/lmfit.svg
   :target: https://pypi.org/project/lmfit

.. image:: https://img.shields.io/badge/docs-read-brightgreen
   :target: https://lmfit.github.io/lmfit-py/

.. image:: https://zenodo.org/badge/4185/lmfit/lmfit-py.svg
   :target: https://doi.org/10.5281/zenodo.598352

.. _LMfit google mailing list: https://groups.google.com/group/lmfit-py
.. _Github Discussions: https://github.com/lmfit/lmfit-py/discussions
.. _Github Issues: https://github.com/lmfit/lmfit-py/issues


..
   Note: the Zenodo target should be
   https://zenodo.org/badge/latestdoi/4185/lmfit/lmfit-py
   but see https://github.com/lmfit/lmfit-py/discussions/862


Overview
---------

The lmfit Python library supports provides tools for non-linear least-squares
minimization and curve fitting.  The goal is to make these optimization
algorithms more flexible, more comprehensible, and easier to use well, with the
key feature of casting variables in minimization and fitting routines as named
parameters that can have many attributes beside just a current value.

LMfit is a pure Python package, built on top of Scipy and Numpy, and so easy to
install with ``pip install lmfit``.

For questions, comments, and suggestions, please use the `LMfit google mailing
list`_ or `Github discussions`_.  For software issues and bugs, use `Github
Issues`_, but please read `Contributing.md <.github/CONTRIBUTING.md>`_ before
creating an Issue.


Parameters and Minimization
------------------------------

LMfit provides optimization routines similar to (and based on) those from
``scipy.optimize``, but with a simple, flexible approach to parameterizing a
model for fitting to data using named parameters. These named Parameters can be
held fixed or freely adjusted in the fit, or held between lower and upper
bounds. Parameters can also be constrained as a simple mathematical expression
of other Parameters.

A Parameters object (which acts like a Python dictionary) contains named
parameters, and can be built as with::

    import lmfit
    fit_params = lmfit.Parameters()
    fit_params['amp'] = lmfit.Parameter(value=1.2)
    fit_params['cen'] = lmfit.Parameter(value=40.0, vary=False)
    fit_params['wid'] = lmfit.Parameter(value=4, min=0)
    fit_params['fwhm'] = lmfit.Parameter(expr='wid*2.355')

or using the equivalent::

    fit_params = lmfit.create_params(amp=1.2,
                                     cen={'value':40, 'vary':False},
                                     wid={'value': 4, 'min':0},
                                     fwhm={'expr': 'wid*2.355'})



In the general minimization case (see below for Curve-fitting), the user will
also write an objective function to be minimized (in the least-squares sense)
with its first argument being this Parameters object, and additional positional
and keyword arguments as desired::

    def myfunc(params, x, data, someflag=True):
        amp = params['amp'].value
        cen = params['cen'].value
        wid = params['wid'].value
        ...
        return residual_array

For each call of this function, the values for the ``params`` may have changed,
subject to the bounds and constraint settings for each Parameter. The function
should return the residual (i.e., ``data-model``) array to be minimized.

The advantage here is that the function to be minimized does not have to be
changed if different bounds or constraints are placed on the fitting Parameters.
The fitting model (as described in myfunc) is instead written in terms of
physical parameters of the system, and remains remains independent of what is
actually varied in the fit. In addition, which parameters are adjusted and which
are fixed happens at run-time, so that changing what is varied and what
constraints are placed on the parameters can easily be modified by the user in
real-time data analysis.

To perform the fit, the user calls::

    result = lmfit.minimize(myfunc, fit_params, args=(x, data), kws={'someflag':True}, ....)

After the fit, a ``MinimizerResult`` class is returned that holds the results
the fit (e.g., fitting statistics and optimized parameters). The dictionary
``result.params`` contains the best-fit values, estimated standard deviations,
and correlations with other variables in the fit.

By default, the underlying fit algorithm is the Levenberg-Marquardt algorithm
with numerically-calculated derivatives from MINPACK's lmdif function, as used
by ``scipy.optimize.leastsq``. Most other solvers that are present in ``scipy``
(e.g., Nelder-Mead, differential_evolution, basin-hopping, and more) are also
supported.


Curve-Fitting with lmfit.Model
----------------------------------

One of the most common use of least-squares minimization is for curve fitting,
where minimization of ``data-model``, or ``(data-model)*weights``.  Using
``lmfit.minimize`` as above, the objective function would take ``data`` and
``weights`` and effectively calculated the model and then return the value of
``(data-model)*weights``.

To simplify this, and make curve-fitting more flexible, lmfit provides a Model
class that wraps a *model function* that represents the model (without the data
or weights).  Parameters are then automatically found from the named arguments
of the model function.  In addition, simple model functions can be readily
combined and reused, and several common model functions are included in lmfit.

Exploration of Confidence Intervals
-------------------------------------

Lmfit tries to always estimate uncertainties in fitting parameters and
correlations between them.  It does this even for those methods where the
corresponding ``scipy.optimize`` routines do not estimate uncertainties.  Lmfit
also provides methods to explicitly explore and evaluate the confidence
intervals in fit results.
