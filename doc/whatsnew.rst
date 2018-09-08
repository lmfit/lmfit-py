.. _whatsnew_chapter:

=====================
Release Notes
=====================

.. _lmfit GitHub repository:   https://github.com/lmfit/lmfit-py

This section discusses changes between versions, especially changes
significant to the use and behavior of the library.  This is not meant
to be a comprehensive list of changes.  For such a complete record,
consult the `lmfit GitHub repository`_.


.. _whatsnew_0910_label:

.. _Andrea Gavana: http://infinity77.net/global_optimization/index.html
.. _AMPGO paper: http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

Version 0.9.10 Release Notes
==========================================
Two new global algorithms were added: basinhopping and AMPGO.
Basinhopping wraps the method present in `scipy`, and more information
can be found in the documentation (:func:`~lmfit.minimizer.Minimizer.basinhopping`
and :scipydoc:`optimize.basinhopping`).
The Adaptive Memory Programming for Global Optimization (AMPGO) algorithm
was adapted from Python code written by `Andrea Gavana`_. A more detailed
explanation of the algorithm is available in the `AMPGO paper`_ and specifics
for lmfit can be found in the :func:`~lmfit.minimizer.Minimizer.ampgo` function.

Lmfit uses the external uncertainties (https://github.com/lebigot/uncertainties)
package (available on PyPI), instead of distributing its own fork.

An `AbortFitException` is now raised when the fit is aborted by the user (i.e., by
using `iter_cb`).

Bugfixes:

- all exceptions are allowed when trying to import matplotlib
- simplify and fix corner-case errors when testing closeness of large integers


.. _whatsnew_099_label:

Version 0.9.9 Release Notes
==========================================
Lmfit now uses the asteval (https://github.com/newville/asteval) package
instead of distributing its own copy. The minimum required asteval version
is 0.9.12, which is available on PyPI. If you see import errors related to
asteval, please make sure that you actually have the latest version installed.


.. _whatsnew_096_label:

Version 0.9.6 Release Notes
==========================================

Support for SciPy 0.14 has been dropped: SciPy 0.15 is now required.  This
is especially important for lmfit maintenance, as it means we can now rely
on SciPy having code for differential evolution and do not need to keep a
local copy.

A brute force method was added, which can be used either with
:meth:`Minimizer.brute` or using the `method='brute'` option to
:meth:`Minimizer.minimize`.  This method requires finite bounds on
all varying parameters, or that parameters have a finite
`brute_step` attribute set to specify the step size.

Custom cost functions can now be used for the scalar minimizers using the
`reduce_fcn` option.

Many improvements to documentation and docstrings in the code were made.
As part of that effort, all API documentation in this main Sphinx
documentation now derives from the docstrings.

Uncertainties in the resulting best-fit for a model can now be calculated
from the uncertainties in the model parameters.

Parameters have two new attributes: `brute_step`, to specify the step
size when using the `brute` method, and `user_data`, which is unused but
can be used to hold additional information the user may desire.  This will
be preserved on copy and pickling.

Several bug fixes and cleanups.

Versioneer was updated to 0.18.

Tests can now be run either with nose or pytest.


.. _whatsnew_095_label:

Version 0.9.5 Release Notes
==========================================

Support for Python 2.6 and SciPy 0.13 has been dropped.

.. _whatsnew_094_label:

Version 0.9.4 Release Notes
==========================================

Some support for the new `least_squares` routine from SciPy 0.17 has been
added.


Parameters can now be used directly in floating point or array expressions,
so that the Parameter value does not need `sigma = params['sigma'].value`.
The older, explicit usage still works, but the docs, samples, and tests
have been updated to use the simpler usage.

Support for Python 2.6 and SciPy 0.13 is now explicitly deprecated and wil
be dropped in version 0.9.5.

.. _whatsnew_093_label:

Version 0.9.3 Release Notes
==========================================

Models involving complex numbers have been improved.

The `emcee` module can now be used for uncertainty estimation.

Many bug fixes, and an important fix for performance slowdown on getting
parameter values.

ASV benchmarking code added.


.. _whatsnew_090_label:

Version 0.9.0 Release Notes
==========================================

This upgrade makes an important, non-backward-compatible change to the way
many fitting scripts and programs will work.  Scripts that work with
version 0.8.3 will not work with version 0.9.0 and vice versa.  The change
was not made lightly or without ample discussion, and is really an
improvement.  Modifying scripts that did work with 0.8.3 to work with 0.9.0
is easy, but needs to be done.



Summary
~~~~~~~~~~~~

The upgrade from 0.8.3 to 0.9.0 introduced the :class:`MinimizerResult`
class (see :ref:`fit-results-label`) which is now used to hold the return
value from :func:`minimize` and :meth:`Minimizer.minimize`.  This returned
object contains many goodness of fit statistics, and holds the optimized
parameters from the fit.  Importantly, the parameters passed into
:func:`minimize` and :meth:`Minimizer.minimize` are no longer modified by
the fit. Instead, a copy of the passed-in parameters is made which is
changed and returns as the :attr:`params` attribute of the returned
:class:`MinimizerResult`.


Impact
~~~~~~~~~~~~~

This upgrade means that a script that does::

    my_pars = Parameters()
    my_pars.add('amp',    value=300.0, min=0)
    my_pars.add('center', value=  5.0, min=0, max=10)
    my_pars.add('decay',  value=  1.0, vary=False)

    result = minimize(objfunc, my_pars)

will still work, but that ``my_pars`` will **NOT** be changed by the fit.
Instead, ``my_pars`` is copied to an internal set of parameters that is
changed in the fit, and this copy is then put in ``result.params``.  To
look at fit results, use ``result.params``, not ``my_pars``.

This has the effect that ``my_pars`` will still hold the starting parameter
values, while all of the results from the fit are held in the ``result``
object returned by :func:`minimize`.

If you want to do an initial fit, then refine that fit to, for example, do
a pre-fit, then refine that result different fitting method, such as::

    result1 = minimize(objfunc, my_pars, method='nelder')
    result1.params['decay'].vary = True
    result2 = minimize(objfunc, result1.params, method='leastsq')

and have access to all of the starting parameters ``my_pars``, the result of the
first fit ``result1``, and the result of the final fit ``result2``.



Discussion
~~~~~~~~~~~~~~

The main goal for making this change were to

1. give a better return value to :func:`minimize` and
   :meth:`Minimizer.minimize` that can hold all of the information
   about a fit.  By having the return value be an instance of the
   :class:`MinimizerResult` class, it can hold an arbitrary amount of
   information that is easily accessed by attribute name, and even
   be given methods.  Using objects is good!

2. To limit or even eliminate the amount of "state information" a
   :class:`Minimizer` holds.  By state information, we mean how much of
   the previous fit is remembered after a fit is done.  Keeping (and
   especially using) such information about a previous fit means that
   a :class:`Minimizer` might give different results even for the same
   problem if run a second time.  While it's desirable to be able to
   adjust a set of :class:`Parameters` re-run a fit to get an improved
   result, doing this by changing an internal attribute
   (:attr:`Minimizer.params`) has the undesirable side-effect of not
   being able to "go back", and makes it somewhat cumbersome to keep
   track of changes made while adjusting parameters and re-running fits.
