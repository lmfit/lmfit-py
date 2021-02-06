.. _whatsnew_chapter:

=============
Release Notes
=============

.. _lmfit GitHub repository: https://github.com/lmfit/lmfit-py

This section discusses changes between versions, especially changes
significant to the use and behavior of the library. This is not meant
to be a comprehensive list of changes. For such a complete record,
consult the `lmfit GitHub repository`_.


.. _whatsnew_1XX_label:


Version 1.0.2 Release Notes
===========================

Version 1.0.2 officially supports Python 3.9 and has dropped support for Python 3.5. The minimum version
of the following dependencies were updated: asteval>=0.9.21, numpy>=1.18, and scipy>=1.3.

New features:

- added two-dimensional Gaussian lineshape and model (PR #642; @mpmdean)
- all built-in models are now registered in lmfit.models.lmfit_models; new Model class attribute "valid_forms" (PR #663; @rayosborn)
- added a SineModel (PR #676; @lneuhaus)
- add the ``run_mcmc_kwargs argument`` to ``Minimizer.emcee`` to pass to the ``emcee.EnsembleSampler.run_mcmc`` function (PR #694; @rbnvrw)

Bug fixes:

- ``ModelResult.eval_uncertainty`` should use provided Parameters (PR #646)
- center in lognormal model can be negative (Issue #644, PR #645; @YoshieraHuang)
- restore best-fit values after calculation of covariance matrix (Issue #655, PR #657)
- add helper-function ``not_zero`` to prevent ZeroDivisionError in lineshapes and use in exponential lineshape (Issue #631, PR #664; @s-weigand)
- save last_internal_values and use to restore internal values if fit is aborted (PR #667)
- dumping a fit using the lbfgsb method now works, convert bytes to string if needed (Issue #677, PR #678; @leonfoks)
- fix use of callable Jacobian for scalar methods (PR #681; @mstimberg)
- preserve float/int types when encoding for JSON (PR #696; @jedzill4)
- better support for saving/loading of ExpressionModels and assure that ``init_params`` and ``init_fit`` are set when loading a ``ModelResult`` (PR #706)

Various:

- update minimum dependencies (PRs #688, #693)
- improvements in coding style, docstrings, CI, and test coverage (PRs #647, #649, #650, #653, #654; #685, #668, #689)
- fix typo in Oscillator (PR #658; @flothesof)
- add example using SymPy (PR #662)
- allow better custom pool for emcee() (Issue #666, PR #667)
- update NIST Strd reference functions and tests (PR #670)
- make building of documentation cross-platform (PR #673; @s-weigand)
- relax module name check in test_check_ast_errors for Python 3.9 (Issue #674, PR #675; @mwhudson)
- fix/update layout of documentation, now uses the sphinx13 theme (PR #687)
- fixed DeprecationWarnings reported by NumPy v1.2.0 (PR #699)
- increase value of ``tiny`` and check for it in bounded parameters to avoid "parameter not moving from initial value" (Issue #700, PR #701)
- add ``max_nfev`` to ``basinhopping`` and ``brute`` (now supported everywhere in lmfit) and set to more uniform default values (PR #701)
- use Azure Pipelines for CI, drop Travis (PRs #696 and #702)


.. _whatsnew_101_label:

Version 1.0.1 Release Notes
============================

**Version 1.0.1 is the last release that supports Python 3.5**. All newer version will
require 3.6+ so that we can use formatting-strings and rely on dictionaries being ordered.

New features:

- added thermal distribution model and lineshape (PR #620; @mpmdean)
- introduced a new argument ``max_nfev`` to uniformly specify the maximum number of function evalutions (PR #610)
  **Please note: all other arguments (e.g., ``maxfev``, ``maxiter``, ...) will no longer be passed to the underlying
  solver. A warning will be emitted stating that one should use ``max_nfev``.**
- the attribute ``call_kws`` was added to the ``MinimizerResult`` class and contains the keyword arguments that are
  supplied to the solver in SciPy.

Bug fixes:

- fixes to the ``load`` and ``__setstate__`` methods of the Parameter class
- fixed failure of ModelResult.dump() due to missing attributes (Issue #611, PR #623; @mpmdean)
- ``guess_from_peak`` function now also works correctly with decreasing x-values or when using
  pandas (PRs #627 and #629; @mpmdean)
- the ``Parameter.set()`` method now correctly first updates the boundaries and then the value (Issue #636, PR #637; @arunpersaud)

Various:

- fixed typo for the use of expressions in the documentation (Issue #610; @jkrogager)
- removal of PY2-compatibility and unused code and improved test coverage (PRs #619, #631, and #633)
- removed deprecated ``isParameter`` function and automatic conversion of an ``uncertainties`` object (PR #626)
- inaccurate FWHM calculations were removed from built-in models, others labeled as estimates (Issue #616 and PR #630)
- corrected spelling mistake for the Doniach lineshape and model (Issue #634; @rayosborn)
- removed unsupported/untested code for IPython notebooks in lmfit/ui/*


.. _whatsnew_100_label:

Version 1.0.0 Release Notes
============================

**Version 1.0.0 supports Python 3.5, 3.6, 3.7, and 3.8**

New features:

- no new features are introduced in 1.0.0.

Improvements:

- support for Python 2 and use of the ``six`` package are removed. (PR #612)

Various:

- documentation updates to clarify the use of ``emcee``. (PR #614)


.. _whatsnew_0915_label:

Version 0.9.15 Release Notes
============================

**Version 0.9.15 is the last release that supports Python 2.7**; it now also fully supports Python 3.8.

New features, improvements, and bug fixes:

- move application of parameter bounds to setter instead of getter (PR #587)
- add support for non-array Jacobian types in least_squares (Issue #588, @ezwelty in PR #589)
- add more information (i.e., acor and acceptance_fraction) about emcee fit (@j-zimmermann in PR #593)
- "name" is now a required positional argument for Parameter class, update the magic methods (PR #595)
- fix nvars count and bound handling in confidence interval calculations (Issue #597, PR #598)
- support Python 3.8; requires asteval >= 0.9.16 (PR #599)
- only support emcee version 3 (i.e., no PTSampler anymore) (PR #600)
- fix and refactor prob_bunc in confidence interval calculations (PR #604)
- fix adding Parameters with custom user-defined symbols (Issue #607, PR #608; thanks to @gbouvignies for the report)

Various:

- bump requirements to LTS version of SciPy/ NumPy and code clean-up (PR #591)
- documentation updates (PR #596, and others)
- improve test coverage and Travis CI updates (PR #595, and others)
- update pre-commit hooks and configuration in setup.cfg

To-be deprecated:
- function Parameter.isParameter and conversion from uncertainties.core.Variable to value in _getval (PR #595)

.. _whatsnew_0914_label:

Version 0.9.14 Release Notes
============================

New features:

- the global optimizers ``shgo`` and ``dual_annealing`` (new in SciPy v1.2) are now supported (Issue #527; PRs #545 and #556)
- ``eval`` method added to the Parameter class (PR #550 by @zobristnicholas)
- avoid ZeroDivisionError in ``printfuncs.params_html_table`` (PR #552 by @aaristov and PR #559)
- add parallelization to ``brute`` method (PR #564, requires SciPy v1.3)

Bug fixes:

- consider only varying parameters when reporting potential issues with calculating errorbars (PR #549) and compare
  ``value`` to both ``min`` and ``max`` (PR #571)
- guard against division by zero in lineshape functions and ``FWHM`` and ``height`` expression calculations (PR #545)
- fix issues with restoring a saved Model (Issue #553; PR #554)
- always set ``result.method`` for ``emcee`` algorithm (PR #558)
- more careful adding of parameters to handle out-of-order constraint expressions (Issue #560; PR #561)
- make sure all parameters in Model.guess() use prefixes (PRs #567 and #569)
- use ``inspect.signature`` for PY3 to support wrapped functions (Issue #570; PR #576)
- fix ``result.nfev``` for ``brute`` method when using parallelization (Issue #578; PR #579)

Various:

- remove "missing" in the Model class (replaced by nan_policy) and "drop" as option to nan_policy
  (replaced by omit) deprecated since 0.9 (PR #565).
- deprecate 'report_errors' in printfuncs.py (PR #571)
- updates to the documentation to use ``jupyter-sphinx`` to include examples/output (PRs #573 and #575)
- include a Gallery with examples in the documentation using ``sphinx-gallery`` (PR #574 and #583)
- improve test-coverage (PRs #571, #572 and #585)
- add/clarify warning messages when NaN values are detected (PR #586)
- several updates to docstrings (Issue #584; PR #583, and others)
- update pre-commit hooks and several docstrings

.. _whatsnew_0913_label:

Version 0.9.13 Release Notes
============================

New features:

- Clearer warning message in fit reports when uncertainties should but cannot be estimated, including guesses of which Parameters to examine (#521, #543)
- SplitLorenztianModel and split_lorentzian function (#523)
- HTML representations for Parameter, MinimizerResult, and Model so that they can be printed better with Jupyter (#524, #548)
- support parallelization for differential evolution (#526)

Bug fixes:

- delay import of matplotlib (and so, the selection of its backend) as late as possible (#528, #529)
- fix for saving, loading, and reloading ModelResults (#534)
- fix to leastsq to report the best-fit values, not the values tried last (#535, #536)
- fix synchronization of all parameter values on Model.guess() (#539, #542)
- improve deprecation warnings for outdated nan_policy keywords (#540)
- fix for edge case in gformat() (#547)

Project management:

- using pre-commit framework to improve and enforce coding style (#533)
- added code coverage report to github main page
- updated docs, github templates, added several tests.
- dropped support and testing for Python 3.4.

.. _whatsnew_0912_label:

Version 0.9.12 Release Notes
============================

Lmfit package is now licensed under BSD-3.

New features:

- SkewedVoigtModel was added as built-in model (Issue #493)
- Parameter uncertainties and correlations are reported for least_squares
- Plotting of complex-valued models is now handled in ModelResult class (PR #503)
- A model's independent variable is allowed to be an object (Issue #492)
- Added ``usersyms`` to Parameters() initialization to make it easier to add custom functions and symbols (Issue #507)
- the ``numdifftools`` package can be used to calculate parameter uncertainties and correlations for all solvers that do not natively support this (PR #506)
- ``emcee`` can now be used as method keyword-argument to Minimizer.minimize and minimize function, which allows for using ``emcee`` in the Model class (PR #512; see ``examples/example_emcee_with_Model.py``)

(Bug)fixes:

- asteval errors are now flushed after raising (Issue #486)
- max_time and evaluation time for ExpressionModel increased to 1 hour (Issue #489)
- loading a saved ModelResult now restores all attributes (Issue #491)
- development versions of scipy and emcee are now supported (Issue #497 and PR #496)
- ModelResult.eval() do no longer overwrite the userkws dictionary (Issue #499)
- running the test suite requires ``pytest`` only (Issue #504)
- improved FWHM calculation for VoigtModel (PR #514)


.. _whatsnew_0910_label:

.. _Andrea Gavana: http://infinity77.net/global_optimization/index.html
.. _AMPGO paper: http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf

Version 0.9.10 Release Notes
============================
Two new global algorithms were added: basinhopping and AMPGO.
Basinhopping wraps the method present in ``scipy``, and more information
can be found in the documentation (:func:`~lmfit.minimizer.Minimizer.basinhopping`
and :scipydoc:`optimize.basinhopping`).
The Adaptive Memory Programming for Global Optimization (AMPGO) algorithm
was adapted from Python code written by `Andrea Gavana`_. A more detailed
explanation of the algorithm is available in the `AMPGO paper`_ and specifics
for lmfit can be found in the :func:`~lmfit.minimizer.Minimizer.ampgo` function.

Lmfit uses the external uncertainties (https://github.com/lebigot/uncertainties)
package (available on PyPI), instead of distributing its own fork.

An ``AbortFitException`` is now raised when the fit is aborted by the user (i.e., by
using ``iter_cb``).

Bugfixes:

- all exceptions are allowed when trying to import matplotlib
- simplify and fix corner-case errors when testing closeness of large integers


.. _whatsnew_099_label:

Version 0.9.9 Release Notes
===========================
Lmfit now uses the asteval (https://github.com/newville/asteval) package
instead of distributing its own copy. The minimum required asteval version
is 0.9.12, which is available on PyPI. If you see import errors related to
asteval, please make sure that you actually have the latest version installed.


.. _whatsnew_096_label:

Version 0.9.6 Release Notes
===========================

Support for SciPy 0.14 has been dropped: SciPy 0.15 is now required. This
is especially important for lmfit maintenance, as it means we can now rely
on SciPy having code for differential evolution and do not need to keep a
local copy.

A brute force method was added, which can be used either with
:meth:`Minimizer.brute` or using the ``method='brute'`` option to
:meth:`Minimizer.minimize`. This method requires finite bounds on
all varying parameters, or that parameters have a finite
``brute_step`` attribute set to specify the step size.

Custom cost functions can now be used for the scalar minimizers using the
``reduce_fcn`` option.

Many improvements to documentation and docstrings in the code were made.
As part of that effort, all API documentation in this main Sphinx
documentation now derives from the docstrings.

Uncertainties in the resulting best-fit for a model can now be calculated
from the uncertainties in the model parameters.

Parameters have two new attributes: ``brute_step``, to specify the step
size when using the ``brute`` method, and ``user_data``, which is unused but
can be used to hold additional information the user may desire. This will
be preserved on copy and pickling.

Several bug fixes and cleanups.

Versioneer was updated to 0.18.

Tests can now be run either with nose or pytest.


.. _whatsnew_095_label:

Version 0.9.5 Release Notes
===========================

Support for Python 2.6 and SciPy 0.13 has been dropped.

.. _whatsnew_094_label:

Version 0.9.4 Release Notes
===========================

Some support for the new ``least_squares`` routine from SciPy 0.17 has been
added.


Parameters can now be used directly in floating point or array expressions,
so that the Parameter value does not need ``sigma = params['sigma'].value``.
The older, explicit usage still works, but the docs, samples, and tests
have been updated to use the simpler usage.

Support for Python 2.6 and SciPy 0.13 is now explicitly deprecated and will
be dropped in version 0.9.5.

.. _whatsnew_093_label:

Version 0.9.3 Release Notes
===========================

Models involving complex numbers have been improved.

The ``emcee`` module can now be used for uncertainty estimation.

Many bug fixes, and an important fix for performance slowdown on getting
parameter values.

ASV benchmarking code added.


.. _whatsnew_090_label:

Version 0.9.0 Release Notes
===========================

This upgrade makes an important, non-backward-compatible change to the way
many fitting scripts and programs will work. Scripts that work with
version 0.8.3 will not work with version 0.9.0 and vice versa. The change
was not made lightly or without ample discussion, and is really an
improvement. Modifying scripts that did work with 0.8.3 to work with 0.9.0
is easy, but needs to be done.



Summary
~~~~~~~

The upgrade from 0.8.3 to 0.9.0 introduced the :class:`MinimizerResult`
class (see :ref:`fit-results-label`) which is now used to hold the return
value from :func:`minimize` and :meth:`Minimizer.minimize`. This returned
object contains many goodness of fit statistics, and holds the optimized
parameters from the fit. Importantly, the parameters passed into
:func:`minimize` and :meth:`Minimizer.minimize` are no longer modified by
the fit. Instead, a copy of the passed-in parameters is made which is
changed and returns as the :attr:`params` attribute of the returned
:class:`MinimizerResult`.


Impact
~~~~~~

This upgrade means that a script that does::

    my_pars = Parameters()
    my_pars.add('amp', value=300.0, min=0)
    my_pars.add('center', value=5.0, min=0, max=10)
    my_pars.add('decay', value=1.0, vary=False)

    result = minimize(objfunc, my_pars)

will still work, but that ``my_pars`` will **NOT** be changed by the fit.
Instead, ``my_pars`` is copied to an internal set of parameters that is
changed in the fit, and this copy is then put in ``result.params``. To
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
~~~~~~~~~~

The main goal for making this change were to

1. give a better return value to :func:`minimize` and
   :meth:`Minimizer.minimize` that can hold all of the information
   about a fit. By having the return value be an instance of the
   :class:`MinimizerResult` class, it can hold an arbitrary amount of
   information that is easily accessed by attribute name, and even
   be given methods. Using objects is good!

2. To limit or even eliminate the amount of "state information" a
   :class:`Minimizer` holds. By state information, we mean how much of
   the previous fit is remembered after a fit is done. Keeping (and
   especially using) such information about a previous fit means that
   a :class:`Minimizer` might give different results even for the same
   problem if run a second time. While it's desirable to be able to
   adjust a set of :class:`Parameters` re-run a fit to get an improved
   result, doing this by changing an internal attribute
   (:attr:`Minimizer.params`) has the undesirable side-effect of not
   being able to "go back", and makes it somewhat cumbersome to keep
   track of changes made while adjusting parameters and re-running fits.
