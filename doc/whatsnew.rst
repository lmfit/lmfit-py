.. _whatsnew_chapter:

=============
Release Notes
=============

.. _lmfit GitHub repository: https://github.com/lmfit/lmfit-py

This section discusses changes between versions, especially changes
significant to the use and behavior of the library. This is not meant
to be a comprehensive list of changes. For such a complete record,
consult the `lmfit GitHub repository`_.

.. _whatsnew_122_label:

Version 1.2.2 Release Notes (July 14, 2023)
=================================================

New features:

- add ``ModelResult.uvars`` output to a ``ModelResult`` after a successful fit
  that contains ``ufloats`` from the ``uncertainties`` package which can be
  used for downstream calculations that propagate the uncertainties (and
  correlations) of the variable Parameters. (PR #888)

- Outputs of residual functions, including ``Model._residual``, are more
  explicitly coerced to 1d-arrays of datatype Float64.  This decreases the
  expectation for the user-supplied code to return ndarrays, and increases the
  tolerance for more "array-like" objects or ndarrays that are not Float64 or
  1-dimensional. (PR #899)

- ``Model.fit`` now takes a ``coerce_farray`` option, defaulting to ``True`` to
  control whether to input data and independent variables that are "array-like"
  are coerced to ndarrays of datatype Float64 or Complex128.  If set to
  ``False`` then independent data that "array-like" (``pandas.Series``, int32
  arrays, etc) will be sent to the model function unaltered. The user may then
  use other features of these objects, but may also need to explicitly coerce
  the datatype of the result the change described above about coercing the
  result causes problems. (Discussion #873; PR #899)

Bug fixes/enhancements:

- fixed bug in ``Model.make_params()`` for non-composite models that use a
  prefix (Discussion #892; Issue #893; PR #895)

- fixed bug with aborted fits for several methods having incorrect or invalid
  fit statistics. (Discussion #894; Issue #896; PR #897)

- ``Model.eval_uncertainty`` now correctly calculates complex (real/imaginary pairs)
  uncertainties for Models that generate complex results. (Issue #900; PR #901)

- ``Model.eval`` now returns and array-like value. This adds to the coercion
  features above and fixes a bug for composite models that return lists (Issue #875; PR #901)

- the HTML representation for a ``ModelResult`` or ``MinimizerResult`` are
  improved, and create fewer entries in the Table of Contents for Jupyter lab.
  (Issue #884; PR #883; PR #902)




.. _whatsnew_121_label:

Version 1.2.1 Release Notes (May 02, 2023)
=================================================

Bug fixes/enhancements:

- fixed bug in ``Model.make_params()`` for initial parameter values that were
  not recognized as floats such as ``np.Int64``.  (Issue #871; PR #872)

- explicitly set ``maxfun`` for ``l-bfgs-b`` method when setting
  ``maxiter``. (Issue #864; Discussion #865; PR #866)

.. _whatsnew_120_label:

Version 1.2.0 Release Notes (April 05, 2023)
=================================================

New features:

- add ``create_params`` function (PR #844)
- add ``chi2_out`` and ``nsigma`` options to ``conf_interval2d()``
- add ``ModelResult.summary()`` to return many resulting fit statistics and attributes into a JSON-able dict.
- add ``correl_table()`` function to ``lmfit.printfuncs`` and ``correl_mode`` option to ``fit_report()`` and
  ``ModelResult.fit_report()`` to optionally display a RST-formatted table of a correlation matrix.

Bug fixes/enhancements:

- fix bug when setting ``param.vary=True`` for a constrained parameter (Issue #859; PR #860)
- fix bug in reported uncertainties for constrained parameters by better propagating uncertainties (Issue #855; PR #856)
- Coercing of user input data and independent data for ``Model`` to float64 ndarrays is somewhat less aggressive and
  will not increase the precision of numpy ndarrays (see :ref:`model_data_coercion_section` for details). The resulting
  calculation from a model or objective function is more aggressively coerced to float64.  (Issue #850; PR #853)
- the default value of ``epsfcn`` is increased to 1.e-10 to allow for handling of data with precision less than float64
  (Issue #850; PR #853)
- fix ``conf_interval2d`` to use "increase chi-square by sigma**2*reduced chi-square" to give the ``sigma``-level
  probabilities (Issue #848; PR #852)
- fix reading of older ``ModelResult`` (Issue #845; included in PR #844)
- fix deepcopy of ``Parameters`` and user data (mguhyo; PR #837)
- improve ``Model.make_params`` and ``create_params`` to take optional dict of Parameter attributes (PR #844)
- fix reporting of ``nfev`` from ``least_squares`` to better reflect actual number of function calls (Issue #842; PR #844)
- fix bug in ``Model.eval`` when mixing parameters and keyword arguments (PR #844, #839)
- re-adds ``residual`` to saved ``Model`` result (PR #844, #830)
- ``ConstantModel`` and ``ComplexConstantModel`` will return an ndarray of the same shape as the independent variable
  ``x`` (JeppeKlitgaard, Issue #840; PR #841)
- update tests for latest versions of NumPy and SciPy.
- many fixes of doc typos and updates of dependencies, pre-commit hooks, and CI.

.. _whatsnew_110_label:

Version 1.1.0 Release Notes (November 27, 2022)
=================================================

New features:

- add ``Pearson4Model`` (@lellid; PR #800)
- add ``SplineModel`` (PR #804)
- add R^2 ``rsquared`` statistic to fit outputs and reports for Model fits (Issue #803; PR #810)
- add calculation of ``dely`` for model components of composite models (Issue #761; PR #826)

Bug fixes/enhancements:

- make sure variable ``spercent`` is always defined in ``params_html_table`` functions (reported by @MySlientWind; Issue #768, PR #770)
- always initialize the variables ``success`` and ``covar`` the ``MinimizerResult`` (reported by Marc W. Pound; PR #771)
- build package following PEP517/PEP518; use ``pyproject.toml`` and ``setup.cfg``; leave ``setup.py`` for now (PR #777)
- components used to create a ``CompositeModel`` can now have different independent variables (@Julian-Hochhaus; Discussion #787; PR #788)
- fixed function definition for ``StepModel(form='linear')``, was not consistent with the other ones (@matpompili; PR #794)
- fixed height factor for ``Gaussian2dModel``, was not correct (@matpompili; PR #795)
- for covariances with negative diagonal elements, we set the covariance to ``None`` (PR #813)
- fixed linear mode for ``RectangleModel`` (@arunpersaud; Issue #815; PR #816)
- report correct initial values for parameters with bounds (Issue #820; PR #821)
- allow recalculation of confidence intervals (@jagerber48; PR #798)
- include 'residual' in JSON output of ModelResult.dumps (@mac01021; PR #830)
- supports and is tested against Python 3.11; updated minimum required version of SciPy, NumPy, and asteval (PR #832)

Deprecations:

- remove support for Python 3.6 which reached EOL on 2021-12-23 (PR #790)


.. _whatsnew_103_label:

Version 1.0.3 Release Notes (October 14, 2021)
==============================================

Potentially breaking change:

- argument ``x`` is now required for the ``guess`` method of Models (Issue #747; PR #748)

To get reasonable estimates for starting values one should always supply both ``x`` and ``y`` values; in some cases it would work
when only providing ``data`` (i.e., y-values). With the change above, ``x`` is now required in the ``guess`` method call, so scripts might
need to be updated to explicitly supply ``x``.

Bug fixes/enhancements:

- do not overwrite user-specified figure titles in Model.plot() functions and allow setting with ``title`` keyword argument (PR #711)
- preserve Parameters subclass in deepcopy (@jenshnielsen; PR #719)
- coerce ``data`` and ``indepdent_vars`` to NumPy array with ``dtype=float64`` or ``dtype=complex128`` where applicable (Issues #723 and #728)
- fix collision between parameter names in built-in models and user-specified parameters (Issue #710 and PR #732)
- correct error message in PolynomialModel (@kremeyer; PR #737)
- improved handling of altered JSON data (Issue #739; PR #740, reported by Matthew Giammar)
- map ``max_nfev`` to ``maxiter`` when using ``differential_evolution`` (PR #749, reported by Olivier B.)
- correct use of noise versus experimental uncertainty in the documentation (PR #751, reported by Andrés Zelcer)
- specify return type of ``eval`` method more precisely and allow for plotting of (Complex)ConstantModel by coercing their
  ``float``, ``int``, or ``complex`` return value to a ``numpy.ndarray`` (Issue #684 and PR #754)
- fix ``dho`` (Damped Harmonic Oscillator) lineshape (PR #755; @rayosborn)
- reset ``Minimizer._abort`` to ``False`` before starting a new fit (Issue #756 and PR #757; @azelcer)
- fix typo in ``guess_from_peak2d`` (@ivan-usovl; PR #758)

Various:

- update asteval dependency to >= 0.9.22 to avoid DeprecationWarnings from NumPy v1.20.0 (PR #707)
- remove incorrectly spelled ``DonaichModel`` and ``donaich`` lineshape, deprecated in version 1.0.1 (PR #707)
- remove occurrences of OrderedDict throughout the code; dict is order-preserving since Python 3.6 (PR #713)
- update the contributing instructions (PR #718; @martin-majlis)
- (again) defer import of matplotlib to when it is needed (@zobristnicholas; PR #721)
- fix description of ``name`` argument in ``Parameters.add`` (@kristianmeyerr; PR #725)
- update dependencies, make sure a functional development environment is installed on Windows (Issue #712)
- use ``setuptools_scm`` for version info instead of ``versioneer`` (PR #729)
- transition to using ``f-strings`` (PR #730)
- mark ``test_manypeaks_speed.py`` as flaky to avoid intermittent test failures (repeat up to 5 times; PR #745)
- update scipy dependency to >= 1.14.0 (PR #751)
- improvement to output of examples in sphinx-gallery and use higher resolution figures (PR #753)
- remove deprecated functions ``lmfit.printfuncs.report_errors`` and ``asteval`` argument in ``Parameters`` class (PR #759)


.. _whatsnew_102_label:

Version 1.0.2 Release Notes (February 7, 2021)
==============================================

Version 1.0.2 officially supports Python 3.9 and has dropped support for Python 3.5. The minimum version
of the following dependencies were updated: asteval>=0.9.21, numpy>=1.18, and scipy>=1.3.

New features:

- added two-dimensional Gaussian lineshape and model (PR #642; @mpmdean)
- all built-in models are now registered in ``lmfit.models.lmfit_models``; new Model class attribute ``valid_forms`` (PR #663; @rayosborn)
- added a SineModel (PR #676; @lneuhaus)
- add the ``run_mcmc_kwargs argument`` to ``Minimizer.emcee`` to pass to the ``emcee.EnsembleSampler.run_mcmc`` function (PR #694; @rbnvrw)

Bug fixes:

- ``ModelResult.eval_uncertainty`` should use provided Parameters (PR #646)
- center in lognormal model can be negative (Issue #644, PR #645; @YoshieraHuang)
- restore best-fit values after calculation of covariance matrix (Issue #655, PR #657)
- add helper-function ``not_zero`` to prevent ZeroDivisionError in lineshapes and use in exponential lineshape (Issue #631, PR #664; @s-weigand)
- save ``last_internal_values`` and use to restore internal values if fit is aborted (PR #667)
- dumping a fit using the ``lbfgsb`` method now works, convert bytes to string if needed (Issue #677, PR #678; @leonfoks)
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
- relax module name check in ``test_check_ast_errors`` for Python 3.9 (Issue #674, PR #675; @mwhudson)
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
- introduced a new argument ``max_nfev`` to uniformly specify the maximum number of function evaluations (PR #610)
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
