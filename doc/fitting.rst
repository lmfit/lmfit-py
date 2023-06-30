.. _minimize_chapter:

.. module:: lmfit.minimizer

=====================================
Performing Fits and Analyzing Outputs
=====================================

As shown in the previous chapter, a simple fit can be performed with the
:func:`minimize` function. For more sophisticated modeling, the
:class:`Minimizer` class can be used to gain a bit more control, especially
when using complicated constraints or comparing results from related fits.

The :func:`minimize` function
=============================

The :func:`minimize` function is a wrapper around :class:`Minimizer` for
running an optimization problem. It takes an objective function (the
function that calculates the array to be minimized), a :class:`Parameters`
object, and several optional arguments. See :ref:`fit-func-label` for
details on writing the objective function.

.. autofunction:: minimize

..  _fit-func-label:

Writing a Fitting Function
==========================

An important component of a fit is writing a function to be minimized --
the *objective function*. Since this function will be called by other
routines, there are fairly stringent requirements for its call signature
and return value. In principle, your function can be any Python callable,
but it must look like this:

.. function:: func(params, *args, **kws):

   Calculate objective residual to be minimized from parameters.

   :param params: Parameters.
   :type  params: :class:`~lmfit.parameter.Parameters`
   :param args:  Positional arguments. Must match ``args`` argument to :func:`minimize`.
   :param kws:   Keyword arguments. Must match ``kws`` argument to :func:`minimize`.
   :return: Residual array (generally ``data-model``) to be minimized in the least-squares sense.
   :rtype: :numpydoc:`ndarray`. The length of this array cannot change between calls.


A common use for the positional and keyword arguments would be to pass in other
data needed to calculate the residual, including things as the data array,
dependent variable, uncertainties in the data, and other data structures for the
model calculation.

The objective function should return the value to be minimized. For the
Levenberg-Marquardt algorithm from :meth:`leastsq`, this returned value **must** be an
array, with a length greater than or equal to the number of fitting variables in the
model. For the other methods, the return value can either be a scalar or an array. If an
array is returned, the sum of squares of the array will be sent to the underlying fitting
method, effectively doing a least-squares optimization of the return values.

Since the function will be passed in a dictionary of :class:`Parameters`, it is advisable
to unpack these to get numerical values at the top of the function. A
simple way to do this is with :meth:`Parameters.valuesdict`, as shown below:

.. jupyter-execute::

    from numpy import exp, sign, sin, pi


    def residual(pars, x, data=None, eps=None):
        # unpack parameters: extract .value attribute for each parameter
        parvals = pars.valuesdict()
        period = parvals['period']
        shift = parvals['shift']
        decay = parvals['decay']

        if abs(shift) > pi/2:
            shift = shift - sign(shift)*pi

        if abs(period) < 1.e-10:
            period = sign(period)*1.e-10

        model = parvals['amp'] * sin(shift + x/period) * exp(-x*x*decay*decay)

        if data is None:
            return model
        if eps is None:
            return model - data
        return (model-data) / eps

In this example, ``x`` is a positional (required) argument, while the
``data`` array is actually optional (so that the function returns the model
calculation if the data is neglected). Also note that the model
calculation will divide ``x`` by the value of the ``period`` Parameter. It
might be wise to ensure this parameter cannot be 0. It would be possible
to use bounds on the :class:`Parameter` to do this:

.. jupyter-execute::
    :hide-code:

    from lmfit import Parameter, Parameters

    params =  Parameters()

.. jupyter-execute::

    params['period'] = Parameter(name='period', value=2, min=1.e-10)

but putting this directly in the function with:

.. jupyter-execute::
    :hide-code:

    period = 1

.. jupyter-execute::

    if abs(period) < 1.e-10:
        period = sign(period)*1.e-10

is also a reasonable approach. Similarly, one could place bounds on the
``decay`` parameter to take values only between ``-pi/2`` and ``pi/2``.

..  _fit-data-label:

Types of Data to Use for Fitting
===================================

Minimization methods assume that data is numerical.  For all the fitting
methods supported by lmfit, data and fitting parameters are also assumed to
be continuous variables.  As the routines make heavy use of numpy and scipy,
the most natural data to use in fitting is then numpy nd-arrays.  In fact, many
of the underlying fitting algorithms - including the default :meth:`leastsq`
method - **require** the values in the residual array used for the
minimization to be a 1-dimensional numpy array with data type (`dtype`) of
"float64": a 64-bit representation of a floating point number (sometimes called
a "double precision float").

Python is generally forgiving about data types, and in the scientific Python
community there is a concept of an object being "array like" which essentially
means that the can usually be coerced or interpreted as a numpy array, often
with that object having an ``__array__()`` method specially designed for that
conversion.  Important examples of objects that can be considered "array like"
include Lists and Tuples that contain only numbers, pandas Series, and HDF5
Datasets. Many objects from data-processing libraries like dask, xarray, zarr,
and more are also "array like".

Lmfit tries to be accommodating in the data that can be used in the fitting
process. When using :class:`Minimizer`, the data you pass in as extra arrays for the
calculation of the residual array will not be altered, and can be used in your
objective function in whatever form you send.  Usually, "array like" data will
work, but some care may be needed.  In the example above, if ``x`` was not a
numpy array but a list of numbers, this would give an error message like::

   TypeError: unsupported operand type(s) for /: 'list' and 'float'

or::

  TypeError: can't multiply sequence by non-int of type 'float'

because a list of numbers is only sometimes "array like".

Sending in a "more array-like" object like a pandas Series will avoid many
(though maybe not all!) such exceptions, but the resulting calculation returned
from the function would then also be a pandas Series.  Lmfit :meth:`minimize` will
always coerce the return value from the objective function into a 1-D numpy
array with ``dtype`` of "float64".  This will usually "just work", but there
may be exceptions.

When in doubt, or if running it trouble, converting data to float64 numpy
arrays before being used in a fit is recommended.  If using complex data or
functions, a ``dtype`` of "complex128" will also always work, and will be
converted to "float64" with ``ndaarray.view("float64")``.  Numpy arrays of other
``dtype`` (say, "int16" or "float32") should be used with caution.  In
particular, "float32" data should be avoided: Multiplying a "float32" array and
a Python float will result in a "float32" array for example.  As fitting
variables may have small changes made to them, the results may be at or below
"float32" precision, which will cause the fit to give up.  For integer data,
results are more sometimes promoted to "float64", but many numpy ufuncs (say,
``numpy.exp()``) will promote only to "float32", so care is still needed.


See also :ref:`model_data_coercion_section` for discussion of data passed in for
curve-fitting.



..  _fit-methods-label:

Choosing Different Fitting Methods
==================================

By default, the `Levenberg-Marquardt
<https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_ algorithm is
used for fitting. While often criticized, including the fact it finds a
*local* minimum, this approach has some distinct advantages. These include
being fast, and well-behaved for most curve-fitting needs, and making it
easy to estimate uncertainties for and correlations between pairs of fit
variables, as discussed in :ref:`fit-results-label`.

Alternative algorithms can also be used by providing the ``method``
keyword to the :func:`minimize` function or :meth:`Minimizer.minimize`
class as listed in the :ref:`Table of Supported Fitting Methods
<fit-methods-table>`. If you have the ``numdifftools`` package installed, lmfit
will try to estimate the covariance matrix and determine parameter
uncertainties and correlations if ``calc_covar`` is ``True`` (default).

.. _fit-methods-table:

 Table of Supported Fitting Methods:

 +--------------------------+------------------------------------------------------------------+
 | Fitting Method           | ``method`` arg to :func:`minimize` or :meth:`Minimizer.minimize` |
 +==========================+==================================================================+
 | Levenberg-Marquardt      |  ``leastsq`` or ``least_squares``                                |
 +--------------------------+------------------------------------------------------------------+
 | Nelder-Mead              |  ``nelder``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | L-BFGS-B                 |  ``lbfgsb``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | Powell                   |  ``powell``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | Conjugate Gradient       |  ``cg``                                                          |
 +--------------------------+------------------------------------------------------------------+
 | Newton-CG                |  ``newton``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | COBYLA                   |  ``cobyla``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | BFGS                     |  ``bfgsb``                                                       |
 +--------------------------+------------------------------------------------------------------+
 | Truncated Newton         |  ``tnc``                                                         |
 +--------------------------+------------------------------------------------------------------+
 | Newton CG trust-region   |  ``trust-ncg``                                                   |
 +--------------------------+------------------------------------------------------------------+
 | Exact trust-region       |  ``trust-exact``                                                 |
 +--------------------------+------------------------------------------------------------------+
 | Newton GLTR trust-region |  ``trust-krylov``                                                |
 +--------------------------+------------------------------------------------------------------+
 | Constrained trust-region |  ``trust-constr``                                                |
 +--------------------------+------------------------------------------------------------------+
 | Dogleg                   |  ``dogleg``                                                      |
 +--------------------------+------------------------------------------------------------------+
 | Sequential Linear        |  ``slsqp``                                                       |
 | Squares Programming      |                                                                  |
 +--------------------------+------------------------------------------------------------------+
 | Differential             |  ``differential_evolution``                                      |
 | Evolution                |                                                                  |
 +--------------------------+------------------------------------------------------------------+
 | Brute force method       |  ``brute``                                                       |
 +--------------------------+------------------------------------------------------------------+
 | Basinhopping             |  ``basinhopping``                                                |
 +--------------------------+------------------------------------------------------------------+
 | Adaptive Memory          |  ``ampgo``                                                       |
 | Programming for Global   |                                                                  |
 | Optimization             |                                                                  |
 +--------------------------+------------------------------------------------------------------+
 | Simplicial Homology      |  ``shgo``                                                        |
 | Global Optimization      |                                                                  |
 +--------------------------+------------------------------------------------------------------+
 | Dual Annealing           |  ``dual_annealing``                                              |
 +--------------------------+------------------------------------------------------------------+
 | Maximum likelihood via   |  ``emcee``                                                       |
 | Monte-Carlo Markov Chain |                                                                  |
 +--------------------------+------------------------------------------------------------------+


.. note::

   The objective function for the Levenberg-Marquardt method **must**
   return an array, with more elements than variables. All other methods
   can return either a scalar value or an array. The Monte-Carlo Markov
   Chain or ``emcee`` method has two different operating methods when the
   objective function returns a scalar value. See the documentation for ``emcee``.


.. warning::

  Much of this documentation assumes that the Levenberg-Marquardt (``leastsq``)
  method is used. Many of the fit statistics and estimates for uncertainties in
  parameters discussed in :ref:`fit-results-label` are done only unconditionally
  for this (and the ``least_squares``) method. Lmfit versions newer than 0.9.11
  provide the capability to use ``numdifftools`` to estimate the covariance matrix
  and calculate parameter uncertainties and correlations for other methods as
  well.

..  _fit-results-label:

:class:`MinimizerResult` -- the optimization result
===================================================

An optimization with :func:`minimize` or :meth:`Minimizer.minimize`
will return a :class:`MinimizerResult` object. This is an otherwise
plain container object (that is, with no methods of its own) that
simply holds the results of the minimization. These results will
include several pieces of informational data such as status and error
messages, fit statistics, and the updated parameters themselves.

Importantly, the parameters passed in to :meth:`Minimizer.minimize`
will be not be changed. To find the best-fit values, uncertainties
and so on for each parameter, one must use the
:attr:`MinimizerResult.params` attribute. For example, to print the
fitted values, bounds and other parameter attributes in a
well-formatted text tables you can execute::

    result.params.pretty_print()

with ``results`` being a ``MinimizerResult`` object. Note that the method
:meth:`~lmfit.parameter.Parameters.pretty_print` accepts several arguments
for customizing the output (e.g., column width, numeric format, etcetera).

.. autoclass:: MinimizerResult


Goodness-of-Fit Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _goodfit-table:

 Table of Fit Results: These values, including the standard Goodness-of-Fit statistics,
 are all attributes of the :class:`MinimizerResult` object returned by
 :func:`minimize` or :meth:`Minimizer.minimize`.

+----------------------+----------------------------------------------------------------------------+
| Attribute Name       | Description / Formula                                                      |
+======================+============================================================================+
|    nfev              | number of function evaluations                                             |
+----------------------+----------------------------------------------------------------------------+
|    nvarys            | number of variables in fit :math:`N_{\rm varys}`                           |
+----------------------+----------------------------------------------------------------------------+
|    ndata             | number of data points: :math:`N`                                           |
+----------------------+----------------------------------------------------------------------------+
|    nfree             | degrees of freedom in fit: :math:`N - N_{\rm varys}`                       |
+----------------------+----------------------------------------------------------------------------+
|    residual          | residual array, returned by the objective function: :math:`\{\rm Resid_i\}`|
+----------------------+----------------------------------------------------------------------------+
|    chisqr            | chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`                    |
+----------------------+----------------------------------------------------------------------------+
|    redchi            | reduced chi-square: :math:`\chi^2_{\nu}= {\chi^2} / {(N - N_{\rm varys})}` |
+----------------------+----------------------------------------------------------------------------+
|    aic               | Akaike Information Criterion statistic (see below)                         |
+----------------------+----------------------------------------------------------------------------+
|    bic               | Bayesian Information Criterion statistic (see below)                       |
+----------------------+----------------------------------------------------------------------------+
|    var_names         | ordered list of variable parameter names used for init_vals and covar      |
+----------------------+----------------------------------------------------------------------------+
|    covar             | covariance matrix (with rows/columns using var_names)                      |
+----------------------+----------------------------------------------------------------------------+
|    init_vals         | list of initial values for variable parameters                             |
+----------------------+----------------------------------------------------------------------------+
|    call_kws          | dict of keyword arguments sent to underlying solver                        |
+----------------------+----------------------------------------------------------------------------+

Note that the calculation of chi-square and reduced chi-square assume
that the returned residual function is scaled properly to the
uncertainties in the data. For these statistics to be meaningful, the
person writing the function to be minimized **must** scale them properly.

After a fit using the :meth:`leastsq` or :meth:`least_squares` method has
completed successfully, standard errors for the fitted variables and
correlations between pairs of fitted variables are automatically calculated from
the covariance matrix. For other methods, the ``calc_covar`` parameter (default
is ``True``) in the :class:`Minimizer` class determines whether or not to use the
``numdifftools`` package to estimate the covariance matrix. The standard error
(estimated :math:`1\sigma` error-bar) goes into the :attr:`stderr` attribute of
the Parameter. The correlations with all other variables will be put into the
:attr:`correl` attribute of the Parameter -- a dictionary with keys for all
other Parameters and values of the corresponding correlation.

In some cases, it may not be possible to estimate the errors and
correlations. For example, if a variable actually has no practical effect
on the fit, it will likely cause the covariance matrix to be singular,
making standard errors impossible to estimate. Placing bounds on varied
Parameters makes it more likely that errors cannot be estimated, as being
near the maximum or minimum value makes the covariance matrix singular. In
these cases, the :attr:`errorbars` attribute of the fit result
(:class:`Minimizer` object) will be ``False``.


.. _information_criteria_label:

Akaike and Bayesian Information Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`MinimizerResult` includes the traditional chi-square and
reduced chi-square statistics:

.. math::
   :nowrap:

   \begin{eqnarray*}
        \chi^2  &=&  \sum_i^N r_i^2 \\
        \chi^2_\nu &=& \chi^2 / (N-N_{\rm varys})
    \end{eqnarray*}

where :math:`r` is the residual array returned by the objective function
(likely to be ``(data-model)/uncertainty`` for data modeling usages),
:math:`N` is the number of data points (``ndata``), and :math:`N_{\rm
varys}` is number of variable parameters.

Also included are the `Akaike Information Criterion
<https://en.wikipedia.org/wiki/Akaike_information_criterion>`_, and
`Bayesian Information Criterion
<https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ statistics,
held in the ``aic`` and ``bic`` attributes, respectively. These give slightly
different measures of the relative quality for a fit, trying to balance
quality of fit with the number of variable parameters used in the fit.
These are calculated as:

.. math::
   :nowrap:

   \begin{eqnarray*}
     {\rm aic} &=&  N \ln(\chi^2/N) + 2 N_{\rm varys} \\
     {\rm bic} &=&  N \ln(\chi^2/N) + \ln(N) N_{\rm varys} \\
    \end{eqnarray*}


When comparing fits with different numbers of varying parameters, one
typically selects the model with lowest reduced chi-square, Akaike
information criterion, and/or Bayesian information criterion. Generally,
the Bayesian information criterion is considered the most conservative of
these statistics.


Uncertainties in Variable Parameters, and their Correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, when a fit is complete the uncertainties for fitted
Parameters as well as the correlations between pairs of Parameters are
usually calculated. This happens automatically either when using the
default :meth:`leastsq` method, the :meth:`least_squares` method, or for
most other fitting methods if the highly-recommended ``numdifftools``
package is available. The estimated standard error (the :math:`1\sigma`
uncertainty) for each variable Parameter will be contained in the
:attr:`stderr`, while the :attr:`correl` attribute for each Parameter will
contain a dictionary of the correlation with each other variable Parameter.

These estimates of the uncertainties are done by inverting the Hessian
matrix which represents the second derivative of fit quality for each
variable parameter. There are situations for which the uncertainties cannot
be estimated, which generally indicates that this matrix cannot be inverted
because one of the fit is not actually sensitive to one of the variables.
This can happen if a Parameter is stuck at an upper or lower bound, if the
variable is simply not used by the fit, or if the value for the variable is
such that it has no real influence on the fit.

In principle, the scale of the uncertainties in the Parameters is closely
tied to the goodness-of-fit statistics chi-square and reduced chi-square
(``chisqr`` and ``redchi``). The standard errors or :math:`1 \sigma`
uncertainties are those that increase chi-square by 1. Since a "good fit"
should have ``redchi`` of around 1, this requires that the data
uncertainties (and to some extent the sampling of the N data points) is
correct. Unfortunately, it is often not the case that one has high-quality
estimates of the data uncertainties (getting the data is hard enough!).
Because of this common situation, the uncertainties reported and held in
:attr:`stderr` are not those that increase chi-square by 1, but those that
increase chi-square by reduced chi-square. This is equivalent to rescaling
the uncertainty in the data such that reduced chi-square would be 1. To be
clear, this rescaling is done by default because if reduced chi-square is
far from 1, this rescaling often makes the reported uncertainties sensible,
and if reduced chi-square is near 1 it does little harm. If you have good
scaling of the data uncertainty and believe the scale of the residual
array is correct,  this automatic rescaling can be turned off using
``scale_covar=False``.

Note that the simple (and fast!) approach to estimating uncertainties and
correlations by inverting the second derivative matrix assumes that the
components of the residual array (if, indeed, an array is used) are
distributed around 0 with a normal (Gaussian distribution), and that a map
of probability distributions for pairs would be elliptical -- the size of
the of ellipse gives the uncertainty itself and the eccentricity of the
ellipse gives the correlation. This simple approach to assessing
uncertainties ignores outliers, highly asymmetric uncertainties, or complex
correlations between Parameters. In fact, it is not too hard to come up
with problems where such effects are important. Our experience is that the
automated results are usually the right scale and quite reasonable as
initial estimates, but a more thorough exploration of the Parameter space
using the tools described in :ref:`label-emcee` and
:ref:`label-confidence-advanced` can give a more complete understanding of
the distributions and relations between Parameters.


..  _fit-reports-label:

Getting and Printing Fit Reports
================================

.. currentmodule:: lmfit.printfuncs

.. autofunction:: fit_report

An example using this to write out a fit report would be:

.. jupyter-execute:: ../examples/doc_fitting_withreport.py
    :hide-output:

which would give as output:

.. jupyter-execute::
    :hide-code:

    print(fit_report(out))

To be clear, you can get at all of these values from the fit result ``out``
and ``out.params``. For example, a crude printout of the best fit variables
and standard errors could be done as

.. jupyter-execute::

    print('-------------------------------')
    print('Parameter    Value       Stderr')
    for name, param in out.params.items():
        print(f'{name:7s} {param.value:11.5f} {param.stderr:11.5f}')


..  _fit-itercb-label:

Using a Iteration Callback Function
===================================

.. currentmodule:: lmfit.minimizer

An iteration callback function is a function to be called at each
iteration, just after the objective function is called. The iteration
callback allows user-supplied code to be run at each iteration, and can
be used to abort a fit.

.. function:: iter_cb(params, iter, resid, *args, **kws):

   User-supplied function to be run at each iteration.

   :param params: Parameters.
   :type  params: :class:`~lmfit.parameter.Parameters`
   :param iter:   Iteration number.
   :type  iter:   int
   :param resid:  Residual array.
   :type  resid:  numpy.ndarray
   :param args:  Positional arguments. Must match ``args`` argument to :func:`minimize`
   :param kws:   Keyword arguments. Must match ``kws`` argument to :func:`minimize`
   :return:      Iteration abort flag.
   :rtype:    None for normal behavior, any value like ``True`` to abort the fit.


Normally, the iteration callback would have no return value or return
``None``. To abort a fit, have this function return a value that is
``True`` (including any non-zero integer). The fit will also abort if any
exception is raised in the iteration callback. When a fit is aborted this
way, the parameters will have the values from the last iteration. The fit
statistics are not likely to be meaningful, and uncertainties will not be computed.


..  _fit-minimizer-label:

Using the :class:`Minimizer` class
==================================

.. currentmodule:: lmfit.minimizer

For full control of the fitting process, you will want to create a
:class:`Minimizer` object.

.. autoclass :: Minimizer

The Minimizer object has a few public methods:

.. automethod:: Minimizer.minimize

.. automethod:: Minimizer.leastsq

.. automethod:: Minimizer.least_squares

.. automethod:: Minimizer.scalar_minimize

.. automethod:: Minimizer.prepare_fit

.. automethod:: Minimizer.brute

For more information, check the examples in ``examples/lmfit_brute_example.ipynb``.

.. automethod:: Minimizer.basinhopping

.. automethod:: Minimizer.ampgo

.. automethod:: Minimizer.shgo

.. automethod:: Minimizer.dual_annealing

.. automethod:: Minimizer.emcee


.. _label-emcee:

:meth:`Minimizer.emcee` - calculating the posterior probability distribution of parameters
==========================================================================================

:meth:`Minimizer.emcee` can be used to obtain the posterior probability
distribution of parameters, given a set of experimental data. Note that this
method does *not* actually perform a fit at all. Instead, it explores
parameter space to determine the probability distributions for the parameters,
but without an explicit goal of attempting to refine the solution. It should
not be used for fitting, but it is a useful method to to more thoroughly
explore the parameter space around the solution after a fit has been done and
thereby get an improved understanding of the probability distribution for the
parameters. It may be able to refine your estimate of the most likely values
for a set of parameters, but it will not iteratively find a good solution to
the minimization problem. To use this method effectively, you should first
use another minimization method and then use this method to explore the
parameter space around those best-fit values.

To illustrate this, we'll use an example problem of fitting data to function
of a double exponential decay, including a modest amount of Gaussian noise to
the data. Note that this example is the same problem used in
:ref:`label-confidence-advanced` for evaluating confidence intervals in the
parameters, which is a similar goal to the one here.

.. jupyter-execute::
    :hide-code:

    import warnings
    warnings.filterwarnings(action="ignore")

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.dpi'] = 150
    %matplotlib inline
    %config InlineBackend.figure_format = 'svg'


.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np

    import lmfit

    x = np.linspace(1, 10, 250)
    np.random.seed(0)
    y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 * np.random.randn(x.size)

Create a Parameter set for the initial guesses:

.. jupyter-execute::

    p = lmfit.Parameters()
    p.add_many(('a1', 4.), ('a2', 4.), ('t1', 3.), ('t2', 3., True))

    def residual(p):
        v = p.valuesdict()
        return v['a1'] * np.exp(-x / v['t1']) + v['a2'] * np.exp(-(x - 0.1) / v['t2']) - y

Solving with :func:`minimize` gives the Maximum Likelihood solution. Note
that we use the robust Nelder-Mead method here. The default Levenberg-Marquardt
method seems to have difficulty with exponential decays, though it can refine
the solution if starting near the solution:

.. jupyter-execute::

    mi = lmfit.minimize(residual, p, method='nelder', nan_policy='omit')
    lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)

and plotting the fit using the Maximum Likelihood solution gives the graph below:

.. jupyter-execute::

    plt.plot(x, y, 'o')
    plt.plot(x, residual(mi.params) + y, label='best fit')
    plt.legend()
    plt.show()

Note that the fit here (for which the ``numdifftools`` package is installed)
does estimate and report uncertainties in the parameters and correlations for
the parameters, and reports the correlation of parameters ``a2`` and ``t2`` to
be very high. As we'll see, these estimates are pretty good, but when faced
with such high correlation, it can be helpful to get the full probability
distribution for the parameters. MCMC methods are very good for this.

Furthermore, we wish to deal with the data uncertainty. This is called
marginalisation of a nuisance parameter. ``emcee`` requires a function that
returns the log-posterior probability. The log-posterior probability is a sum
of the log-prior probability and log-likelihood functions. The log-prior
probability is assumed to be zero if all the parameters are within their
bounds and ``-np.inf`` if any of the parameters are outside their bounds.

If the objective function returns an array of unweighted residuals (i.e.,
``data-model``) as is the case here, you can use ``is_weighted=False`` as an
argument for ``emcee``. In that case, ``emcee`` will automatically add/use the
``__lnsigma`` parameter to estimate the true uncertainty in the data. To
place boundaries on this parameter one can do:

.. jupyter-execute::

    mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

Now we have to set up the minimizer and do the sampling (again, just to be
clear, this is *not* doing a fit):

.. jupyter-execute::
    :hide-output:

    res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,
                         params=mi.params, is_weighted=False, progress=False)

As mentioned in the Notes for :meth:`Minimizer.emcee`, the ``is_weighted``
argument will be ignored if your objective function returns a float instead of
an array. For the documentation we set ``progress=False``; the default is to
print a progress bar to the Terminal if the ``tqdm`` package is installed.

The success of the method (i.e., whether or not the sampling went well) can be
assessed by checking the integrated autocorrelation time and/or the acceptance
fraction of the walkers. For this specific example the autocorrelation time
could not be estimated because the "chain is too short". Instead, we plot the
acceptance fraction per walker and its mean value suggests that the sampling
worked as intended (as a rule of thumb the value should be between 0.2 and
0.5).

.. jupyter-execute::

    plt.plot(res.acceptance_fraction, 'o')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    plt.show()

With the results from ``emcee``, we can visualize the posterior distributions
for the parameters using the ``corner`` package:

.. jupyter-execute::

    import corner

    emcee_plot = corner.corner(res.flatchain, labels=res.var_names,
                               truths=list(res.params.valuesdict().values()))

The values reported in the :class:`MinimizerResult` are the medians of the
probability distributions and a 1 :math:`\sigma` quantile, estimated as half
the difference between the 15.8 and 84.2 percentiles. Printing these values:


.. jupyter-execute::

    print('median of posterior probability distribution')
    print('--------------------------------------------')
    lmfit.report_fit(res.params)

You can see that this recovered the right uncertainty level on the data. Note
that these values agree pretty well with the results, uncertainties and
correlations found by the fit and using ``numdifftools`` to estimate the
covariance matrix. That is, even though the parameters ``a2``, ``t1``, and
``t2`` are all highly correlated and do not display perfectly Gaussian
probability distributions, the probability distributions found by explicitly
sampling the parameter space are not so far from elliptical as to make the
simple (and much faster) estimates from inverting the covariance matrix
completely invalid.

As mentioned above, the result from ``emcee`` reports the median values, which
are not necessarily the same as the Maximum Likelihood Estimate. To obtain
the values for the Maximum Likelihood Estimation (MLE) we find the location in
the chain with the highest probability:

.. jupyter-execute::

    highest_prob = np.argmax(res.lnprob)
    hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
    mle_soln = res.chain[hp_loc]
    for i, par in enumerate(p):
        p[par].value = mle_soln[i]


    print('\nMaximum Likelihood Estimation from emcee       ')
    print('-------------------------------------------------')
    print('Parameter  MLE Value   Median Value   Uncertainty')
    fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
    for name, param in p.items():
        print(fmt(name, param.value, res.params[name].value,
                  res.params[name].stderr))


Here the difference between MLE and median value are seen to be below 0.5%,
and well within the estimated 1-:math:`\sigma` uncertainty.

Finally, we can use the samples from ``emcee`` to work out the 1- and
2-:math:`\sigma` error estimates.

.. jupyter-execute::

    print('\nError estimates from emcee:')
    print('------------------------------------------------------')
    print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')

    for name in p.keys():
        quantiles = np.percentile(res.flatchain[name],
                                  [2.275, 15.865, 50, 84.135, 97.275])
        median = quantiles[2]
        err_m2 = quantiles[0] - median
        err_m1 = quantiles[1] - median
        err_p1 = quantiles[3] - median
        err_p2 = quantiles[4] - median
        fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
        print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

And we see that the initial estimates for the 1-:math:`\sigma` standard error
using ``numdifftools`` was not too bad. We'll return to this example
problem in :ref:`label-confidence-advanced` and use a different method to
calculate the 1- and 2-:math:`\sigma` error bars.
