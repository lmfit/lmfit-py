.. _minimize_chapter:

=======================================
Performing Fits, Analyzing Outputs
=======================================

As shown in the previous chapter, a simple fit can be performed with the
:func:`minimize` function.  For more sophisticated modeling, the
:class:`Minimizer` class can be used to gain a bit more control, especially
when using complicated constraints or comparing results from related fits.


The :func:`minimize` function
===============================

The :func:`minimize` function is a wrapper around :class:`Minimizer` for
running an optimization problem.  It takes an objective function (the
function that calculates the array to be minimized), a :class:`Parameters`
object, and several optional arguments.  See :ref:`fit-func-label` for
details on writing the objective.

.. function:: minimize(function, params[, args=None[, kws=None[, method='leastsq'[, scale_covar=True[, iter_cb=None[, **fit_kws]]]]]])

   find values for the ``params`` so that the sum-of-squares of the array returned
   from ``function`` is minimized.

   :param function:  function to return fit residual.  See :ref:`fit-func-label` for details.
   :type  function:  callable.
   :param params:  a :class:`Parameters` dictionary.  Keywords must be strings
                   that match ``[a-z_][a-z0-9_]*`` and cannot be a python
                   reserved word.  Each value must be :class:`Parameter`.
   :type  params:  :class:`Parameters`.
   :param args:  arguments tuple to pass to the residual function as  positional arguments.
   :type  args:  tuple
   :param kws:   dictionary to pass to the residual function as keyword arguments.
   :type  kws:  dict
   :param method:  name of fitting method to use. See  :ref:`fit-methods-label` for details
   :type  method:  string (default ``leastsq``)
   :param scale_covar:  whether to automatically scale covariance matrix (``leastsq`` only)
   :type  scale_covar:  bool (default ``True``)
   :param iter_cb:  function to be called at each fit iteration
   :type  iter_cb:  callable or ``None``
   :param fit_kws:  dictionary to pass to :func:`scipy.optimize.leastsq` or :func:`scipy.optimize.minimize`.
   :type  fit_kws:  dict

   :return: :class:`MinimizerResult` instance, which will contain the
            optimized parameter, and several goodness-of-fit statistics.

   On output, the params will be unchanged.  The best-fit values, and where
   appropriate, estimated uncertainties and correlations, will all be
   contained in the returned :class:`MinimizerResult`.  See
   :ref:`fit-results-label` for further details.

   If provided, the ``iter_cb`` function should take arguments of ``params,
   iter, resid, *args, **kws``, where ``params`` will have the current
   parameter values, ``iter`` the iteration, ``resid`` the current residual
   array, and ``*args`` and ``**kws`` as passed to the objective function.

   For clarity, it should be emphasized that this function is simply a
   wrapper around :class:`Minimizer` that runs a single fit, implemented as::

    fitter = Minimizer(fcn, params, fcn_args=args, fcn_kws=kws,
                       iter_cb=iter_cb, scale_covar=scale_covar, **fit_kws)
    return fitter.minimize(method=method)


..  _fit-func-label:



Writing a Fitting Function
===============================

An important component of a fit is writing a function to be minimized --
the *objective function*.  Since this function will be called by other
routines, there are fairly stringent requirements for its call signature
and return value.  In principle, your function can be any python callable,
but it must look like this:

.. function:: func(params, *args, **kws):

   calculate objective residual to be minimized from parameters.

   :param params: parameters.
   :type  params: :class:`Parameters`.
   :param args:  positional arguments.  Must match ``args`` argument to :func:`minimize`
   :param kws:   keyword arguments.  Must match ``kws`` argument to :func:`minimize`
   :return: residual array (generally data-model) to be minimized in the least-squares sense.
   :rtype: numpy array.  The length of this array cannot change between calls.


A common use for the positional and keyword arguments would be to pass in other
data needed to calculate the residual, including such things as the data array,
dependent variable, uncertainties in the data, and other data structures for the
model calculation.

The objective function should return the value to be minimized.  For the
Levenberg-Marquardt algorithm from :meth:`leastsq`, this returned value **must** be an
array, with a length greater than or equal to the number of fitting variables in the
model.  For the other methods, the return value can either be a scalar or an array.  If an
array is returned, the sum of squares of the array will be sent to the underlying fitting
method, effectively doing a least-squares optimization of the return values.


Since the function will be passed in a dictionary of :class:`Parameters`, it is advisable
to unpack these to get numerical values at the top of the function.  A
simple way to do this is with :meth:`Parameters.valuesdict`, as with::


    def residual(pars, x, data=None, eps=None):
        # unpack parameters:
        #  extract .value attribute for each parameter
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
            return (model - data)
        return (model - data)/eps

In this example, ``x`` is a positional (required) argument, while the
``data`` array is actually optional (so that the function returns the model
calculation if the data is neglected).  Also note that the model
calculation will divide ``x`` by the value of the 'period' Parameter.  It
might be wise to ensure this parameter cannot be 0.  It would be possible
to use the bounds on the :class:`Parameter` to do this::

    params['period'] = Parameter(value=2, min=1.e-10)

but putting this directly in the function with::

        if abs(period) < 1.e-10:
            period = sign(period)*1.e-10

is also a reasonable approach.   Similarly, one could place bounds on the
``decay`` parameter to take values only between ``-pi/2`` and ``pi/2``.

..  _fit-methods-label:

Choosing Different Fitting Methods
===========================================

By default, the `Levenberg-Marquardt
<http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_ algorithm is
used for fitting.  While often criticized, including the fact it finds a
*local* minima, this approach has some distinct advantages.  These include
being fast, and well-behaved for most curve-fitting needs, and making it
easy to estimate uncertainties for and correlations between pairs of fit
variables, as discussed in :ref:`fit-results-label`.

Alternative algorithms can also be used by providing the ``method``
keyword to the :func:`minimize` function or :meth:`Minimizer.minimize`
class as listed in the :ref:`Table of Supported Fitting Methods
<fit-methods-table>`.

.. _fit-methods-table:

 Table of Supported Fitting Method, eithers:

 +-----------------------+------------------------------------------------------------------+
 | Fitting Method        | ``method`` arg to :func:`minimize` or :meth:`Minimizer.minimize` |
 +=======================+==================================================================+
 | Levenberg-Marquardt   |  ``leastsq``                                                     |
 +-----------------------+------------------------------------------------------------------+
 | Nelder-Mead           |  ``nelder``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | L-BFGS-B              |  ``lbfgsb``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | Powell                |  ``powell``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | Conjugate Gradient    |  ``cg``                                                          |
 +-----------------------+------------------------------------------------------------------+
 | Newton-CG             |  ``newton``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | COBYLA                |  ``cobyla``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | Truncated Newton      |  ``tnc``                                                         |
 +-----------------------+------------------------------------------------------------------+
 | Dogleg                |  ``dogleg``                                                      |
 +-----------------------+------------------------------------------------------------------+
 | Sequential Linear     |  ``slsqp``                                                       |
 | Squares Programming   |                                                                  |
 +-----------------------+------------------------------------------------------------------+
 | Differential          |  ``differential_evolution``                                      |
 | Evolution             |                                                                  |
 +-----------------------+------------------------------------------------------------------+


.. note::

   The objective function for the Levenberg-Marquardt method **must**
   return an array, with more elements than variables.  All other methods
   can return either a scalar value or an array.


.. warning::

  Much of this documentation assumes that the Levenberg-Marquardt method is
  the method used.  Many of the fit statistics and estimates for
  uncertainties in parameters discussed in :ref:`fit-results-label` are
  done only for this method.


..  _fit-results-label:

:class:`MinimizerResult` -- the optimization result
========================================================


.. class:: MinimizerResult(**kws)

An optimization with :func:`minimize` or :meth:`Minimizer.minimize`
will return a :class:`MinimizerResult` object.  This is an otherwise
plain container object (that is, with no methods of its own) that
simply holds the results of the minimization.  These results will
include several pieces of informational data such as status and error
messages, fit statistics, and the updated parameters themselves.

Importantly, the parameters passed in to :meth:`Minimizer.minimize`
will be not be changed.  To to find the best-fit values, uncertainties
and so on for each parameter, one must use the
:attr:`MinimizerResult.params` attribute.

.. attribute::   params

  the :class:`Parameters` actually used in the fit, with updated
  values, :attr:`stderr` and :attr:`correl`.

.. attribute::  var_names

  ordered list of variable parameter names used in optimization, and
  useful for understanding the the values in :attr:`init_vals` and
  :attr:`covar`.

.. attribute:: covar

  covariance matrix from minimization (`leastsq` only), with
  rows/columns using :attr:`var_names`.

.. attribute:: init_vals

  list of initial values for variable parameters using :attr:`var_names`.

.. attribute::  nfev

  number of function evaluations

.. attribute::  success

  boolean (``True``/``False``) for whether fit succeeded.

.. attribute::  errorbars

  boolean (``True``/``False``) for whether uncertainties were
  estimated.

.. attribute::  message

  message about fit success.

.. attribute::  ier

  integer error value from :func:`scipy.optimize.leastsq`  (`leastsq`
  only).

.. attribute::  lmdif_message

  message from :func:`scipy.optimize.leastsq` (`leastsq` only).


.. attribute::  nvarys

  number of variables in fit  :math:`N_{\rm varys}`

.. attribute::  ndata

  number of data points:  :math:`N`

.. attribute::  nfree `

  degrees of freedom in fit:  :math:`N - N_{\rm varys}`

.. attribute::  residual

  residual array, return value of :func:`func`:  :math:`{\rm Resid}`

.. attribute::  chisqr

  chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`

.. attribute::  redchi

  reduced chi-square: :math:`\chi^2_{\nu}= {\chi^2} / {(N - N_{\rm
  varys})}`

.. attribute::  aic

  Akaike Information Criterion statistic (see below)

.. attribute::  bic

  Bayesian Information Criterion statistic (see below).





Goodness-of-Fit Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _goodfit-table:

 Table of Fit Results:  These values, including the standard Goodness-of-Fit statistics,
 are all attributes of the :class:`MinimizerResult` object returned by
 :func:`minimize` or :meth:`Minimizer.minimize`.

+----------------------+----------------------------------------------------------------------------+
| Attribute Name       | Description / Formula                                                      |
+======================+============================================================================+
|    nfev              | number of function evaluations                                             |
+----------------------+----------------------------------------------------------------------------+
|    nvarys            | number of variables in fit  :math:`N_{\rm varys}`                          |
+----------------------+----------------------------------------------------------------------------+
|    ndata             | number of data points:  :math:`N`                                          |
+----------------------+----------------------------------------------------------------------------+
|    nfree `           | degrees of freedom in fit:  :math:`N - N_{\rm varys}`                      |
+----------------------+----------------------------------------------------------------------------+
|    residual          | residual array, return value of :func:`func`:  :math:`{\rm Resid}`         |
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
|    covar             | covariance matrix (with rows/columns using var_names                       |
+----------------------+----------------------------------------------------------------------------+
|    init_vals         | list of initial values for variable parameters                             |
+----------------------+----------------------------------------------------------------------------+

Note that the calculation of chi-square and reduced chi-square assume
that the returned residual function is scaled properly to the
uncertainties in the data.  For these statistics to be meaningful, the
person writing the function to be minimized must scale them properly.

After a fit using using the :meth:`leastsq` method has completed
successfully, standard errors for the fitted variables and correlations
between pairs of fitted variables are automatically calculated from the
covariance matrix.  The standard error (estimated :math:`1\sigma`
error-bar) go into the :attr:`stderr` attribute of the Parameter.  The
correlations with all other variables will be put into the
:attr:`correl` attribute of the Parameter -- a dictionary with keys for
all other Parameters and values of the corresponding correlation.

In some cases, it may not be possible to estimate the errors and
correlations.  For example, if a variable actually has no practical effect
on the fit, it will likely cause the covariance matrix to be singular,
making standard errors impossible to estimate.  Placing bounds on varied
Parameters makes it more likely that errors cannot be estimated, as being
near the maximum or minimum value makes the covariance matrix singular.  In
these cases, the :attr:`errorbars` attribute of the fit result
(:class:`Minimizer` object) will be ``False``.

Akaike and Bayesian Information Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`MinimizerResult` includes the tradtional chi-square and reduced chi-square statistics:

.. math::
   :nowrap:

   \begin{eqnarray*}
        \chi^2  &=&  \sum_i^N r_i^2 \\
	\chi^2_\nu &=& = \chi^2 / (N-N_{\rm varys})
    \end{eqnarray*}

where :math:`r` is the residual array returned by the objective function
(likely to be ``(data-model)/uncertainty`` for data modeling usages),
:math:`N` is the number of data points (``ndata``), and :math:`N_{\rm
varys}` is number of variable parameters.

Also included are the `Akaike Information Criterion
<http://en.wikipedia.org/wiki/Akaike_information_criterion>`_, and
`Bayesian Information Criterion
<http://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ statistics,
held in the ``aic`` and ``bic`` attributes, respectively.  These give slightly
different measures of the relative quality for a fit, trying to balance
quality of fit with the number of variable parameters used in the fit.
These are calculated as

.. math::
   :nowrap:

   \begin{eqnarray*}
     {\rm aic} &=&  N \ln(\chi^2/N) + 2 N_{\rm varys} \\
     {\rm bic} &=&  N \ln(\chi^2/N) + \ln(N) *N_{\rm varys} \\
    \end{eqnarray*}


Generally, when comparing fits with different numbers of varying
parameters, one typically selects the model with lowest reduced chi-square,
Akaike information criterion, and/or Bayesian information criterion.
Generally, the Bayesian information criterion is considered themost
conservative of these statistics.


.. module:: Minimizer

..  _fit-minimizer-label:

Using the :class:`Minimizer` class
=======================================

For full control of the fitting process, you'll want to create a
:class:`Minimizer` object.

.. class:: Minimizer(function, params, fcn_args=None, fcn_kws=None, iter_cb=None, scale_covar=True, **kws)

   creates a Minimizer, for more detailed access to fitting methods and attributes.

   :param function:  objective function to return fit residual.  See :ref:`fit-func-label` for details.
   :type  function:  callable.
   :param params:  a dictionary of Parameters.  Keywords must be strings
                   that match ``[a-z_][a-z0-9_]*`` and is not a python
                   reserved word.  Each value must be :class:`Parameter`.
   :type  params:  dict
   :param fcn_args:  arguments tuple to pass to the residual function as  positional arguments.
   :type  fcn_args: tuple
   :param fcn_kws:  dictionary to pass to the residual function as keyword arguments.
   :type  fcn_kws:  dict
   :param iter_cb:  function to be called at each fit iteration
   :type  iter_cb:  callable or ``None``
   :param scale_covar:  flag for automatically scaling covariance matrix and uncertainties to reduced chi-square (``leastsq`` only)
   :type  scale_cover:  bool (default ``True``).
   :param kws:      dictionary to pass as keywords to the underlying :mod:`scipy.optimize` method.
   :type  kws:      dict

The Minimizer object has a few public methods:

.. method:: minimize(method='leastsq', params=None, **kws)

   perform fit using either :meth:`leastsq` or :meth:`scalar_minimize`.

   :param method: name of fitting method.  Must be one of the naemes in
                  :ref:`Table of Supported Fitting Methods <fit-methods-table>`
   :type  method:  str.
   :param params:  a :class:`Parameters` dictionary for starting values
   :type  params:  :class:`Parameters` or `None`

   :return: :class:`MinimizerResult` object, containing updated
            parameters, fitting statistics, and information.

   Additonal keywords are passed on to the correspond :meth:`leastsq`
   or :meth:`scalar_minimize` method.

.. method:: leastsq(params=None, scale_covar=True, **kws)

   perform fit with Levenberg-Marquardt algorithm.  Keywords will be
   passed directly to :func:`scipy.optimize.leastsq`.  By default,
   numerical derivatives are used, and the following arguments are set:


    +------------------+----------------+------------------------------------------------------------+
    | :meth:`leastsq`  |  Default Value | Description                                                |
    | arg              |                |                                                            |
    +==================+================+============================================================+
    |   xtol           |  1.e-7         | Relative error in the approximate solution                 |
    +------------------+----------------+------------------------------------------------------------+
    |   ftol           |  1.e-7         | Relative error in the desired sum of squares               |
    +------------------+----------------+------------------------------------------------------------+
    |   maxfev         | 2000*(nvar+1)  | maximum number of function calls (nvar= # of variables)    |
    +------------------+----------------+------------------------------------------------------------+
    |   Dfun           | ``None``       | function to call for Jacobian calculation                  |
    +------------------+----------------+------------------------------------------------------------+


.. method:: scalar_minimize(method='Nelder-Mead', params=None, hess=None, tol=None, **kws)

   perform fit with any of the scalar minimization algorithms supported by
   :func:`scipy.optimize.minimize`.

    +-------------------------+-----------------+-----------------------------------------------------+
    | :meth:`scalar_minimize` | Default Value   | Description                                         |
    | arg                     |                 |                                                     |
    +=========================+=================+=====================================================+
    |   method                | ``Nelder-Mead`` | fitting method                                      |
    +-------------------------+-----------------+-----------------------------------------------------+
    |   tol                   | 1.e-7           | fitting and parameter tolerance                     |
    +-------------------------+-----------------+-----------------------------------------------------+
    |   hess                  | None            | Hessian of objective function                       |
    +-------------------------+-----------------+-----------------------------------------------------+


.. method:: prepare_fit(**kws)

   prepares and initializes model and Parameters for subsequent
   fitting. This routine prepares the conversion of :class:`Parameters`
   into fit variables, organizes parameter bounds, and parses, "compiles"
   and checks constrain expressions.   The method also creates and returns
   a new instance of a :class:`MinimizerResult` object that contains the
   copy of the Parameters that will actually be varied in the fit.

   This method is called directly by the fitting methods, and it is
   generally not necessary to call this function explicitly.


Getting and Printing Fit Reports
===========================================

.. function:: fit_report(result, modelpars=None, show_correl=True, min_correl=0.1)

   generate and return text of report of best-fit values, uncertainties,
   and correlations from fit.

   :param result:       :class:`MinimizerResult` object as returned by :func:`minimize`.
   :param modelpars:    Parameters with "Known Values" (optional, default None)
   :param show_correl:  whether to show list of sorted correlations [``True``]
   :param min_correl:   smallest correlation absolute value to show [0.1]

   If the first argument is a :class:`Parameters` object,
   goodness-of-fit statistics will not be included.

.. function:: report_fit(result, modelpars=None, show_correl=True, min_correl=0.1)

   print text of report from :func:`fit_report`.


An example fit with report would be

.. literalinclude:: ../examples/doc_withreport.py

which would write out::

    [[Fit Statistics]]
        # function evals   = 85
        # data points      = 1001
        # variables        = 4
        chi-square         = 498.812
        reduced chi-square = 0.500
    [[Variables]]
        amp:      13.9121944 +/- 0.141202 (1.01%) (init= 13)
        period:   5.48507044 +/- 0.026664 (0.49%) (init= 2)
        shift:    0.16203677 +/- 0.014056 (8.67%) (init= 0)
        decay:    0.03264538 +/- 0.000380 (1.16%) (init= 0.02)
    [[Correlations]] (unreported correlations are <  0.100)
        C(period, shift)             =  0.797
        C(amp, decay)                =  0.582
        C(amp, shift)                = -0.297
        C(amp, period)               = -0.243
        C(shift, decay)              = -0.182
        C(period, decay)             = -0.150
