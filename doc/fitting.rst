.. _minimize_chapter:

=======================================
Performing Fits, Analyzing Outputs
=======================================

As shown in the previous chapter, a simple fit can be performed with the
:func:`minimize` function.  For more sophisticated modeling, the
:class:`Minimizer` class can be used to gain a bit more control, especially
when using complicated constraints.


The :func:`minimize` function
===============================

The minimize function takes a objective function (the function that
calculates the array to be minimized), a :class:`Parameters` ordered
dictionary, and several optional arguments.  See :ref:`fit-func-label` for
details on writing the function to minimize.

.. function:: minimize(function, params[, args=None[, kws=None[, method='leastsq'[, scale_covar=True[, iter_cb=None[, **leastsq_kws]]]]]])

   find values for the ``params`` so that the sum-of-squares of the array returned
   from ``function`` is minimized.

   :param function:  function to return fit residual.  See :ref:`fit-func-label` for details.
   :type  function:  callable.
   :param params:  a :class:`Parameters` dictionary.  Keywords must be strings
                   that match ``[a-z_][a-z0-9_]*`` and is not a python
                   reserved word.  Each value must be :class:`Parameter`.
   :type  params:  dict or :class:`Parameters`.
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
   :param leastsq_kws:  dictionary to pass to :func:`scipy.optimize.leastsq`.
   :type  leastsq_kws:  dict

   :return: Minimizer object, which can be used to inspect goodness-of-fit
            statistics, or to re-run fit.

   On output, the params will be updated with best-fit values and, where
   appropriate, estimated uncertainties and correlations.  See
   :ref:`fit-results-label` for further details.

   If provided, the ``iter_cb`` function should take arguments of ``params,
   iter, resid, *args, **kws``, where ``params`` will have the current
   parameter values, ``iter`` the iteration, ``resid`` the current residual
   array, and ``*args`` and ``**kws`` as passed to the objective function.

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
   :type  params:  dict
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

Alternative algorithms can also be used by providing the ``method`` keyword
to the :func:`minimize` function or use the corresponding method name from
the :class:`Minimizer` class as listed in the :ref:`Table of Supported
Fitting Methods <fit-methods-table>`.

.. _fit-methods-table:

 Table of Supported Fitting Methods:

 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Fitting Meth          | ``method`` arg to            | :class:`Minimizer`  | ``method`` arg to           |
 |                       | :func:`minimize`             | method              | :meth:`scalar_minimize`     |
 +=======================+==============================+=====================+=============================+
 | Levenberg-Marquardt   |  ``leastsq``                 | :meth:`leastsq`     |   Not available             |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Nelder-Mead           |  ``nelder``                  | :meth:`fmin`        | ``Nelder-Mead``             |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | L-BFGS-B              |  ``lbfgsb``                  | :meth:`lbfgsb`      | ``L-BFGS-B``                |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Powell                |  ``powell``                  |                     | ``Powell``                  |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Conjugate Gradient    |  ``cg``                      |                     | ``CG``                      |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Newton-CG             |  ``newton``                  |                     | ``Newton-CG``               |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | COBYLA                |  ``cobyla``                  |                     |  ``COBYLA``                 |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | COBYLA                |  ``cobyla``                  |                     |  ``COBYLA``                 |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Truncated Newton      |  ``tnc``                     |                     |  ``TNC``                    |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Trust Newton-CGn      |  ``trust-ncg``               |                     |  ``trust-ncg``              |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Dogleg                |  ``dogleg``                  |                     |  ``dogleg``                 |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Sequential Linear     |  ``slsqp``                   |                     |  ``SLSQP``                  |
 | Squares Programming   |                              |                     |                             |
 +-----------------------+------------------------------+---------------------+-----------------------------+
 | Differential          |  ``differential_evolution``  |                     | ``differential_evolution``  |
 | Evolution             |                              |                     |                             |
 +-----------------------+------------------------------+---------------------+-----------------------------+

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

Goodness-of-Fit and estimated uncertainty and correlations
===================================================================

On a successful fit using the `leastsq` method, several goodness-of-fit
statistics and values related to the uncertainty in the fitted variables will be
calculated.  These are all encapsulated in the :class:`Minimizer` object for the
fit, as returned by :func:`minimize`.  The values related to the entire fit are
stored in attributes of the :class:`Minimizer` object, as shown in :ref:`Table
of Fit Results <goodfit-table>` while those related to each fitted variables are
stored as attributes of the corresponding :class:`Parameter`.


.. _goodfit-table:

 Table of Fit Results:  These values, including the standard Goodness-of-Fit statistics,
 are all attributes of the :class:`Minimizer` object returned by :func:`minimize`.

+----------------------+----------------------------------------------------------------------------+
| :class:`Minimizer`   | Description / Formula                                                      |
| Attribute            |                                                                            |
+======================+============================================================================+
|    nfev              | number of function evaluations                                             |
+----------------------+----------------------------------------------------------------------------+
|    success           | boolean (``True``/``False``) for whether fit succeeded.                    |
+----------------------+----------------------------------------------------------------------------+
|    errorbars         | boolean (``True``/``False``) for whether uncertainties were estimated.     |
+----------------------+----------------------------------------------------------------------------+
|    message           | message about fit success.                                                 |
+----------------------+----------------------------------------------------------------------------+
|    ier               | integer error value from :func:`scipy.optimize.leastsq`                    |
+----------------------+----------------------------------------------------------------------------+
|    lmdif_message     | message from :func:`scipy.optimize.leastsq`                                |
+----------------------+----------------------------------------------------------------------------+
|    nvarys            | number of variables in fit  :math:`N_{\rm varys}`                          |
+----------------------+----------------------------------------------------------------------------+
|    ndata             | number of data points:  :math:`N`                                          |
+----------------------+----------------------------------------------------------------------------+
|    nfree `           | degrees of freedom in fit:  :math:`N - N_{\rm varys}`                      |
+----------------------+----------------------------------------------------------------------------+
|    residual          | residual array (return of :func:`func`:  :math:`{\rm Resid}`               |
+----------------------+----------------------------------------------------------------------------+
|    chisqr            | chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`                    |
+----------------------+----------------------------------------------------------------------------+
|    redchi            | reduced chi-square: :math:`\chi^2_{\nu}= {\chi^2} / {(N - N_{\rm varys})}` |
+----------------------+----------------------------------------------------------------------------+
|    var_map           | list of variable parameter names for rows/columns of covar                 |
+----------------------+----------------------------------------------------------------------------+
|    covar             | covariance matrix (with rows/columns using var_map                         |
+----------------------+----------------------------------------------------------------------------+

Note that the calculation of chi-square and reduced chi-square assume that the
returned residual function is scaled properly to the uncertainties in the data.
For these statistics to be meaningful, the person writing the function to
be minimized must scale them properly.

After a fit using using the :meth:`leastsq` method has completed successfully,
standard errors for the fitted variables and correlations between pairs of
fitted variables are automatically calculated from the covariance matrix.
The standard error (estimated :math:`1\sigma` error-bar) go into the
:attr:`stderr` attribute of the Parameter.  The correlations with all other
variables will be put into the :attr:`correl` attribute of the Parameter --
a dictionary with keys for all other Parameters and values of the
corresponding correlation.

In some cases, it may not be possible to estimate the errors and
correlations.  For example, if a variable actually has no practical effect
on the fit, it will likely cause the covariance matrix to be singular,
making standard errors impossible to estimate.  Placing bounds on varied
Parameters makes it more likely that errors cannot be estimated, as being
near the maximum or minimum value makes the covariance matrix singular.  In
these cases, the :attr:`errorbars` attribute of the fit result
(:class:`Minimizer` object) will be ``False``.

.. module:: Minimizer

..  _fit-minimizer-label:

Using the :class:`Minimizer` class
=======================================

For full control of the fitting process, you'll want to create a
:class:`Minimizer` object, or at least use the one returned from the
:func:`minimize` function.

.. class:: Minimizer(function, params, fcn_args=None, fcn_kws=None, iter_cb=None, scale_covar=True, **kws)

   creates a Minimizer, for fine-grain access to fitting methods and attributes.

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
   :return: Minimizer object, which can be used to inspect goodness-of-fit
            statistics, or to re-run fit.


The Minimizer object has a few public methods:

.. method:: leastsq(scale_covar=True, **kws)

   perform fit with Levenberg-Marquardt algorithm.  Keywords will be passed directly to
   :func:`scipy.optimize.leastsq`.
   By default, numerical derivatives are used, and the following arguments are set:

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



.. method:: mcmc(samples, burn=0, thin=1)

   :param samples:     The number of samples the MCMC model draws
   :type  sample:      int
   :param burn:        The number of samples the MCMC model discards from the start of the sampling
                       process.
   :type  burn:        int
   :param thin:        Once the burn process has completed a tally occurs every *thin* samples.
                       Increase this number to reduce the correlation between successive draws (e.g. 30)
   :type  thin:        int

   :return: pymc.MCMC instance, which can be processed to obtain statistics about the sampling process, or
                      be used to conduct further sampling.

   Fits data with a Bayesian modelling approach via Markov Chain Monte Carlo (MCMC).  One obtains
   the posterior probability (uncertainty) distribution for each varied parameter. The ``pymc``
   package must be installed to use this method.  Once the sampling process is completed the method
   calculates statistics for each parameter, as if a 'leastsq' fit had been completed.  i.e. the mean
   and standard deviation of the posterior distribution provides the fitted parameter value and the
   ``stderr`` respectively.  In addition, the interparameter correlations are calculated.  Very detailed
   information can be obtained from the ``Minimizer.MDL`` attribute, but is beyond the scope of this
   documentation. The user is advised to consult the `PyMC <http://pymc-devs.github.io/pymc/>`_ site
   for further details.


.. method:: lbfgsb(**kws)


   perform fit with L-BFGS-B algorithm.  Keywords will be passed directly to
   :func:`scipy.optimize.fmin_l_bfgs_b`.


    +------------------+----------------+------------------------------------------------------------+
    | :meth:`lbfgsb`   |  Default Value | Description                                                |
    | arg              |                |                                                            |
    +==================+================+============================================================+
    |   factr          | 1000.0         |                                                            |
    +------------------+----------------+------------------------------------------------------------+
    |   approx_grad    |  ``True``      | calculate approximations of gradient                       |
    +------------------+----------------+------------------------------------------------------------+
    |   maxfun         | 2000*(nvar+1)  | maximum number of function calls (nvar= # of variables)    |
    +------------------+----------------+------------------------------------------------------------+

.. warning::

  :meth:`lbfgsb` is deprecated.  Use :meth:`minimize` with ``method='lbfgsb'``.

.. method:: fmin(**kws)

   perform fit with Nelder-Mead downhill simplex algorithm.  Keywords will be passed directly to
   :func:`scipy.optimize.fmin`.

    +------------------+----------------+------------------------------------------------------------+
    | :meth:`fmin`     |  Default Value | Description                                                |
    | arg              |                |                                                            |
    +==================+================+============================================================+
    |   ftol           | 1.e-4          | function tolerance                                         |
    +------------------+----------------+------------------------------------------------------------+
    |   xtol           | 1.e-4          | parameter tolerance                                        |
    +------------------+----------------+------------------------------------------------------------+
    |   maxfun         | 5000*(nvar+1)  | maximum number of function calls (nvar= # of variables)    |
    +------------------+----------------+------------------------------------------------------------+

.. warning::

  :meth:`fmin` is deprecated.  Use :meth:`minimize` with ``method='nelder'``.


.. method:: scalar_minimize(method='Nelder-Mead', hess=None, tol=None, **kws)

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
   into fit variables, organizes parameter bounds, and parses, checks and
   "compiles" constrain expressions.


   This is called directly by the fitting methods, and it is generally not
   necessary to call this function explicitly.  An exception is when you
   would like to call your function to minimize prior to running one of the
   minimization routines, for example, to calculate the initial residual
   function.  In that case, you might want to do something like::

      myfit = Minimizer(my_residual, params,  fcn_args=(x,), fcn_kws={'data':data})

      myfit.prepare_fit()
      init = my_residual(p_fit, x)
      pylab.plot(x, init, 'b--')

      myfit.leastsq()

   That is, this method should be called prior to your fitting function being called.


Getting and Printing Fit Reports
===========================================

.. function:: fit_report(params, modelpars=None, show_correl=True, min_correl=0.1)

   generate and return text of report of best-fit values, uncertainties,
   and correlations from fit.

   :param params:       Parameters from fit, or Minimizer object as returned by :func:`minimize`.
   :param modelpars:    Parameters with "Known Values" (optional, default None)
   :param show_correl:  whether to show list of sorted correlations [``True``]
   :param min_correl:   smallest correlation absolute value to show [0.1]

   If the first argument is a Minimizer object, as returned from
   :func:`minimize`, the report will include some goodness-of-fit statistics.

.. function:: report_fit(params, modelpars=None, show_correl=True, min_correl=0.1)

   print text of report from :func:`fit_report`.


An example fit with report would be

.. literalinclude:: ../examples/doc_withreport.py

which would write out::


    [[Variables]]
        amp:      13.9121944 +/- 0.141202 (1.01%) (init= 13)
        decay:    0.03264538 +/- 0.000380 (1.16%) (init= 0.02)
        period:   5.48507044 +/- 0.026664 (0.49%) (init= 2)
        shift:    0.16203677 +/- 0.014056 (8.67%) (init= 0)
    [[Correlations]] (unreported correlations are <  0.100)
        C(period, shift)             =  0.797
        C(amp, decay)                =  0.582
