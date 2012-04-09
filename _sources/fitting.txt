
=======================================
Performing Fits, Analyzing Outputs
=======================================

As shown in the previous sections, a simple fit can be performed with
the :func:`minimize` function.    For more sophisticated modeling,
the :class:`Minimizer` class can be used to gain a bit more control,
especially when using complicated constraints.


The :func:`minimize` function
===============================

The minimize function takes a function to minimize, a dictionary of
:class:`Parameter` , and several optional arguments.    See
:ref:`fit-func-label` for details on writing the function to minimize.

.. function:: minimize(function, params[, args=None[, kws=None[, engine='leastsq'[, **leastsq_kws]]]])

   find values for the params so that the sum-of-squares of the returned array
   from function is minimized.

   :param function:  function to return fit residual.  See :ref:`fit-func-label` for details.
   :type  function:  callable.
   :param params:  a dictionary of Parameters.  Keywords must be strings
                   that match ``[a-z_][a-z0-9_]*`` and is not a python
                   reserved word.  Each value must be :class:`Parameter`.
   :type  params:  dict
   :param args:  arguments tuple to pass to the residual function as  positional arguments.
   :type  args:  tuple
   :param kws:   dictionary to pass to the residual function as keyword arguments.
   :type  kws:  dict
   :param engine:  name of fitting engine to use. See  :ref:`fit-engines-label` for details
   :type  engine:  string
   :param leastsq_kws:  dictionary to pass to scipy.optimize.leastsq
   :type  leastsq_kws:  dict
   :return: Minimizer object, which can be used to inspect goodness-of-fit
            statistics, or to re-run fit.

   On output, the params will be updated with best-fit values and, where
   appropriate, estimated uncertainties and correlations.  See
   :ref:`fit-results-label` for further details.

..  _fit-func-label:

Writing a Fitting Function
===============================

An important component of a fit is writing a function to be minimized in
the least-squares sense.   Since this function will be called by other
routines, there are fairly stringent requirements for its call signature
and return value.   In principle, your function can be any python callable,
but it must look like this:

.. function:: func(params, *args, **kws):

   calculate residual from parameters.

   :param params: parameters.
   :type  params:  dict
   :param args:  positional arguments.  Must match `args` argument to :func:`minimize`
   :param kws:   keyword arguments.  Must match `kws` argument to :func:`minimize`
   :return: residual array (generally data-model) to be minimized in the least-squares sense.
   :rtype: numpy array.  The length of this array cannot change between calls.


A common use for the positional and keyword arguments would be to pass in other
data needed to calculate the residual, including such things as the data array,
dependent variable, uncertainties in the data, and other data structures for the
model calculation.

As the function will be passed in a dictionary of :class:`Parameter` s, it is
advisable to unpack these to get numerical values at the top of the function.  A
simple example would look like::

    def residual(pars, x, data=None):
        # unpack parameters:
        #  extract .value attribute for each parameter
        amp = pars['amp'].value
        period = pars['period'].value
        shift = pars['shift'].value
        decay = pars['decay'].value

        if abs(shift) > pi/2:
            shift = shift - sign(shift)*pi

        if abs(period) < 1.e-10:
            period = sign(period)*1.e-10

        model = amp * sin(shift + x/per) * exp(-x*x*decay*decay)

        if data is None:
            return model
        return (model - data)

In this example, ``x`` is a positional (required) argument, while the ``data``
array is actually optional (so that the function returns the model calculation
if the data is neglected).   Also note that the model calculation will divide
``x`` by the varied value of the 'period' Parameter.  It might be wise to
make sure this parameter cannot be 0.   It would be possible to use the bounds
on the :class:`Parameter` to do this::

    params['period'] = Parameter(value=2, min=1.e-10)

but might be wiser to put this directly in the function with::

        if abs(period) < 1.e-10:
            period = sign(period)*1.e-10


..  _fit-engines-label:

Choosing Different Fitting Engines
===========================================

By default, the `Levenberg-Marquardt
<http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_ algorithm is
used for fitting.  While often criticized, including the fact it finds a
*local* minima, this approach has some distinct advantages.  These include
being fast, and well-behaved for most curve-fitting needs, and making it
easy to estimate uncertainties for and correlations between pairs of fit
variables, as discussed in :ref:`fit-results-label`.

Alternative algorithms can also be used. These include `simulated annealing
<http://en.wikipedia.org/wiki/Simulated_annealing>`_ which promises a
better ability to avoid local minima, and `BFGS
<http://en.wikipedia.org/wiki/Limited-memory_BFGS>`_, which is a
modification of the quasi-Newton method.

To Select which of these algorithms to use, use the ``engine`` keyword to the
:func:`minimize` function or use the corresponding method name from the
:class:`Minimizer` class as listed in the
:ref:`Table of Supported Fitting Engines <fit-engine-table>`.

.. _fit-engine-table:

 Table of Supported Fitting Engines:

  +------------------------+-------------------------------------+------------------------------+
  | Engine                 |  ``engine`` arg to :func:`minimize` | :class:`Minimizer` method    |
  +========================+=====================================+==============================+
  | Levenberg-Marquardt    |  ``leastsq``                        |  :meth:`leastsq`             |
  +------------------------+-------------------------------------+------------------------------+
  | L-BFGS-B               |  ``lbfgsb``                         |  :meth:`lbfgsb`              |
  +------------------------+-------------------------------------+------------------------------+
  | Simulated Annealing    |  ``anneal``                         |  :meth:`anneal`              |
  +------------------------+-------------------------------------+------------------------------+


.. warning::

  The Levenberg-Marquardt method is *by far* the most tested fit method,
  and much of this documentation assumes that this is the method used.  For
  example, many of the fit statistics and estimates for uncertainties in
  parameters discussed in :ref:`fit-results-label` are done only for the
  ``leastsq`` method.

In particular, the simulated annealing method appears to not work
correctly.... understanding this is on the ToDo list.

..  _fit-results-label:

Goodness-of-Fit and estimated uncertainty and correlations
===================================================================

On a successful fit using the `leastsq` engine, several goodness-of-fit
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
| Minimizer Attribute  | Description / Formula                                                      |
+======================+============================================================================+
|  ``nfev``            | number of function evaluations                                             |
+----------------------+----------------------------------------------------------------------------+
|  ``success``         | boolean (``True``/``False``) for whether fit succeeded.                    |
+----------------------+----------------------------------------------------------------------------+
|  ``errorbars``       | boolean (``True``/``False``) for whether uncertainties were estimated.     |
+----------------------+----------------------------------------------------------------------------+
|  ``message``         | message about fit success.                                                 |
+----------------------+----------------------------------------------------------------------------+
|  ``ier``             | integer error value from scipy.optimize.leastsq                            |
+----------------------+----------------------------------------------------------------------------+
|  ``lmdif_message``   | message from scipy.optimize.leastsq                                        |
+----------------------+----------------------------------------------------------------------------+
|  ``nvarys``          | number of variables in fit  :math:`N_{\rm varys}`                          |
+----------------------+----------------------------------------------------------------------------+
|  ``ndata``           | number of data points:  :math:`N`                                          |
+----------------------+----------------------------------------------------------------------------+
|  ``nfree``           | degrees of freedom in fit:  :math:`N - N_{\rm varys}`                      |
+----------------------+----------------------------------------------------------------------------+
|  ``residual``        | residual array (return of :func:`func`:  :math:`{\rm Resid}`               |
+----------------------+----------------------------------------------------------------------------+
|  ``chisqr``          | chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`                    |
+----------------------+----------------------------------------------------------------------------+
|  ``redchi``          | reduced chi-square: :math:`\chi^2_{\nu}= {\chi^2} / {(N - N_{\rm varys})}` |
+----------------------+----------------------------------------------------------------------------+

Note that the calculation of chi-square and reduced chi-square assume that the
returned residual function is scaled properly to the uncertainties in the data.
For these statistics to be meaningful, the person writing the function to
function to be minimized must scale them properly.

After a fit using using the `leastsq` engine has completed succsessfully,
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


..  _fit-minimizer-label:

Using the :class:`Minimizer` class
=======================================

For full control of the fitting process, you'll want to create a
:class:`Minimizer` object, or at least use the one returned from the
:func:`minimize` function.

.. class:: Minimizer(function, params[, fcn_args=None[, fcn_kws=None[, **kws]]]])

   creates a Minimizer, for fine-grain access to fitting methods and attributes.

   :param function:  function to return fit residual.  See :ref:`fit-func-label` for details.
   :type  function:  callable.
   :param params:  a dictionary of Parameters.  Keywords must be strings
                   that match ``[a-z_][a-z0-9_]*`` and is not a python
                   reserved word.  Each value must be :class:`Parameter`.
   :type  params:  dict
   :param fcn_args:  arguments tuple to pass to the residual function as  positional arguments.
   :type  fcn_args:  tuple
   :param fcn_kws:   dictionary to pass to the residual function as keyword arguments.
   :type  fcn_kws:  dict
   :param leastsq_kws:  dictionary to pass to scipy.optimize.leastsq
   :type  leastsq_kws:  dict
   :return: Minimizer object, which can be used to inspect goodness-of-fit
            statistics, or to re-run fit.


The Minimizer object has a few public methods:

.. method:: leastsq(**kws)

   perform fit with Levenberg-Marquardt algorithm.  Keywords will be passed directly to
   `scipy.optimize.leastsq <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html>`_.
   By default, numerical derivatives are used, and the following arguments are set:

    +----------------------+----------------+------------------------------------------------------------+
    | ``leastsq`` argument |  Default Value | Description                                                |
    +======================+================+============================================================+
    | ``xtol``             |  1.e-7         | Relative error in the approximate solution                 |
    +----------------------+----------------+------------------------------------------------------------+
    | ``ftol``             |  1.e-7         | Relative error in the desired sum of squares               |
    +----------------------+----------------+------------------------------------------------------------+
    | ``maxfev``           | 1000*(nvar+1)  | maximum number of function calls (nvar= # of variables)    |
    +----------------------+----------------+------------------------------------------------------------+


.. method:: anneal(**kws)

   perform fit with Simulated Annealing.  Keywords will be passed directly to
   `scipy.optimize.anneal <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.anneal.html>`_.

.. method:: lbfgsb(**kws)

   perform fit with L-BFGS-B algorithm.  Keywords will be passed directly to
   `scipy.optimize.fmin_l_bfgs_b <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.


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


