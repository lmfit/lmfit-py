
=======================================
Performing Fits, Analyzing Outputs
=======================================

As shown in the previous sections, a simple fit can be performed with 
the :func:`minimize` function.    For more sophisticated modeling, 
the :class:`Minimizer` class can be used to gain a bit more control,
especially when using complicated constraints.


The :func:`minimize` function 
===============================

The minimize function takes a function to minimze, a dictionary of
:class:`Parameter` , and several optional arguments.    See
:ref:`fit-func-label` for details on writing the function to minimize.

.. function:: minimize(function, params[, args=None[, kws=None[, **leastsq_kws]]])

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
        #  extract .value attribute for each parametr
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




..  _fit-results-label:

Goodness-of-Fit and estimated uncertainty and correlations
===================================================================

On a successful fit, several goodness-of-fit statistics and values related to the uncertainty in
the fitted variables will be calculated.  These are all encapsulated in the :class:`Minimizer`
object for the fit, as returned by :func:`minimize`.  The values related to the entire fit are
stored in attributes of the :class:`Minimizer` object, as shown in :ref:`Table of Goodness-of-Fit
Statistics <goodfit-table>` while those related to each fitted variables are stored as attributes
of the corresponding :class:`Parameter`.


.. _goodfit-table:

 Table of Goodness-of-Fit Statistics:  These statistics are all attributes of the :class:`Minimizer` object returned by :func:`minimize`.

+----------------------+--------------------------------------------------------------------------+
| Minimizer Attribute  |  Description / Formula                                                   +
+======================+==========================================================================+
| ``nfev``             |  number of function evaluations                                          |
+----------------------+--------------------------------------------------------------------------+
| ``success``          | boolean (``True``/``False``) for whether fit succeeded.                  |
+----------------------+--------------------------------------------------------------------------+
| ``errorbars``        | boolean (``True``/``False``) for whether uncertainities were estimated.  |
+----------------------+--------------------------------------------------------------------------+
| ``message``          | message about fit success.                                               |
+----------------------+--------------------------------------------------------------------------+
|  ``ier``             | integer error value from scipy.optimize.leastsq                          |
+----------------------+--------------------------------------------------------------------------+
|  ``lmdif_message``   | message from scipy.optimize.leastsq                                      |
+----------------------+--------------------------------------------------------------------------+
|   ``nvarys``         |  number of variables in fit  :math:`N_{\rm varys}`                       |
+----------------------+--------------------------------------------------------------------------+
|   ``ndata``          |  number of data points:  :math:`N`                                       |
+----------------------+--------------------------------------------------------------------------+
|   ``nfree``          |  degrees of freedom in fit:  :math:`N - N_{\rm varys}`                   |
+----------------------+--------------------------------------------------------------------------+
|   ``residual``       |  residual array (return of :func:`func`:  :math:`{\rm Resid}`            |
+----------------------+--------------------------------------------------------------------------+
|   ``chisqr``         |  chi-square: :math:`\chi^2 = \sum_i^N [{\rm Resid}_i]^2`                 |
+----------------------+--------------------------------------------------------------------------+
|   ``redchi``         | reduced chi-square: :math:`\chi^2_{\nu}= {\chi^2} / {N - N_{\rm varys}}` |                    
+----------------------+--------------------------------------------------------------------------+

Note that the calculation of chi-square and reduced chi-square assume that the
returned residual function is scaled properly to the uncertainties in the data.
For these statistics to be meaningful, the person writing the function to
function to be minimized must scale them properly.



..  _fit-minimizer-label:

Using the :class:`Minimizer` class
=======================================

For full control of the fitting process, you'll want to create a
:class:`Minimizer` object, or at least use the one returned from the
:func:`minimize` function. 

.. class:: Minimizer(fcn, params[, fcn_args=None[, fcn_kwsn=None[, engine='leastsq'[, **kws]]]])

   create a Minimizer object.



As 


