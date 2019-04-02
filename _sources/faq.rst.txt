%\.. _faq_chapter:

====================================
Frequently Asked Questions
====================================

A list of common questions.

What's the best way to ask for help or submit a bug report?
================================================================

See :ref:`support_chapter`.


Why did my script break when upgrading from lmfit 0.8.3 to 0.9.0?
====================================================================

See :ref:`whatsnew_090_label`


I get import errors from IPython
==============================================================

If you see something like::

    from IPython.html.widgets import Dropdown

    ImportError: No module named 'widgets'

then you need to install the ``ipywidgets`` package, try:  ``pip install ipywidgets``.




How can I fit multi-dimensional data?
========================================

The fitting routines accept data arrays that are one dimensional and double
precision.  So you need to convert the data and model (or the value
returned by the objective function) to be one dimensional.  A simple way to
do this is to use :numpydoc:`ndarray.flatten`, for example::

    def residual(params, x, data=None):
        ....
        resid = calculate_multidim_residual()
        return resid.flatten()

How can I fit multiple data sets?
========================================

As above, the fitting routines accept data arrays that are one dimensional
and double precision.  So you need to convert the sets of data and models
(or the value returned by the objective function) to be one dimensional.  A
simple way to do this is to use :numpydoc:`concatenate`.  As an
example, here is a residual function to simultaneously fit two lines to two
different arrays.  As a bonus, the two lines share the 'offset' parameter::

    import numpy as np
    def fit_function(params, x=None, dat1=None, dat2=None):
        model1 = params['offset'] + x * params['slope1']
        model2 = params['offset'] + x * params['slope2']

        resid1 = dat1 - model1
        resid2 = dat2 - model2
        return np.concatenate((resid1, resid2))



How can I fit complex data?
===================================

As with working with multi-dimensional data, you need to convert your data
and model (or the value returned by the objective function) to be double
precision floating point numbers. The simplest approach is to use
:numpydoc:`ndarray.view`, perhaps like::

   import numpy as np
   def residual(params, x, data=None):
       ....
       resid = calculate_complex_residual()
       return resid.view(np.float)

Alternately, you can use the lmfit.Model class to wrap a fit function
that returns a complex vector. It will automatically apply the above
prescription when calculating the residual. The benefit to this method
is that you also get access to the plot routines from the ModelResult
class, which are also complex-aware.


Can I constrain values to have integer values?
===============================================

Basically, no.  None of the minimizers in lmfit support integer
programming.  They all (I think) assume that they can make a very small
change to a floating point value for a parameters value and see a change in
the value to be minimized.


How should I cite LMFIT?
==================================

See https://dx.doi.org/10.5281/zenodo.11813

I get errors from NaN in my fit.  What can I do?
======================================================

The solvers used by lmfit use NaN (see
https://en.wikipedia.org/wiki/NaN) values as signals that the calculation
cannot continue.  If any value in the residual array (typically
`(data-model)*weight`) is NaN, then calculations of chi-square or
comparisons with other residual arrays to try find a better fit will also
give NaN and fail. There is no sensible way for lmfit or any of the
optimization routines to know how to handle such NaN values.  They
indicate that numerical calculations are not sensible and must stop.

This means that if your objective function (if using ``minimize``) or model
function (if using ``Model``) generates a NaN, the fit will stop
immediately. If your objective or model function generates a NaN, you
really must handle that.

`nan_policy`
~~~~~~~~~~~~~~~~~~

If you are using :class:`lmfit.Model` and the NaN values come from your
data array and are meant to indicate missing values, or if you using
:func:`lmfit.minimize` with the same basic intention, then it might be
possible to get a successful fit in spite of the NaN values. To do this,
you can add a ``nan_policy='omit'``` argument to :func:`lmfit.minimize`, or
when creating a :class:`lmfit.Model`, or when running
:meth:`lmfit.Model.fit`.

In order for this to be effective, the number of NaN values cannot ever
change during the fit.  If the NaN values come from the data and not the
calculated model, that should be the case.


Common sources of NaN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are seeing erros due to NaN values, you will need to figure out
where they are coming from and eliminate them.  It is sometimes difficult
to tell what causes NaN values.  Keep in mind that all values should be
assumed to be either scalar values or numpy arrays of double precision real
numbers when fitting.  Some of the most likely causes of NaNs are:

   * taking ``sqrt(x)`` or ``log(x)`` where ``x`` is negative.

   * doing ``x**y`` where ```x`` is negative.  Since ``y`` is real, there will
     be a fractional component, and a negative number to a fractional
     exponent is not a real number.

   * doing ``x/y`` where both ``x`` and ``y`` are 0.

If you use these very common constructs in your objective or model
function, you should take some caution for what values you are passing
these functions and operators.  Many special functions have similar
limitations and should also be viewed with some suspicion if NaNs are being
generated.

A related problem is the generation of Inf (Infinity in floating point),
which generally comes from ``exp(x)`` where ``x`` has values greater than 700
or so, so that the resulting value is greater than 1.e308.  Inf is only
slightly better than NaN. It will completely ruin the ability to do the
fit.  However, unlike NaN, it is also usually clear how to handle Inf, as
you probably won't ever have values greater than 1.e308 and can therefore
(usually) safely clip the argument passed to ``exp()`` to be smaller than
about 700.
