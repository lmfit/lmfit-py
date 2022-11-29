.. _faq_chapter:

==========================
Frequently Asked Questions
==========================

A list of common questions.

What's the best way to ask for help or submit a bug report?
===========================================================

See :ref:`support_chapter`.


Why did my script break when upgrading from lmfit 0.8.3 to 0.9.0?
=================================================================

See :ref:`whatsnew_090_label`.


I get import errors from IPython
================================

If you see something like::

    from IPython.html.widgets import Dropdown

    ImportError: No module named 'widgets'

then you need to install the ``ipywidgets`` package, try: ``pip install ipywidgets``.


How can I fit multi-dimensional data?
=====================================

The fitting routines accept data arrays that are one-dimensional and double
precision. So you need to convert the data and model (or the value
returned by the objective function) to be one-dimensional. A simple way to
do this is to use :numpydoc:`ndarray.flatten`, for example::

    def residual(params, x, data=None):
        ....
        resid = calculate_multidim_residual()
        return resid.flatten()


How can I fit multiple data sets?
=================================

As above, the fitting routines accept data arrays that are one-dimensional
and double precision. So you need to convert the sets of data and models
(or the value returned by the objective function) to be one-dimensional. A
simple way to do this is to use :numpydoc:`concatenate`. As an
example, here is a residual function to simultaneously fit two lines to two
different arrays. As a bonus, the two lines share the 'offset' parameter::

    import numpy as np


    def fit_function(params, x=None, dat1=None, dat2=None):
        model1 = params['offset'] + x * params['slope1']
        model2 = params['offset'] + x * params['slope2']

        resid1 = dat1 - model1
        resid2 = dat2 - model2
        return np.concatenate((resid1, resid2))


How can I fit complex data?
===========================

As with working with multi-dimensional data, you need to convert your data
and model (or the value returned by the objective function) to be double
precision, floating point numbers. The simplest approach is to use
:numpydoc:`ndarray.view`, perhaps like::

   import numpy as np


   def residual(params, x, data=None):
       ....
       resid = calculate_complex_residual()
       return resid.view(float)

Alternately, you can use the :class:`lmfit.Model` class to wrap a fit function
that returns a complex vector. It will automatically apply the above
prescription when calculating the residual. The benefit to this method
is that you also get access to the plot routines from the ModelResult
class, which are also complex-aware.


How should I cite LMFIT?
========================

See https://dx.doi.org/10.5281/zenodo.11813


I get errors from NaN in my fit. What can I do?
================================================

The solvers used by lmfit use NaN (see
https://en.wikipedia.org/wiki/NaN) values as signals that the calculation
cannot continue. If any value in the residual array (typically
``(data-model)*weight``) is NaN, then calculations of chi-square or
comparisons with other residual arrays to try find a better fit will also
give NaN and fail. There is no sensible way for lmfit or any of the
optimization routines to know how to handle such NaN values. They
indicate that numerical calculations are not sensible and must stop.

This means that if your objective function (if using ``minimize``) or model
function (if using ``Model``) generates a NaN, the fit will stop
immediately. If your objective or model function generates a NaN, you
really must handle that.


``nan_policy``
~~~~~~~~~~~~~~

If you are using :class:`lmfit.Model` and the NaN values come from your
data array and are meant to indicate missing values, or if you using
:func:`lmfit.minimize` with the same basic intention, then it might be
possible to get a successful fit in spite of the NaN values. To do this,
you can add a ``nan_policy='omit'`` argument to :func:`lmfit.minimize`, or
when creating a :class:`lmfit.Model`, or when running
:meth:`lmfit.Model.fit`.

In order for this to be effective, the number of NaN values cannot ever
change during the fit. If the NaN values come from the data and not the
calculated model, that should be the case.


Common sources of NaN
~~~~~~~~~~~~~~~~~~~~~

If you are seeing errors due to NaN values, you will need to figure out
where they are coming from and eliminate them. It is sometimes difficult
to tell what causes NaN values. Keep in mind that all values should be
assumed to be either scalar values or numpy arrays of double precision real
numbers when fitting. Some of the most likely causes of NaNs are:

   * taking ``sqrt(x)`` or ``log(x)`` where ``x`` is negative.

   * doing ``x**y`` where ``x`` is negative. Since ``y`` is real, there will
     be a fractional component, and a negative number to a fractional
     exponent is not a real number.

   * doing ``x/y`` where both ``x`` and ``y`` are 0.

If you use these very common constructs in your objective or model
function, you should take some caution for what values you are passing
these functions and operators. Many special functions have similar
limitations and should also be viewed with some suspicion if NaNs are being
generated.

A related problem is the generation of Inf (Infinity in floating point),
which generally comes from ``exp(x)`` where ``x`` has values greater than 700
or so, so that the resulting value is greater than 1.e308. Inf is only
slightly better than NaN. It will completely ruin the ability to do the
fit. However, unlike NaN, it is also usually clear how to handle Inf, as
you probably won't ever have values greater than 1.e308 and can therefore
(usually) safely clip the argument passed to ``exp()`` to be smaller than
about 700.


.. _faq_params_stuck:

Why are Parameter values sometimes stuck at initial values?
===========================================================

In order for a Parameter to be optimized in a fit, changing its value must
have an impact on the fit residual (``data-model`` when curve fitting, for
example).  If a fit has not changed one or more of the Parameters, it means
that changing those Parameters did not change the fit residual.

Normally (that is, unless you specifically provide a function for
calculating the derivatives, in which case you probably would not be asking
this question ;)), the fitting process begins by making a very small change
to each Parameter value to determine which way and how large of a change to
make for the parameter: This is the derivative or Jacobian (change in
residual per change in parameter value).  By default, the change made for
each variable Parameter is to multiply its value by (1.0+1.0e-8) or so
(unless the value is below about 1.e-15, in which case it adds 1.0e-8).  If
that small change does not change the residual, then the value of the
Parameter will not be updated.

Parameter values that are "way off" are a common reason for Parameters
being stuck at initial values.  As an example, imagine fitting peak-like
data with and ``x`` range of 0 to 10, peak centered at 6, and a width of 1 or
2 or so, as in the example at
:ref:`sphx_glr_examples_documentation_model_gaussian.py`.  A Gaussian
function with an initial value of for the peak center at 5 and an initial
width or 5 will almost certainly find a good fit.  An initial value of the
peak center of -50 will end up being stuck with a "bad fit" because a small
change in Parameters will still lead the modeled Gaussian to have no
intensity over the actual range of the data.  You should make sure that
initial values for Parameters are reasonable enough to actually effect the
fit.  As it turns out in the example linked to above, changing the center
value to any value between about 0 and 10 (that is, the data range) will
result to a good fit.

Another common cause for Parameters being stuck at initial values is when
the initial value is at a boundary value.  For this case, too, a small
change in the initial value for the Parameter will still leave the value at
the boundary value and not show any real change in the residual.

If you're using bounds, make sure the initial values for the Parameters are
not at the boundary values.

Finally, one reason for a Parameter to not change is that they are actually
used as discrete values.  This is discussed below in :ref:`faq_discrete_params`.


.. _faq_params_no_uncertainties:

Why are uncertainties in Parameters sometimes not determined?
=============================================================

In order for Parameter uncertainties to be estimated, each variable
Parameter must actually change the fit, and cannot be stuck at an initial
value or at a boundary value.  See :ref:`faq_params_stuck` for why values may
not change from their initial values.


.. _faq_discrete_params:

Can Parameters be used for Array Indices or Discrete Values?
=============================================================

The short answer is "No": variables in all of the fitting methods used in
``lmfit`` (and all of those available in ``scipy.optimize``) are treated as
continuous values, and represented as double precision floating point
values.  As an important example, you cannot have a variable that is
somehow constrained to be an integer.

Still, it is a rather common question of how to fit data to a model that
includes a breakpoint, perhaps

    .. math::

       f(x; x_0, a, b, c) =
       \begin{cases}
       c          & \quad \text{for} \> x < x_0 \\
       a + bx^2  & \quad \text{for} \> x > x_0
       \end{cases}


That you implement with a model function and use to fit data like this:

.. jupyter-execute::

    import numpy as np

    import lmfit


    def quad_off(x, x0, a, b, c):
        model = a + b * x**2
        model[np.where(x < x0)] = c
        return model


    x0 = 19
    b = 0.02
    a = 2.0
    xdat = np.linspace(0, 100, 101)
    ydat = a + b * xdat**2
    ydat[np.where(xdat < x0)] = a + b * x0**2
    ydat += np.random.normal(scale=0.1, size=xdat.size)

    mod = lmfit.Model(quad_off)
    pars = mod.make_params(x0=22, a=1, b=1, c=1)

    result = mod.fit(ydat, pars, x=xdat)
    print(result.fit_report())

This will not result in a very good fit, as the value for ``x0`` cannot be
found by making a small change in its value.  Specifically,
``model[np.where(x < x0)]`` will give the same result for ``x0=22`` and
``x0=22.001``, and so that value is not changed during the fit.

There are a couple ways around this problem. First, you may be able to
make the fit depend on ``x0`` in a way that is not just discrete.  That
depends on your model function. A second option is to treat the break not as a
hard break but as a more gentle transition with a sigmoidal function, such
as an error function.  Like the break-point, these will go from 0 to 1, but
more gently and with some finite value leaking into neighboring points.
The amount of leakage or width of the step can also be adjusted.

A simple modification of the above to use an error function would
look like this and give better fit results:

.. jupyter-execute::

    import numpy as np
    from scipy.special import erf

    import lmfit


    def quad_off(x, x0, a, b, c):
        m1 = a + b * x**2
        m2 = c * np.ones(len(x))
        # step up from 0 to 1 at x0: (erf(x-x0)+1)/2
        # step down from 1 to 0 at x0: (1-erf(x-x0))/2
        model = m1 * (erf(x-x0)+1)/2 + m2 * (1-erf(x-x0))/2
        return model


    x0 = 19
    b = 0.02
    a = 2.0
    xdat = np.linspace(0, 100, 101)
    ydat = a + b * xdat**2
    ydat[np.where(xdat < x0)] = a + b * x0**2
    ydat += np.random.normal(scale=0.1, size=xdat.size)

    mod = lmfit.Model(quad_off)
    pars = mod.make_params(x0=22, a=1, b=1, c=1)

    result = mod.fit(ydat, pars, x=xdat)
    print(result.fit_report())

The natural width of the error function is about 2 ``x`` units, but you can
adjust this, shortening it with ``erf((x-x0)*2)`` to give a sharper
transition for example.
