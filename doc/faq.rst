.. _faq_chapter:

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

then you need to install the ipywidgets package.   Try 'pip install ipywidgets'.




How can I fit multi-dimensional data?
========================================

The fitting routines accept data arrays that are 1 dimensional and double
precision.  So you need to convert the data and model (or the value
returned by the objective function) to be one dimensional.  A simple way to
do this is to use numpy's :numpydoc:`ndarray.flatten`, for example::

    def residual(params, x, data=None):
        ....
        resid = calculate_multidim_residual()
        return resid.flatten()

How can I fit multiple data sets?
========================================

As above, the fitting routines accept data arrays that are 1 dimensional
and double precision.  So you need to convert the sets of data and models
(or the value returned by the objective function) to be one dimensional.  A
simple way to do this is to use numpy's :numpydoc:`concatenate`.  As an
example, here is a residual function to simultaneously fit two lines to two
different arrays.  As a bonus, the two lines share the 'offset' parameter::

    def fit_function(params, x=None, dat1=None, dat2=None):
        model1 = params['offset'] + x * params['slope1']
        model2 = params['offset'] + x * params['slope2']

	resid1 = dat1 - model1
        resid2 = dat2 - model2
        return numpy.concatenate((resid1, resid2))



How can I fit complex data?
===================================

As with working with multidimensional data, you need to convert your data
and model (or the value returned by the objective function) to be double
precision floating point numbers. The simplest approach is to use numpy's
:numpydoc:`ndarray.view` method, perhaps like::

   import numpy as np
   def residual(params, x, data=None):
        ....
        resid = calculate_complex_residual()
        return resid.view(np.float)


Can I constrain values to have integer values?
===============================================

Basically, no.  None of the minimizers in lmfit support integer
programming.  They all (I think) assume that they can make a very small
change to a floating point value for a parameters value and see a change in
the value to be minimized.


How should I cite LMFIT?
==================================

See http://dx.doi.org/10.5281/zenodo.11813
