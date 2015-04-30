====================================
Frequently Asked Questions
====================================

A list of common questions.

How can I fit multi-dimensional data?
========================================

The fitting routines accept data arrays that are 1 dimensional and double
precision.  So you need to convert the data and model (or the value
returned by the objective function) to be one dimensional.  A simple way to 
do this is to use numpy's :meth:`numpy.ndarray.flatten`, for example::

    def residual(params, x, data=None):
        ....
        resid = calculate_multidim_residual()
        return resid.flatten()

How can I fit multiple data sets?
========================================

As above, the fitting routines accept data arrays that are 1 dimensional and double
precision.  So you need to convert the sets of data and models (or the value
returned by the objective function) to be one dimensional.  A simple way to 
do this is to use numpy's :meth:`numpy.concatenate`.  As an example, here
is a residual function to simultaneously fit two lines to two different
arrays.  As a bonus, the two lines share the 'offset' parameter:

    def fit_function(params, x=None, dat1=None, dat2=None):
        model1 = params['offset'].value + x * params['slope1'].value
        model2 = params['offset'].value + x * params['slope2'].value

	resid1 = dat1 - model1
        resid2 = dat2 - model2
        return numpy.concatenate((resid1, resid2)) 



How can I fit complex data?
===================================

As with working with multidimensional data, you need to convert your data
and model (or the value returned by the objective function) to be double precision
floating point numbers. One way to do this would be to use a function like this::

    def realimag(array):
        return np.array([(x.real, x.imag) for x in array]).flatten()

to convert the complex array into an array of alternating real and
imaginary values.  You can then use this function on the result returned by
your objective function::

    def residual(params, x, data=None):
        ....
        resid = calculate_complex_residual()
        return realimag(resid)


Can I constrain values to have integer values?
===============================================

Basically, no.  None of the minimizers in lmfit support integer
programming.  They all (I think) assume that they can make a very small
change to a floating point value for a parameters value and see a change in
the value to be minimized.


How should I cite LMFIT?
==================================

See http://dx.doi.org/10.5281/zenodo.11813 

