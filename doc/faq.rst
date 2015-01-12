====================================
Frequently Asked Questions
====================================

A list of common questions.

How can I fit multi-dimensional data?
========================================

The fitting routines except data arrays that are 1 dimensional and double
precision.  So you need to convert the data and model (or the value
returned by the objective function) to be one dimensional by using
numpy's :meth:`numpy.ndarray.flatten`, for example::

    def residual(params, x, data=None):
        ....
        resid = calculate_multidim_residual()
        return resid.flatten()


How can I fit complex data?
===================================

As with working with multidimensional data, you need to convert your data
and model (or the value returned by the objective function) to be real.
One way to do this would be to use a function like this::

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
