===========================================================
Getting started with Non-Linear Least-Squares Fitting
===========================================================

The lmfit package is designed to provide simple tools to help you build of
complex fitting models for non-linear least-squares problems and apply
these models to real data.  This section gives an overview of the concepts
and describes how to set up and perform simple fits.  Some basic knowledge
of Python, numpy, and modeling data are assumed.

To do a non-linear least-squares fit of a model to data or for a variety of other
optimization problems, the main task is to write an *objective function*
that takes the values of the fitting variables and calculates either a
scalar value to be minimized or an array of values that is to be minimized
in the least-squares sense.   For many data fitting processes, the
least-squares approach is used, and the objective function should
return an array of (data-model), perhaps scaled by some weighting factor
such as the inverse of the uncertainty in the data.  For such a problem,
the chi-square (:math:`\chi^2`) statistic is often defined as:

.. math::

 \chi^2 =  \sum_i^{N} \frac{[y^{\rm meas}_i - y_i^{\rm model}({\bf{v}})]^2}{\epsilon_i^2}

where :math:`y_i^{\rm meas}` is the set of measured data, :math:`y_i^{\rm
model}({\bf{v}})` is the model calculation, :math:`{\bf{v}}` is the set of
variables in the model to be optimized in the fit, and :math:`\epsilon_i`
is the estimated uncertainty in the data.

In a traditional non-linear fit, one writes an objective function that takes the
variable values and calculates the residual :math:`y^{\rm meas}_i -
y_i^{\rm model}({\bf{v}})`, or the residual scaled by the data
uncertainties, :math:`[y^{\rm meas}_i - y_i^{\rm
model}({\bf{v}})]/{\epsilon_i}`, or some other weighting factor.  As a
simple example, one might write an objective function like this::

    def residual(vars, x, data, eps_data):
        amp = vars[0]
        phaseshift = vars[1]
	freq = vars[2]
        decay = vars[3]

	model = amp * sin(x * freq  + phaseshift) * exp(-x*x*decay)

        return (data-model)/eps_data

To perform the minimization with scipy, one would do::

    from scipy.optimize import leastsq
    vars = [10.0, 0.2, 3.0, 0.007]
    out = leastsq(residual, vars, args=(x, data, eps_data))

Though it is wonderful to be able to use python for such optimization
problems, and the scipy library is robust and easy to use, the approach
here is not terribly different from how one would do the same fit in C or
Fortran.

