LMfit-py provides a Least-Squares Minimization routine and class
with a simple, flexible approach to parameterizing a model for
fitting to data.  Named Parameters can be held fixed or freely
adjusted in the fit, or held between lower and upper bounds.  In
addition, parameters can be constrained as a simple mathematical
expression of other Parameters.

To do this, the programmer defines a Parameters object, an enhanced
dictionary, containing named parameters:

    fit_params = Parameters()
    fit_params['amp'] = Parameter(value=1.2, min=0.1, max=1000)
    fit_params['cen'] = Parameter(value=40.0, vary=False),
    fit_params['wid'] = Parameter(value=4, min=0)}

or using the equivalent

    fit_params = Parameters()
    fit_params.add('amp', value=1.2, min=0.1, max=1000)
    fit_params.add('cen', value=40.0, vary=False),
    fit_params.add('wid', value=4, min=0)

The programmer will also write a function to be minimized (in the
least-squares sense) with its first argument being this Parameters object,
and additional positional and keyword arguments as desired:

  def myfunc(params, x, data, someflag=True):
      amp = params['amp'].value
      cen = params['cen'].value
      wid = params['wid'].value
      ...
      return residual_array

For each call of this function, the values for the params may have changed,
subject to the bounds and constraint settings for each Parameter.  The function
should return the residual (ie, data-model) array to be minimized.

The advantage here is that the function to be minimized does not have to be
changed if different bounds or constraints are placed on the fitting
Parameters.  The fitting model (as described in myfunc) is instead written
in terms of physical parameters of the system, and remains remains
independent of what is actually varied in the fit.  In addition, which
parameters are adjuested and which are fixed happens at run-time, so that
changing what is varied and what constraints are placed on the parameters
can easily be modified by the consumer in real-time data analysis.

To perform the fit, the user calls
  result = minimize(myfunc, fit_params, args=(x, data), kws={'someflag':True}, ....)

After the fit, each real variable in the fit_params dictionary is updated
to have best-fit values, estimated standard deviations, and correlations
with other variables in the fit, while the results dictionary holds fit
statistics and information.

By default, the underlying fit algorithm is the Levenberg-Marquart
algorithm with numerically-calculated derivatives from MINPACK's lmdif
function, as used by scipy.optimize.leastsq.  Other solvers (currently
Simulated Annealing and L-BFGS-B) are also available, though slightly less
well-tested and supported.
