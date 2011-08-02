Minimizer provides a simple, flexible interface to a Least Squares Fit.

With Minimizer, the user creates a model that is a function of a set of
named Parameters.  Each Parameter can be fixed, adjusted in the fit, or
constrained as a simple mathematical expression of other Parameters.
In addition, upper and lower Bounds can be placed on any Parameter.

The user defines a Python dictionary of named Parameters, such as:

fit_params = {'amp': Parameter(value=1.2, min=0.1, max=1000),
              'cen': Parameter(value=40.0, vary=False),
              'wid': Parameter(value=4, min=0)}

The user also writes a function to be minimized (in the least-squares sense)
with its first argument being a dictionary of Parameter names and values, and
additional positional and keyword arguments as needed:

  def myfunc(params, x, data, someflag=True):
      amp = params['amp'].value
      cen = params['cen'].value
      wid = params['wid'].value
      ...
      return residual_array

For each call of this function, the values for the params may have changed,
subject to the bounds and constraint settings for each Parameter.  The function
should return the residual array (ie, data-model) to be minimized.

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

Initially, the underlying fit algorithm is the Levenberg-Marquart algorithm with
numerically-calculated derivatives from MINPACK's lmdif function, as used by
scipy.optimize.leastsq.  Other solvers will be considered once the code is
working well.
