Minimizer provides a simple, flexible interface to a Least Squares Fit.

With Minimizer, the user creates a model that is a function of a set of
named Parameters.  Each Parameter can be fixed, adjusted in the fit, or
constrained as a simple mathematical expression of other Parameters.
In addition, upper and lower Bounds can be placed on any Parameter.

The user defines a Python dictionary of named Parameters, such as:

FitParams = {'amp': Parameter(value=1.2, float=True, min=0.1, max=1000),
	     'cen': Parameter(value=40.0, float=False),
	     'wid': Parameter(value=4, float=True, min=0)}

The user also writes a function to be minimized (in the least-squares sense)
with its first argument being a dictionary of Paramter names and values.

  def myfunc(params, x, data, ...):
      amp = params['amp']
      cen = params['cen']
      wid = params['wid']
      ...

which may be updated with new values on each call.

The user then calls

   minimize(myfunc, FitParams, args=(x, data), ....)

to perform the fit.

Initially, the underlying fit algorithm is the Levenberg-Marquart algorithm
with numerically-calculated derivatives from MINPACK's lmdif function, as
used by scipy.optimize.leastsq.  Other solvers will be considered once the
code is working well.
