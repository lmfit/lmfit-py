#!/usr/bin/env python

from inspect import getargspec
from .parameter import Parameters

def make_paras_and_func(fcn, x0, used_kwargs=None):
    """A function which takes a function a makes a parameters-dict for it.

    Takes the function fcn. A starting guess x0 for the
    non kwargs paramter must be also given. If kwargs
    are used, used_kwargs is dict were the keys are the
    used kwarg and the values are the starting values.
    """
    import inspect
    args = inspect.getargspec(fcn)
    defaults = args[-1]
    len_def = len(defaults) if defaults is not None else 0
    # have_defaults = args[-len(defaults):]

    args_without_defaults = len(args[0]) - len_def

    if len(x0) < args_without_defaults:
        raise ValueError('x0 to short')
    p = Parameters()
    for i, val in enumerate(x0):
        p.add(args[0][i], val)

    if used_kwargs:
        for arg, val in used_kwargs.items():
            p.add(arg, val)
    else:
        used_kwargs = {}

    def func(para):
        "wrapped func"
        kwdict = {}
        for arg in used_kwargs.keys():
            kwdict[arg] = para[arg].value

        vals = [para[i].value for i in p]
        return fcn(*vals[:len(x0)], **kwdict)

    return p, func


def wrap_function(fcn, x0=None, non_params=None):
    """wrap a function, transforming the functions positional arguments
    into a Parameters dictionay object.   Thea
    new wrapper function using these Parameters that can be used
    as an objective function.

    Arguments
    ---------
    fcn          function to be wrapped using positional arguments
    x0           list or array of starting values for positional arguments
    non_params   list or arguments that should *not* be tranformer to
                 Parameters, but be turned into keyword arguments.
    Returns
    -------
    params, wrapped_function

    Example
    -------
     >>> def lorentz(x, amp, cen, wid):
     ...     'standard definition of Lorentzian function'
     ...     return  amp/(1.0 + ((x-cen)/wid)**2)
     >>>
     >>> pars, fwrap = wrap_function(lorentz, x0=(5., 0., 1.2),
     ...                             non_params=['x'])
     >>> for p in pars.values(): print(p)
     <Parameter 'amp', 5.0, bounds=[-inf:inf]>
     <Parameter 'cen', 0.0, bounds=[-inf:inf]>
     <Parameter 'wid', 1.2, bounds=[-inf:inf]>

     >>> x = np.linspace(-5, 5, 101)
     >>> fwrap(pars, x=x)
    """
    x0 = x0 if x0 is not None else []
    non_params = non_params if non_params is not None else []
    required_kwargs = []
    kwargs = {}

    argspec = getargspec(fcn)
    defaults = argspec.defaults
    len_def = len(defaults) if defaults is not None else 0
    nposargs = len(argspec.args) - len_def

    p = Parameters()
    # positional arguments
    i = 0
    for parname in argspec.args[:nposargs]:
        if parname in non_params:
            required_kwargs.append(parname)
        else:
            val = 0.0
            if i < len(x0):
                val = float(x0[i])
                i += 1
            p.add(parname, val)
    # keyword arguments
    if len_def > 0:
        for ikw in range(len_def):
            parname = argspec.args[nposargs+ikw]
            defval = argspec.defaults[ikw]
            if len(x0) > nposargs and len(x0) > nposargs+ikw:
                defval = x0[nposargs+ikw]
            if (parname in non_params or
                not isinstance(defval, (int, float))):
                required_kwargs.append(parname)
                kwargs[parname] = defval
            else:
                p.add(parname, defval)

    def func(params, **kws):
        "wrapped function"
        vals = []
        kwdict = {}
        kwdict.update(kwargs)

        for varname in required_kwargs:
            if varname not in kws and varname not in kwargs:
                raise ValueError("No value for required argument %s" % varname)

        for varname in argspec.args:
            if varname in p:
                vals.append(p[varname].value)
            elif varname in kws:
                vals.append(kws.pop(varname))
            else:
                raise ValueError("No value for %s" % varname)

        kwdict.update(kws)
        return fcn(*vals, **kwdict)

    tmpl = "wrapping of %s for Parameters. Original doc:\n%s"
    func.__doc__ = tmpl % (fcn.__name__, fcn.__doc__)
    return p, func
