"""
Concise nonlinear curve fitting.
"""

import warnings
import inspect
import copy
import numpy as np
from . import Parameters, Parameter, minimize
from .printfuncs import fit_report

# Use pandas.isnull for aligning missing data is pandas is available.
# otherwise use numpy.isnan
try:
    from pandas import isnull, Series
except ImportError:
    isnull = np.isnan
    Series = type(NotImplemented)

def _align(var, mask, data):
    "align missing data, with pandas is available"
    if isinstance(data, Series) and isinstance(var, Series):
        return var.reindex_like(data).dropna()
    elif mask is not None:
        return var[mask]
    return var

class Model(object):
    """Create a model from a user-defined function.

    Parameters
    ----------
    func: function to be wrapped
    independent_vars: list of strings or None (default)
        arguments to func that are independent variables
    param_names: list of strings or None (default)
        names of arguments to func that are to be made into parameters
    missing: None, 'none', 'drop', or 'raise'
        'none' or None: Do not check for null or missing values (default)
        'drop': Drop null or missing observations in data.
            if pandas is installed, pandas.isnull is used, otherwise
            numpy.isnan is used.
        'raise': Raise a (more helpful) exception when data contains null
            or missing values.

    Note
    ----
    Parameter names are inferred from the function arguments,
    and a residual function is automatically constructed.

    Example
    -------
    >>> def decay(t, tau, N):
    ...     return N*np.exp(-t/tau)
    ...
    >>> my_model = Model(decay, independent_vars=['t'])
    """

    _forbidden_args = ('data', 'weights', 'params')
    _invalid_ivar  = "Invalid independent variable name ('%s') for function %s"
    _invalid_par   = "Invalid parameter name ('%s') for function %s"
    _invalid_missing = "missing must be None, 'none', 'drop', or 'raise'."
    def __init__(self, func, independent_vars=None, param_names=None,
                 missing='none', prefix='', components=None, **kws):
        self.func = func
        self.prefix = prefix
        self.param_names = param_names
        self.independent_vars = independent_vars
        self.func_allargs = []
        self.func_haskeywords = False
        self.has_initial_guess = False
        self.components = components
        if not missing in [None, 'none', 'drop', 'raise']:
            raise ValueError(self._invalid_missing)
        self.missing = missing
        self.opts = kws
        self.result = None
        self.params = Parameters()
        self._parse_params()
        self._residual = self._build_residual()
        if self.independent_vars is None:
            self.independent_vars = []

    def _parse_params(self):
        "build params from function arguments"
        argspec = inspect.getargspec(self.func)
        pos_args = argspec.args[:]
        keywords = argspec.keywords
        kw_args = {}
        if argspec.defaults is not None:
            for val in reversed(argspec.defaults):
                kw_args[pos_args.pop()] = val
        #
        self.func_haskeywords = keywords is not None
        self.func_allargs = pos_args + list(kw_args.keys())
        allargs = self.func_allargs

        if len(allargs) == 0 and keywords is not None:
            return

        # default independent_var = 1st argument
        if self.independent_vars is None:
            self.independent_vars = [pos_args[0]]

        # default param names: all positional args
        # except independent variables
        def_vals = {}
        if self.param_names is None:
            self.param_names = pos_args[:]
            for key, val in kw_args.items():
                if (not isinstance(val, bool) and
                    isinstance(val, (float, int))):
                    self.param_names.append(key)
                    def_vals[key] = val
            for p in self.independent_vars:
                if p in self.param_names:
                    self.param_names.remove(p)

        # check variables names for validity
        # The implicit magic in fit() requires us to disallow some
        fname = self.func.__name__
        for arg in self.independent_vars:
            if arg not in allargs or arg in self._forbidden_args:
                raise ValueError(self._invalid_ivar % (arg, fname))
        for arg in self.param_names:
            if arg not in allargs or arg in self._forbidden_args:
                raise ValueError(self._invalid_par % (arg, fname))

        names = []
        for pname in self.param_names:
            if not pname.startswith(self.prefix):
                pname = "%s%s" % (self.prefix, pname)
            names.append(pname)
        self.param_names = set(names)
        for name in self.param_names:
            self.params.add(name)
        for key, val in def_vals.items():
            self.set_paramval(key, val)

    def guess_starting_values(self, data=None, **kws):
        """stub for guess starting values --
        should be implemented for each model subclass
        """
        cname = self.__class__.__name__
        msg = 'guess_starting_values() not implemented for %s' % cname
        raise NotImplementedError(msg)

    def _build_residual(self):
        "Generate and return a residual function."
        def residual(params, data, weights, **kwargs):
            "default residual:  (data-model)*weights"
            diff = self.eval(params=params, **kwargs) - data
            if weights is not None:
                diff *= weights
            return np.asarray(diff)  # for compatibility with pandas.Series
        return residual

    def make_funcargs(self, params, kwargs):
        """convert parameter values and keywords to function arguments"""
        out = {}
        out.update(self.opts)
        npref = len(self.prefix)
        for name, par in params.items():
            if npref > 0 and name.startswith(self.prefix):
                name = name[npref:]
            if name in self.func_allargs or self.func_haskeywords:
                out[name] = par.value
        for name, val in kwargs.items():
            if name in self.func_allargs or self.func_haskeywords:
                out[name] = val
                if name in params:
                    params[name].value = val
        if self.func_haskeywords and self.components is not None:
            out['__components__'] = self.components
        return out

    def _handle_missing(self, data):
        "handle missing data"
        if self.missing == 'raise':
            if np.any(isnull(data)):
                raise ValueError("Data contains a null value.")
        elif self.missing == 'drop':
            mask = ~isnull(data)
            if np.all(mask):
                return None  # short-circuit this -- no missing values
            mask = np.asarray(mask)  # for compatibility with pandas.Series
            return mask

    def set_paramval(self, paramname, value, min=None, max=None, vary=True):
        """set parameter value, as for initial guess.
        name can include prefix or not
        """
        pname = paramname
        if pname not in self.params:
            pname = "%s%s" % (self.prefix, pname)
        if pname not in self.params:
            raise KeyError("%s not a parameter name")
        self.params[pname].value = value
        self.params[pname].vaary = vary
        if min is not None:
            self.params[pname].min = min
        if max is not None:
            self.params[pname].max = max

    def eval(self, params=None, **kwargs):
        """evaluate the model with the supplied or current parameters"""
        if params is None:
            params = self.params
        fcnargs = self.make_funcargs(params, kwargs)
        return self.func(**fcnargs)

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        data: array-like
        params: Parameters object, optional
        weights: array-like of same size as data
            used for weighted fit
        method: fitting method to use (default = 'leastsq')
        iter_cb:  None or callable  callback function to call at each iteration.
        scale_covar:  bool (default True) whether to auto-scale covariance matrix
        keyword arguments: optional, named like the arguments of the
            model function, will override params. See examples below.

        Returns
        -------
        lmfit.Minimizer

        Examples
        --------
        # Take t to be the independent variable and data to be the
        # curve we will fit.

        # Using keyword arguments to set initial guesses
        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        # Or, for more control, pass a Parameters object.
        >>> result = my_model.fit(data, params, t=t)

        # Keyword arguments override Parameters.
        >>> result = my_model.fit(data, params, tau=5, t=t)

        Note
        ----
        All parameters, however passed, are copied on input, so the original
        Parameter objects are unchanged.

        """
        if params is None:
            params = self.params
        else:
            params = copy.deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & self.param_names
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = copy.deepcopy(p)
            else:
                params[name] = Parameter(name=name, value=p)
            del kwargs[name]

        # Keep a pristine copy of the initial params.
        init_params = copy.deepcopy(params)

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs.keys():
            if not name in self.independent_vars:
                warnings.warn("The keyword argument %s does not" % name +
                              "match any arguments of the model function." +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any([p not in params.keys()
                             for p in self.param_names])
        blank_param = any([(p.value is None and p.expr is None)
                           for p in params.values()])
        if missing_param or blank_param:
            raise ValueError("""Assign each parameter an initial value by
 passing Parameters or keyword arguments to fit""")


        # Handle null/missing values.
        mask = None
        if self.missing not in (None, 'none'):
            mask = self._handle_missing(data)  # This can raise.
            if mask is not None:
                data = data[mask]
            if weights is not None:
                weights = _align(weights, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            if not np.isscalar(self.independent_vars):  # just in case
                kwargs[var] = _align(kwargs[var], mask, data)

        kwargs['__components__'] = self.components
        result = minimize(self._residual, params, args=(data, weights),
                          method=method, iter_cb=iter_cb,
                          scale_covar=scale_covar, kws=kwargs)

        # Monkey-patch the Minimizer object with some extra information.
        result.model = self
        result.init_params = init_params
        result.init_values = self.make_funcargs(init_params, {})
        if '__components__' in result.init_values:
            result.init_values.pop('__components__')
        result.init_fit = self.eval(params=init_params, **kwargs)
        result.best_fit = self.eval(params=result.params, **kwargs)
        self.result = result
        return result

    def fit_report(self, modelpars=None, show_correl=True, min_correl=0.1):
        "return fit report"
        if self.result is None:
            raise ValueError("must run .fit() first")

        return fit_report(self.result, modelpars=modelpars,
                          show_correl=show_correl,
                          min_correl=min_correl)

    def __add__(self, other):
        colliding_param_names = self.param_names & other.param_names
        if len(colliding_param_names) != 0:
            collision = colliding_param_names.pop()
            raise NameError("Both models have parameters called " +
                            "%s. Redefine the models " % collision +
                            "with distinct names.")

        def composite_func(**kwargs):
            "composite model function"
            components = kwargs.get('__components__', None)
            out = None
            if components is not None:
                for comp in components:
                    pars = Parameters()
                    prefix = comp.prefix
                    for p in self.params.values():
                        if p.name.startswith(prefix):
                            pars.__setitem__(p.name, p)
                            pars[p.name].value = kwargs[p.name]

                    fcnargs = comp.make_funcargs(pars, kwargs)
                    comp_out = comp.func(**fcnargs)
                    if out is None:
                        out = np.zeros_like(comp_out)
                    out += comp_out
            return out

        components = self.components
        if components is None:
            components = [self]
        if other.components is None:
            components.append(other)
        else:
            components.extend(other.components)
        all_params = self.params
        for key, val in other.params.items():
            all_params[key] = val

        out = Model(func=composite_func, independent_vars=self.independent_vars,
                    param_names=self.param_names | other.param_names,
                    missing=self.missing, components=components)
        out.components = components
        out.params = all_params
        return out
