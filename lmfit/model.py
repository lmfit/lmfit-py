"""
Concise nonlinear curve fitting.

"""

import warnings
import inspect
import copy
import lmfit
import numpy as np

# Use pandas.isnull if available. Fall back on numpy.isnan.
try:
    import pandas
except ImportError:
    isnull = np.isnan
else:
    isnull = pandas.isnull

# When handling missing data or data not the same size as independent vars,
# use pandas to align. If pandas is not available, data and vars must be the
# same size, but missing data can still be handled with masks.
try:
    import pandas
except ImportError:
    def _align(var, mask, data):
        if mask is not None:
            return var[mask]
        return var
else:
    def _align(var, mask, data):
        if isinstance(data, pandas.Series) and isinstance(var, pandas.Series):
            return var.reindex_like(data).dropna()
        elif mask is not None:
            return var[mask]
        else:
            return var

class Model(object):

    def __init__(self, func, independent_vars=[], missing='none'):
        """Create a model.
        Parameters
        ----------
        func: function
        independent_vars: list of strings, optional
            matching argument(s) to func
        missing: 'none', 'drop', or 'raise'
            'none': Do not check for null or missing values.
            'drop': Drop null or missing observations in data.
                Use pandas.isnull if pandas is available; otherwise,
                silently fall back to numpy.isnan.
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
        >>> my_model = Model(decay, independent_vars = 't')
        """
        self.model_arg_names = inspect.getargspec(func)[0]
        # The implicit magic in fit() requires us to disallow some
        # variable names.
        forbidden_args = ['data', 'sigma', 'params']
        for arg in forbidden_args:
            if arg in self.model_arg_names:
                raise ValueError("The model function cannot have an " +
                                 "argument named %s. " % arg +
                                 "Choose a different name.")
        self.param_names = set(self.model_arg_names) - set(independent_vars)
        self.func = func
        self.independent_vars = independent_vars
        if not missing in ['none', 'drop', 'raise']:
            raise ValueError("missing must be 'none', 'drop', or 'raise'.")
        self.missing = missing
        self._residual = self._build_residual()

    def params(self):
        """Return a blank copy of model params.

        Example
        -------
        >>> params = my_model.params()
        >>> params['N'].value = 1.0  # initial guess
        >>> params['tau'].value = 2.0  # initial guess
        >>> params['tau'].min = 0  # (optional) lower bound
        """
        params = lmfit.Parameters()
        [params.add(name) for name in self.param_names]
        return params

    def _build_residual(self):
        "Generate and return a residual function."
        def residual(params, *args, **kwargs):
            # Unpack Parameter objects into simple key -> value pairs,
            # and combine them with any non-parameter kwargs.
            data, sigma = args
            params = {name: p.value for name, p in params.items()}
            kwargs = dict(params.items() + kwargs.items())
            f = self.func(**kwargs)
            if sigma is None:
                e = f - data
            else:
                e = (f - data)/sigma
            return np.asarray(e)  # for compatibility with pandas.Series

        return residual

    def _handle_missing(self, data):
        if self.missing == 'raise':
            if np.any(isnull(data)):
                raise ValueError("Data contains a null value.")
        elif self.missing == 'drop':
            mask = ~isnull(data)
            if np.all(mask):
                return None # short-circuit this -- no missing values
            mask = np.asarray(mask)  # for compatibility with pandas.Series
            return mask

    def fit(self, data, params=None, sigma=None, **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        data: array-like
        params: Parameters object, optional
        sigma: array-like of same size as data
            used for weighted fit, sigma=1/weights
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
        >>> result = fit(my_model, data, tau=5, N=3, t=t)

        # Or, for more control, pass a Parameters object.
        # See docstring for Model.params()
        >>> result = fit(my_model, data, params, t=t)

        # Keyword arguments override Parameters.
        >>> result = fit(my_model, data, params, tau=5, t=t)

        Note
        ----
        All parameters, however passed, are copied on input, so the original
        Parameter objects are unchanged.

        """
        if params is None:
            params = self.params()
        else:
            params = copy.deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & self.param_names
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, lmfit.Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = copy.deepcopy(p)
            else:
                params[name] = lmfit.Parameter(name=name, value=p)
            del kwargs[name]

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs.keys():
            if not name in self.independent_vars:
                warnings.warn(UserWarning, "The keyword argument %s " % name +
                              "does not match any arguments of the model " +
                              "function. It will be ignored.")

        # If any parameter is not initialized raise a more helpful error.
        missing_param = set(params.keys()) != self.param_names
        blank_param = any([p.value is None for p in params.values()])
        if missing_param or blank_param:
            raise ValueError("Assign each parameter an initial value by " +
                             "passing Parameters or keyword arguments to " +
                             "fit().")

        # Handle null/missing values.
        mask = None
        if self.missing != 'none':
            mask = self._handle_missing(data)  # This can raise.
            if mask is not None:
                data = data[mask]
            if sigma is not None:
                sigma = _align(sigma, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            if not np.isscalar(self.independent_vars):  # just in case
                kwargs[var] = _align(kwargs[var], mask, data)

        result = lmfit.minimize(self._residual, params,
                                args=(data, sigma), kws=kwargs)
        return result
