"""
Concise nonlinear curve fitting.
"""

import warnings
import inspect
import copy
import numpy as np
from . import Parameters, Parameter, minimize

# Use pandas.isnull for aligning missing data is pandas is available.
# otherwise use numpy.isnan
try:
    from upandas import isnull, Series
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

def funcargs(func):
    """return names for positional arguments and
    dict of keyword arguments for a function.
    """
    argspec = inspect.getargspec(func)
    posargs = argspec.args[:]
    kwargs = {}
    if argspec.defaults is not None:
        for val in reversed(argspec.defaults):
            kwargs[posargs.pop()] = val
    return (posargs, kwargs, argspec.keywords)

class Model(object):
    _forbidden_args = ('data', 'weights', 'params')
    _invalid_ivar  = "Invalid independent variable name ('%s') for function %s"
    _invalid_par   = "Invalid parameter name ('%s') for function %s"

    def __init__(self, func, independent_vars=None, param_names=None,
                 missing=None, components=None):
        """Create a model from a user-defined function.

        Parameters
        ----------
        func: function to be wrapped
        independent_vars: list of strings or None (default)
            arguments to func that are independent variables
        param_names: list of strings or None (default)
            names of arguments to func that are to be made into parameters
        missing: None, 'drop', or 'raise'
            None: Do not check for null or missing values (default)
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
        self.func = func
        self.param_names = param_names
        self.independent_vars = independent_vars
        self.components = components
        if not missing in [None, 'drop', 'raise']:
            raise ValueError("missing must be None, 'drop', or 'raise'.")
        self.missing = missing
        self._parse_params()
        self._residual = self._build_residual()
        if self.independent_vars is None:
            self.independent_vars = []

    def _parse_params(self):
        pos_args, kw_args, keywords = funcargs(self.func)
        all_args = pos_args + kw_args.keys()
        if len(all_args) == 0 and keywords is not None:
            return

        # default independent_var = 1st argument
        if self.independent_vars is None:
            self.independent_vars = [pos_args[0]]

        # default param names: all positional args
        # except independent variables
        if self.param_names is None:
            self.param_names = pos_args[:]
            for key, val in kw_args.items():
                if isinstance(val, (float, int)):
                    self.param_names.append(key)
            for p in self.independent_vars:
                if p in self.param_names:
                    self.param_names.remove(p)

        # check variables names for validity
        # The implicit magic in fit() requires us to disallow some
        all_args = pos_args + kw_args.keys()
        for arg in self.independent_vars:
            if arg not in all_args or arg in self._forbidden_args:
                raise ValueError(self._invalid_ivar % (arg, self.func.__name__))
        for arg in self.param_names:
            if arg not in all_args or arg in self._forbidden_args:
                raise ValueError(self._invalid_par % (arg, self.func.__name__))
        self.param_names = set(self.param_names)


    def guess_starting_values(self, values, **kws):
        raise NotImplementedError('must overwrite guess_starting_values')

    def __set_param_names(self, param_names):
        # used when models are added
        self.param_names = param_names

    def params(self):
        """Return a blank copy of model params.

        Example
        -------
        >>> params = my_model.params()
        >>> params['N'].value = 1.0  # initial guess
        >>> params['tau'].value = 2.0  # initial guess
        >>> params['tau'].min = 0  # (optional) lower bound
        """
        params = Parameters()
        [params.add(name) for name in self.param_names]
        return params

    def _build_residual(self):
        "Generate and return a residual function."
        def residual(params, *args, **kwargs):
            # Unpack Parameter objects into simple key -> value pairs,
            # and combine them with any non-parameter kwargs.
            data, weights = args
            params = dict([(name, p.value) for name, p in params.items()])
            kwargs = dict(list(params.items()) + list(kwargs.items()))
            f = self.func(**kwargs)
            if weights is None:
                e = f - data
            else:
                e = (f - data)*weights
            return np.asarray(e)  # for compatibility with pandas.Series
        return residual

    def _handle_missing(self, data):
        if self.missing == 'raise':
            if np.any(isnull(data)):
                raise ValueError("Data contains a null value.")
        elif self.missing == 'drop':
            mask = ~isnull(data)
            if np.all(mask):
                return None  # short-circuit this -- no missing values
            mask = np.asarray(mask)  # for compatibility with pandas.Series
            return mask

    def fit(self, data, params=None, weights=None, **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        data: array-like
        params: Parameters object, optional
        weights: array-like of same size as data
            used for weighted fit
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
        # See docstring for Model.params()
        >>> result = my_model.fit(data, params, t=t)

        # Keyword arguments override Parameters.
        >>> result = my_model.fit(data, params, tau=5, t=t)

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
        missing_param = set(params.keys()) != self.param_names
        blank_param = any([p.value is None for p in params.values()])
        if missing_param or blank_param:
            raise ValueError("Assign each parameter an initial value by " +
                             "passing Parameters or keyword arguments to " +
                             "fit().")

        # Handle null/missing values.
        mask = None
        if self.missing is not None:
            mask = self._handle_missing(data)  # This can raise.
            if mask is not None:
                data = data[mask]
            if weights is not None:
                weights = _align(weights, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            #if var not in kwargs:
            #    raise ValueError("Must include independent variable ('%s') to fit()" % var)

            if not np.isscalar(self.independent_vars):  # just in case
                kwargs[var] = _align(kwargs[var], mask, data)

        result = minimize(self._residual, params,
                          args=(data, weights), kws=kwargs)

        # Monkey-patch the Minimizer object with some extra information.
        result.model = self
        result.init_params = init_params
        result.init_values = dict([(name, p.value) for name, p
                                  in init_params.items()])
        indep_vars = dict([(k, v) for k, v in kwargs.items() if k in
                           self.independent_vars])
        evaluation_kwargs = dict(list(indep_vars.items()) +
                                 list(result.init_values.items()))
        result.init_fit = self.func(**evaluation_kwargs)
        evaluation_kwargs = dict(list(indep_vars.items()) +
                                 list(result.values.items()))
        result.best_fit = self.func(**evaluation_kwargs)
        return result

    def __add__(self, other):
        colliding_param_names = self.param_names & other.param_names
        if len(colliding_param_names) != 0:
            collision = colliding_param_names.pop()
            raise NameError("Both models have parameters called " +
                            "%s. Redefine the models " % collision +
                            "with distinct names.")
        # if self.independent_vars != other.independent_vars:
        #     raise ValueError("Both models need to have identical " +
        #                      "independent variables ")

        def func(**kwargs):
            self_kwargs = dict([(k, kwargs.get(k)) for k in
                                self.param_names | set(self.independent_vars)])
            other_kwargs = dict([(k, kwargs.get(k)) for k in
                                 other.param_names | set(other.independent_vars)])
            return self.func(**self_kwargs) + other.func(**other_kwargs)
        components = []
        out = Model(func=func, independent_vars=self.independent_vars,
                    param_names=self.param_names | other.param_names,
                    missing=self.missing, components=(self, other))

        return out
