"""
Concise nonlinear curve fitting.

"""

import lmfit
import inspect

class Model(object):

    def __init__(self, model_func, independent_vars):
        """Create a model.

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
        self.model_arg_names = inspect.getargspec(model_func)[0]
        self.param_names = set(self.model_arg_names) - set(independent_vars)
        self.model_func = model_func
        self.independent_vars = independent_vars

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

    def _build_residual(self, data):
        "Generate and return a residual function."
        def residual(params, *args, **kwargs):
            # Unpack Parameter objects into simple key -> value pairs,
            # and combine them with any non-parameter kwargs.
            params = {name: p.value for name, p in params.items()}
            kwargs = dict(params.items() + kwargs.items())
            f = self.model_func(*args, **kwargs)
            e = data - f
            return e
        return residual

    def fit(self, data, params=None, *args, **kwargs):
        """Fit the model to the data.

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

        """
        if params is None:
            params = self.params()

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & self.param_names
        for name in param_kwargs:
            if isinstance(kwargs[name], lmfit.Parameter):
                params[name] = kwargs[name]
            else:
                params[name] = lmfit.Parameter(name=name, value=kwargs[name])
            del kwargs[name]

        # If any parameter is not initialized raise a more helpful error.
        missing_param = set(params.keys()) != self.param_names
        blank_param = any([p.value is None for p in params.values()])
        if missing_param or blank_param:
            raise ValueError("Assign each parameter an initial value by " +
                             "passing Parameters or keyword arguments to " +
                             "fit().")

        result = lmfit.minimize(self._build_residual(data), params, 
                                args=args, kws=kwargs)
        return result
