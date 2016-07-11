"""
Concise nonlinear curve fitting.
"""
from __future__ import print_function
import warnings
import inspect
import operator
from copy import deepcopy
import numpy as np
from . import Parameters, Parameter, Minimizer
from .printfuncs import fit_report, ci_report
from .confidence import conf_interval

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

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


try:
    from matplotlib import pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _ensureMatplotlib(function):
    if _HAS_MATPLOTLIB:
        return function
    else:
        def no_op(*args, **kwargs):
            print('matplotlib module is required for plotting the results')

        return no_op


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
    name: None or string
        name for the model. When `None` (default) the name is the same as
        the model function (`func`).

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
    _valid_missing   = (None, 'none', 'drop', 'raise')

    _invalid_hint = "unknown parameter hint '%s' for param '%s'"
    _hint_names = ('value', 'vary', 'min', 'max', 'expr')

    def __init__(self, func, independent_vars=None, param_names=None,
                 missing='none', prefix='', name=None, **kws):
        self.func = func
        self._prefix = prefix
        self._param_root_names = param_names  # will not include prefixes
        self.independent_vars = independent_vars
        self._func_allargs = []
        self._func_haskeywords = False
        if not missing in self._valid_missing:
            raise ValueError(self._invalid_missing)
        self.missing = missing
        self.opts = kws
        self.param_hints = OrderedDict()
        # the following has been changed from OrderedSet for the time being
        self._param_names = []
        self._parse_params()
        if self.independent_vars is None:
            self.independent_vars = []
        if name is None and hasattr(self.func, '__name__'):
            name = self.func.__name__
        self._name = name

    def _reprstring(self, long=False):
        out = self._name
        opts = []
        if len(self._prefix) > 0:
            opts.append("prefix='%s'" % (self._prefix))
        if long:
            for k, v in self.opts.items():
                opts.append("%s='%s'" % (k, v))
        if len(opts) > 0:
            out = "%s, %s" % (out, ', '.join(opts))
        return "Model(%s)" % out

    @property
    def name(self):
        return self._reprstring(long=False)

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def prefix(self):
        return self._prefix

    @property
    def param_names(self):
        return self._param_names

    def __repr__(self):
        return "<lmfit.Model: %s>" % (self.name)

    def copy(self, **kwargs):
        """DOES NOT WORK"""
        raise NotImplementedError("Model.copy does not work. Make a new Model")

    def _parse_params(self):
        "build params from function arguments"
        if self.func is None:
            return
        argspec = inspect.getargspec(self.func)
        pos_args = argspec.args[:]
        keywords = argspec.keywords
        kw_args = {}
        if argspec.defaults is not None:
            for val in reversed(argspec.defaults):
                kw_args[pos_args.pop()] = val

        self._func_haskeywords = keywords is not None
        self._func_allargs = pos_args + list(kw_args.keys())
        allargs = self._func_allargs

        if len(allargs) == 0 and keywords is not None:
            return

        # default independent_var = 1st argument
        if self.independent_vars is None:
            self.independent_vars = [pos_args[0]]

        # default param names: all positional args
        # except independent variables
        self.def_vals = {}
        might_be_param = []
        if self._param_root_names is None:
            self._param_root_names = pos_args[:]
            for key, val in kw_args.items():
                if (not isinstance(val, bool) and
                    isinstance(val, (float, int))):
                    self._param_root_names.append(key)
                    self.def_vals[key] = val
                elif val is None:
                    might_be_param.append(key)
            for p in self.independent_vars:
                if p in self._param_root_names:
                    self._param_root_names.remove(p)

        new_opts = {}
        for opt, val in self.opts.items():
            if (opt in self._param_root_names or opt in might_be_param and
                isinstance(val, Parameter)):
                self.set_param_hint(opt, value=val.value,
                                    min=val.min, max=val.max, expr=val.expr)
            elif opt in self._func_allargs:
                new_opts[opt] = val
        self.opts = new_opts

        names = []
        if self._prefix is None:
            self._prefix = ''
        for pname in self._param_root_names:
            names.append("%s%s" % (self._prefix, pname))

        # check variables names for validity
        # The implicit magic in fit() requires us to disallow some
        fname = self.func.__name__
        for arg in self.independent_vars:
            if arg not in allargs or arg in self._forbidden_args:
                raise ValueError(self._invalid_ivar % (arg, fname))
        for arg in names:
            if (self._strip_prefix(arg) not in allargs or
                arg in self._forbidden_args):
                raise ValueError(self._invalid_par % (arg, fname))
        # the following as been changed from OrderedSet for the time being.
        self._param_names = names[:]

    def set_param_hint(self, name, **kwargs):
        """set hints for parameter, including optional bounds
        and constraints  (value, vary, min, max, expr)
        these will be used by make_params() when building
        default parameters

        example:
          model = GaussianModel()
          model.set_param_hint('amplitude', min=-100.0, max=0.)
        """
        npref = len(self._prefix)
        if npref > 0 and name.startswith(self._prefix):
            name = name[npref:]

        if name not in self.param_hints:
            self.param_hints[name] = OrderedDict()

        for key, val in kwargs.items():
            if key in self._hint_names:
                self.param_hints[name][key] = val
            else:
                warnings.warn(self._invalid_hint % (key, name))

    def print_param_hints(self, colwidth=8):
        """Prints a nicely aligned text-table of parameters hints.

        The argument `colwidth` is the width of each column,
        except for first and last columns.
        """
        name_len = max(len(s) for s in self.param_hints)
        print('{:{name_len}}  {:>{n}} {:>{n}} {:>{n}} {:>{n}}    {:{n}}'
              .format('Name', 'Value', 'Min', 'Max', 'Vary', 'Expr',
                      name_len=name_len, n=colwidth))
        line = ('{name:<{name_len}}  {value:{n}g} {min:{n}g} {max:{n}g} '
                '{vary!s:>{n}}    {expr}')
        for name, values in sorted(self.param_hints.items()):
            pvalues = dict(name=name, value=np.nan, min=-np.inf, max=np.inf,
                           vary=True, expr='')
            pvalues.update(**values)
            print(line.format(name_len=name_len, n=colwidth, **pvalues))

    def make_params(self, verbose=False, **kwargs):
        """create and return a Parameters object for a Model.
        This applies any default values
        """
        params = Parameters()

        # make sure that all named parameters are in params
        for name in self.param_names:
            if name in params:
                par = params[name]
            else:
                par = Parameter(name=name)
            par._delay_asteval = True
            basename = name[len(self._prefix):]
            # apply defaults from model function definition
            if basename in self.def_vals:
                par.value = self.def_vals[basename]
            if par.value in (None, -np.inf, np.inf, np.nan):
                for key, val in self.def_vals.items():
                    if key in name.lower():
                        par.value = val
            # apply defaults from parameter hints
            if basename in self.param_hints:
                hint = self.param_hints[basename]
                for item in self._hint_names:
                    if item in hint:
                        setattr(par, item, hint[item])
            # apply values passed in through kw args
            if basename in kwargs:
                # kw parameter names with no prefix
                par.value = kwargs[basename]
            if name in kwargs:
                # kw parameter names with prefix
                par.value = kwargs[name]
            params.add(par)
            if verbose:
                print( ' - Adding parameter "%s"' % name)

        # next build parameters defined in param_hints
        # note that composites may define their own additional
        # convenience parameters here
        for basename, hint in self.param_hints.items():
            name = "%s%s" % (self._prefix, basename)
            if name in params:
                par = params[name]
            else:
                par = Parameter(name=name)
                params.add(par)
                if verbose:
                    print( ' - Adding parameter for hint "%s"' % name)
            par._delay_asteval = True
            for item in self._hint_names:
                if item in  hint:
                    setattr(par, item, hint[item])
            # Add the new parameter to self._param_names
            if name not in self._param_names:
                self._param_names.append(name)

        for p in params.values():
            p._delay_asteval = False
        return params

    def guess(self, data=None, **kws):
        """stub for guess starting values --
        should be implemented for each model subclass to
        run self.make_params(), update starting values
        and return a Parameters object"""
        cname = self.__class__.__name__
        msg = 'guess() not implemented for %s' % cname
        raise NotImplementedError(msg)

    def _residual(self, params, data, weights, **kwargs):
        """default residual:  (data-model)*weights

        If the model returns complex values, the residual is computed by treating the real and imaginary
        parts separately. In this case, if the weights provided are real, they are assumed to apply equally to the
        real and imaginary parts. If the weights are complex, the real part of the weights are applied to the real
        part of the residual and the imaginary part is treated correspondingly.

        Since the underlying scipy.optimize routines expect np.float arrays, the only complex type supported is
        np.complex.

        The "ravels" throughout are necessary to support pandas.Series.
        """
        diff = self.eval(params, **kwargs) - data

        if diff.dtype == np.complex:
            # data/model are complex
            diff = diff.ravel().view(np.float)
            if weights is not None:
                if weights.dtype == np.complex:
                    # weights are complex
                    weights = weights.ravel().view(np.float)
                else:
                    # real weights but complex data
                    weights = (weights + 1j * weights).ravel().view(np.float)
        if weights is not None:
            diff *= weights
        return np.asarray(diff).ravel()  # for compatibility with pandas.Series

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

    def _strip_prefix(self, name):
        npref = len(self._prefix)
        if npref > 0 and name.startswith(self._prefix):
            name = name[npref:]
        return name

    def make_funcargs(self, params=None, kwargs=None, strip=True):
        """convert parameter values and keywords to function arguments"""
        if params is None: params = {}
        if kwargs is None: kwargs = {}
        out = {}
        out.update(self.opts)
        for name, par in params.items():
            if strip:
                name = self._strip_prefix(name)
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = par.value

        # kwargs handled slightly differently -- may set param value too!
        for name, val in kwargs.items():
            if strip:
                name = self._strip_prefix(name)
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = val
                if name in params:
                    params[name].value = val
        return out

    def _make_all_args(self, params=None, **kwargs):
        """generate **all** function args for all functions"""
        args = {}
        for key, val in self.make_funcargs(params, kwargs).items():
            args["%s%s" % (self._prefix, key)] = val
        return args

    def eval(self, params=None, **kwargs):
        """evaluate the model with the supplied parameters"""
        return self.func(**self.make_funcargs(params, kwargs))

    @property
    def components(self):
        """return components for composite model"""
        return [self]

    def eval_components(self, params=None, **kwargs):
        """
        evaluate the model with the supplied parameters and returns a ordered
        dict containting name, result pairs.
        """
        key = self._prefix
        if len(key) < 1:
            key = self._name
        return {key: self.eval(params=params, **kwargs)}

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None, **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        data: array-like
        params: Parameters object
        weights: array-like of same size as data
            used for weighted fit
        method: fitting method to use (default = 'leastsq')
        iter_cb:  None or callable  callback function to call at each iteration.
        scale_covar:  bool (default True) whether to auto-scale covariance matrix
        verbose: bool (default True) print a message when a new parameter is
            added because of a hint.
        fit_kws: dict
            default fitting options, such as xtol and maxfev, for scipy optimizer
        keyword arguments: optional, named like the arguments of the
            model function, will override params. See examples below.

        Returns
        -------
        lmfit.ModelResult

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
            params = self.make_params(verbose=verbose)
        else:
            params = deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & set(self.param_names)
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = deepcopy(p)
            else:
                params[name].set(value=p)
            del kwargs[name]

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs.keys():
            if name not in self.independent_vars:
                warnings.warn("The keyword argument %s does not" % name +
                              "match any arguments of the model function." +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any([p not in params.keys()
                             for p in self.param_names])
        blank_param = any([(p.value is None and p.expr is None)
                           for p in params.values()])
        if missing_param or blank_param:
            msg = ('Assign each parameter an initial value by passing '
                   'Parameters or keyword arguments to fit.\n')
            missing = [p for p in self.param_names if p not in params.keys()]
            blank = [name for name, p in params.items()
                                    if (p.value is None and p.expr is None)]
            msg += 'Missing parameters: %s\n' % str(missing)
            msg += 'Non initialized parameters: %s' % str(blank)
            raise ValueError(msg)

        # Do not alter anything that implements the array interface (np.array, pd.Series)
        # but convert other iterables (e.g., Python lists) to numpy arrays.
        if not hasattr(data, '__array__'):
            data = np.asfarray(data)
        for var in self.independent_vars:
            var_data = kwargs[var]
            if (not hasattr(var_data, '__array__')) and (not np.isscalar(var_data)):
                kwargs[var] = np.asfarray(var_data)

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
            if not np.isscalar(kwargs[var]):
                kwargs[var] = _align(kwargs[var], mask, data)

        if fit_kws is None:
            fit_kws = {}

        output = ModelResult(self, params, method=method, iter_cb=iter_cb,
                             scale_covar=scale_covar, fcn_kws=kwargs,
                             **fit_kws)
        output.fit(data=data, weights=weights)
        output.components = self.components
        return output

    def __add__(self, other):
        return CompositeModel(self, other, operator.add)

    def __sub__(self, other):
        return CompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        return CompositeModel(self, other, operator.mul)

    def __div__(self, other):
        return CompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        return CompositeModel(self, other, operator.truediv)


class CompositeModel(Model):
    """Create a composite model -- a binary operator of two Models

    Parameters
    ----------
    left_model:    left-hand side model-- must be a Model()
    right_model:   right-hand side model -- must be a Model()
    oper:          callable binary operator (typically, operator.add, operator.mul, etc)

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
    name: None or string
        name for the model. When `None` (default) the name is the same as
        the model function (`func`).

    """
    _names_collide = ("\nTwo models have parameters named '{clash}'. "
                      "Use distinct names.")
    _bad_arg   = "CompositeModel: argument {arg} is not a Model"
    _bad_op    = "CompositeModel: operator {op} is not callable"
    _known_ops = {operator.add: '+', operator.sub: '-',
                  operator.mul: '*', operator.truediv: '/'}

    def __init__(self, left, right, op, **kws):
        if not isinstance(left, Model):
            raise ValueError(self._bad_arg.format(arg=left))
        if not isinstance(right, Model):
            raise ValueError(self._bad_arg.format(arg=right))
        if not callable(op):
            raise ValueError(self._bad_op.format(op=op))

        self.left  = left
        self.right = right
        self.op    = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += self._names_collide.format(clash=collision)
            raise NameError(msg)

        # we assume that all the sub-models have the same independent vars
        if 'independent_vars' not in kws:
            kws['independent_vars'] = self.left.independent_vars
        if 'missing' not in kws:
            kws['missing'] = self.left.missing

        def _tmp(self, *args, **kws): pass
        Model.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints["%s%s" % (prefix, basename)] = hint

    def _parse_params(self):
        self._func_haskeywords = (self.left._func_haskeywords or
                                  self.right._func_haskeywords)
        self._func_allargs = (self.left._func_allargs +
                              self.right._func_allargs)
        self.def_vals = deepcopy(self.right.def_vals)
        self.def_vals.update(self.left.def_vals)
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=False):
        return "(%s %s %s)" % (self.left._reprstring(long=long),
                               self._known_ops.get(self.op, self.op),
                               self.right._reprstring(long=long))

    def eval(self, params=None, **kwargs):
        return self.op(self.left.eval(params=params, **kwargs),
                       self.right.eval(params=params, **kwargs))

    def eval_components(self, **kwargs):
        """return ordered dict of name, results for each component"""
        out = OrderedDict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    @property
    def param_names(self):
        return  self.left.param_names + self.right.param_names

    @property
    def components(self):
        """return components for composite model"""
        return self.left.components + self.right.components

    def _make_all_args(self, params=None, **kwargs):
        """generate **all** function args for all functions"""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out


class ModelResult(Minimizer):
    """Result from Model fit

    Attributes
    -----------
    model         instance of Model -- the model function
    params        instance of Parameters -- the fit parameters
    data          array of data values to compare to model
    weights       array of weights used in fitting
    init_params   copy of params, before being updated by fit()
    init_values   array of parameter values, before being updated by fit()
    init_fit      model evaluated with init_params.
    best_fit      model evaluated with params after being updated by fit()

    Methods:
    --------
    fit(data=None, params=None, weights=None, method=None, **kwargs)
         fit (or re-fit) model with params to data (with weights)
         using supplied method.  The keyword arguments are sent to
         as keyword arguments to the model function.

         all inputs are optional, defaulting to the value used in
         the previous fit.  This allows easily changing data or
         parameter settings, or both.

    eval(params=None, **kwargs)
         evaluate the current model, with parameters (defaults to the current
         parameter values), with values in kwargs sent to the model function.

    eval_components(params=Nones, **kwargs)
         evaluate the current model, with parameters (defaults to the current
         parameter values), with values in kwargs sent to the model function
         and returns an ordered dict with the model names as the key and the
         component results as the values.

   fit_report(modelpars=None, show_correl=True, min_correl=0.1)
         return a fit report.

   plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--', xlabel = None, ylabel=None,
            numpoints=None,  data_kws=None, fit_kws=None, init_kws=None,
            ax_kws=None)
        Plot the fit results using matplotlib.

   plot_residuals(self, ax=None, datafmt='o', data_kws=None, fit_kws=None,
                  ax_kws=None)
        Plot the fit residuals using matplotlib.

   plot(self, datafmt='o', fitfmt='-', initfmt='--', xlabel=None, ylabel=None, numpoints=None,
        data_kws=None, fit_kws=None, init_kws=None, ax_res_kws=None,
        ax_fit_kws=None, fig_kws=None)
        Plot the fit results and residuals using matplotlib.
    """
    def __init__(self, model, params, data=None, weights=None,
                 method='leastsq', fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **fit_kws):
        self.model = model
        self.data = data
        self.weights = weights
        self.method = method
        self.ci_out = None
        self.init_params = deepcopy(params)
        Minimizer.__init__(self, model._residual, params, fcn_args=fcn_args,
                           fcn_kws=fcn_kws, iter_cb=iter_cb,
                           scale_covar=scale_covar, **fit_kws)

    def fit(self, data=None, params=None, weights=None, method=None, **kwargs):
        """perform fit for a Model, given data and params"""
        if data is not None:
            self.data = data
        if params is not None:
            self.init_params = params
        if weights is not None:
            self.weights = weights
        if method is not None:
            self.method = method
        self.ci_out = None
        self.userargs = (self.data, self.weights)
        self.userkws.update(kwargs)
        self.init_fit    = self.model.eval(params=self.params, **self.userkws)

        _ret = self.minimize(method=self.method)

        for attr in dir(_ret):
            if not attr.startswith('_') :
                try:
                    setattr(self, attr, getattr(_ret, attr))
                except AttributeError:
                    pass

        self.init_values = self.model._make_all_args(self.init_params)
        self.best_values = self.model._make_all_args(_ret.params)
        self.best_fit    = self.model.eval(params=_ret.params, **self.userkws)

    def eval(self, params=None, **kwargs):
        """
        evaluate model function
        Arguments:
            params (Parameters):  parameters, defaults to ModelResult .params
            kwargs (variable):  values of options, independent variables, etc

        Returns:
            ndarray or float for evaluated model
        """
        self.userkws.update(kwargs)
        if params is None:
           params = self.params
        return self.model.eval(params=params, **self.userkws)

    def eval_components(self, params=None, **kwargs):
        """
        evaluate each component of a composite model function
        Arguments:
            params (Parameters):  parameters, defaults to ModelResult .params
            kwargs (variable):  values of options, independent variables, etc

        Returns:
            ordered dictionary with keys of prefixes, and values of values for
            each component of the model.
        """
        self.userkws.update(kwargs)
        if params is None:
            params = self.params
        return self.model.eval_components(params=params, **self.userkws)

    def conf_interval(self, **kwargs):
        """return explicitly calculated confidence intervals"""
        if self.ci_out is None:
            self.ci_out = conf_interval(self, self, **kwargs)
        return self.ci_out

    def ci_report(self, with_offset=True, ndigits=5, **kwargs):
        """return nicely formatted report about confidence intervals"""
        return ci_report(self.conf_interval(**kwargs),
                         with_offset=with_offset, ndigits=ndigits)

    def fit_report(self,  **kwargs):
        "return fit report"
        return '[[Model]]\n    %s\n%s\n' % (self.model._reprstring(long=True),
                                            fit_report(self, **kwargs))

    @_ensureMatplotlib
    def plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--', xlabel=None, ylabel=None, yerr=None,
                 numpoints=None,  data_kws=None, fit_kws=None, init_kws=None,
                 ax_kws=None):
        """Plot the fit results using matplotlib.

        The method will plot results of the fit using matplotlib, including:
        the data points, the initial fit curve and the fitted curve. If the fit
        model included weights, errorbars will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : string, optional
            matplotlib format string for data points
        fitfmt : string, optional
            matplotlib format string for fitted curve
        initfmt : string, optional
            matplotlib format string for initial conditions for the fit
        xlabel : string, optional
            matplotlib format string for labeling the x-axis
        ylabel : string, optional
            matplotlib format string for labeling the y-axis
        yerr : ndarray, optional
            array of uncertainties for data array
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        init_kws : dictionary, optional
            keyword arguments passed on to the plot function for the initial
            conditions of the fit
        ax_kws : dictionary, optional
            keyword arguments for a new axis, if there is one being created

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        ----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `ax` is None then matplotlib.pyplot.gca(**ax_kws) is called.

        See Also
        --------
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_kws is None:
            ax_kws = {}

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        x_array = self.userkws[independent_var]

        # make a dense array for x-axis if data is not dense
        if numpoints is not None and len(self.data) < numpoints:
            x_array_dense = np.linspace(min(x_array), max(x_array), numpoints)
        else:
            x_array_dense = x_array

        ax.plot(x_array_dense, self.model.eval(self.init_params,
                **{independent_var: x_array_dense}), initfmt,
                label='init', **init_kws)
        ax.plot(x_array_dense, self.model.eval(self.params,
                **{independent_var: x_array_dense}), fitfmt,
                label='best-fit', **fit_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights
        if yerr is not None:
            ax.errorbar(x_array, self.data, yerr=yerr,
                        fmt=datafmt, label='data', **data_kws)
        else:
            ax.plot(x_array, self.data, datafmt, label='data', **data_kws)

        ax.set_title(self.model.name)
        if xlabel is None:
            ax.set_xlabel(independent_var)
        else: ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel('y')
        else: ax.set_ylabel(ylabel)
        ax.legend()

        return ax

    @_ensureMatplotlib
    def plot_residuals(self, ax=None, datafmt='o', yerr=None, data_kws=None,
                       fit_kws=None, ax_kws=None):
        """Plot the fit residuals using matplotlib.

        The method will plot residuals of the fit using matplotlib, including:
        the data points and the fitted curve (as horizontal line). If the fit
        model included weights, errorbars will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : string, optional
            matplotlib format string for data points
        yerr : ndarray, optional
            array of uncertainties for data array
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        ax_kws : dictionary, optional
            keyword arguments for a new axis, if there is one being created

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        ----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `ax` is None then matplotlib.pyplot.gca(**ax_kws) is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if ax_kws is None:
            ax_kws = {}

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        x_array = self.userkws[independent_var]

        ax.axhline(0, **fit_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights
        if yerr is not None:
            ax.errorbar(x_array, self.eval() - self.data, yerr=yerr,
                        fmt=datafmt, label='residuals', **data_kws)
        else:
            ax.plot(x_array, self.eval() - self.data, datafmt,
                    label='residuals', **data_kws)

        ax.set_title(self.model.name)
        ax.set_ylabel('residuals')
        ax.legend()

        return ax

    @_ensureMatplotlib
    def plot(self, datafmt='o', fitfmt='-', initfmt='--', xlabel=None, ylabel=None, yerr=None,
             numpoints=None, fig=None, data_kws=None, fit_kws=None,
             init_kws=None, ax_res_kws=None, ax_fit_kws=None,
             fig_kws=None):
        """Plot the fit results and residuals using matplotlib.

        The method will produce a matplotlib figure with both results of the
        fit and the residuals plotted. If the fit model included weights,
        errorbars will also be plotted.

        Parameters
        ----------
        datafmt : string, optional
            matplotlib format string for data points
        fitfmt : string, optional
            matplotlib format string for fitted curve
        initfmt : string, optional
            matplotlib format string for initial conditions for the fit
        xlabel : string, optional
            matplotlib format string for labeling the x-axis
        ylabel : string, optional
            matplotlib format string for labeling the y-axis
        yerr : ndarray, optional
            array of uncertainties for data array
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default in None, which means use the
            current pyplot figure or create one if there is none.
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        init_kws : dictionary, optional
            keyword arguments passed on to the plot function for the initial
            conditions of the fit
        ax_res_kws : dictionary, optional
            keyword arguments for the axes for the residuals plot
        ax_fit_kws : dictionary, optional
            keyword arguments for the axes for the fit plot
        fig_kws : dictionary, optional
            keyword arguments for a new figure, if there is one being created

        Returns
        -------
        matplotlib.figure.Figure

        Notes
        ----
        The method combines ModelResult.plot_fit and ModelResult.plot_residuals.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `fig` is None then matplotlib.pyplot.figure(**fig_kws) is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_res_kws is None:
            ax_res_kws = {}
        if ax_fit_kws is None:
            ax_fit_kws = {}
        if fig_kws is None:
            fig_kws = {}

        if len(self.model.independent_vars) != 1:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(fig, plt.Figure):
            fig = plt.figure(**fig_kws)

        gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 4])
        ax_res = fig.add_subplot(gs[0], **ax_res_kws)
        ax_fit = fig.add_subplot(gs[1], sharex=ax_res, **ax_fit_kws)

        self.plot_fit(ax=ax_fit, datafmt=datafmt, fitfmt=fitfmt, yerr=yerr,
                      initfmt=initfmt, xlabel=xlabel, ylabel=ylabel, numpoints=numpoints, data_kws=data_kws,
                      fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws)
        self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr,
                            data_kws=data_kws, fit_kws=fit_kws,
                            ax_kws=ax_res_kws)

        return fig
