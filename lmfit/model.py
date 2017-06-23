"""Concise nonlinear curve fitting."""
from __future__ import print_function

from collections import OrderedDict
from copy import deepcopy
from functools import wraps
import inspect
import operator
import warnings

import numpy as np
from scipy.special import erf
from scipy.stats import t

from . import Minimizer, Parameter, Parameters
from .confidence import conf_interval
from .printfuncs import ci_report, fit_report

# Use pandas.isnull for aligning missing data if pandas is available.
# otherwise use numpy.isnan
try:
    from pandas import isnull, Series
except ImportError:
    isnull = np.isnan
    Series = type(NotImplemented)


def _align(var, mask, data):
    """Align missing data, if pandas is available."""
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
        @wraps(function)
        def wrapper(*args, **kws):
            return function(*args, **kws)
        return wrapper
    else:
        def no_op(*args, **kwargs):
            print('matplotlib module is required for plotting the results')

        return no_op


class Model(object):
    """Create a model from a user-supplied model function.

    The model function will normally take an independent variable
    (generally, the first argument) and a series of arguments that are
    meant to be parameters for the model. It will return an array of
    data to model some data as for a curve-fitting problem.

    """

    _forbidden_args = ('data', 'weights', 'params')
    _invalid_ivar = "Invalid independent variable name ('%s') for function %s"
    _invalid_par = "Invalid parameter name ('%s') for function %s"
    _invalid_missing = "missing must be None, 'none', 'drop', or 'raise'."
    _valid_missing = (None, 'none', 'drop', 'raise')

    _invalid_hint = "unknown parameter hint '%s' for param '%s'"
    _hint_names = ('value', 'vary', 'min', 'max', 'expr')

    def __init__(self, func, independent_vars=None, param_names=None,
                 missing='none', prefix='', name=None, **kws):
        """
        Parameters
        ----------
        func : callable
            Function to be wrapped.
        independent_vars : list of str, optional
            Arguments to func that are independent variables (default is None).
        param_names : list of str, optional
            Names of arguments to func that are to be made into parameters
            (default is None).
        missing : str, optional
            How to handle NaN and missing values in data. One of:

            - 'none' or None : Do not check for null or missing values (default).

            - 'drop' : Drop null or missing observations in data. If pandas is
              installed, `pandas.isnull` is used, otherwise `numpy.isnan` is used.
            - 'raise' : Raise a (more helpful) exception when data contains
              null or missing values.

        prefix : str, optional
            Prefix used for the model.
        name : str, optional
            Name for the model. When None (default) the name is the same as
            the model function (`func`).
        **kws : dict, optional
            Additional keyword arguments to pass to model function.

        Notes
        -----
        1. Parameter names are inferred from the function arguments,
        and a residual function is automatically constructed.

        2. The model function must return an array that will be the same
        size as the data being modeled.

        Examples
        --------
        The model function will normally take an independent variable (generally,
        the first argument) and a series of arguments that are meant to be
        parameters for the model.  Thus, a simple peak using a Gaussian
        defined as:

        >>> import numpy as np
        >>> def gaussian(x, amp, cen, wid):
        ...     return amp * np.exp(-(x-cen)**2 / wid)

        can be turned into a Model with:

        >>> gmodel = Model(gaussian)

        this will automatically discover the names of the independent variables
        and parameters:

        >>> print(gmodel.param_names, gmodel.independent_vars)
        ['amp', 'cen', 'wid'] ['x']

        """
        self.func = func
        self._prefix = prefix
        self._param_root_names = param_names  # will not include prefixes
        self.independent_vars = independent_vars
        self._func_allargs = []
        self._func_haskeywords = False
        if missing not in self._valid_missing:
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
        """Return Model name."""
        return self._reprstring(long=False)

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def prefix(self):
        """Return Model prefix."""
        return self._prefix

    @property
    def param_names(self):
        """Return the parameters of the Model."""
        return self._param_names

    def __repr__(self):
        """ Return representation of Model."""
        return "<lmfit.Model: %s>" % (self.name)

    def copy(self, **kwargs):
        """DOES NOT WORK."""
        raise NotImplementedError("Model.copy does not work. Make a new Model")

    def _parse_params(self):
        """Build parameters from function arguments."""
        if self.func is None:
            return
        if hasattr(self.func, 'argnames') and hasattr(self.func, 'kwargs'):
            pos_args = self.func.argnames[:]
            kw_args = {}
            for name, defval in self.func.kwargs:
                kw_args[name] = defval
            keywords_ = list(kw_args.keys())
        else:
            try:  # PY3
                argspec = inspect.getfullargspec(self.func)
                keywords_ = argspec.varkw
            except AttributeError:  # PY2
                argspec = inspect.getargspec(self.func)
                keywords_ = argspec.keywords

            pos_args = argspec.args
            kw_args = {}
            if argspec.defaults is not None:
                for val in reversed(argspec.defaults):
                    kw_args[pos_args.pop()] = val

        self._func_haskeywords = keywords_ is not None
        self._func_allargs = pos_args + list(kw_args.keys())
        allargs = self._func_allargs

        if len(allargs) == 0 and keywords_ is not None:
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
        """Set *hints* to use when creating parameters with `make_params()` for
        the named parameter.

        This is especially convenient for setting initial values. The `name`
        can include the models `prefix` or not. The hint given can also
        include optional bounds and constraints ``(value, vary, min, max, expr)``,
        which will be used by make_params() when building default parameters.

        Parameters
        ----------
        name : string
            Parameter name.

        **kwargs : optional
            Arbitrary keyword arguments, needs to be a Parameter attribute.
            Can be any of the following:

            - value : float, optional
                Numerical Parameter value.
            - vary : bool, optional
                Whether the Parameter is varied during a fit (default is True).
            - min : float, optional
                Lower bound for value (default is `-numpy.inf`, no lower bound).
            - max : float, optional
                Upper bound for value (default is `numpy.inf`, no upper bound).
            - expr : str, optional
                Mathematical expression used to constrain the value during the fit.


        Example
        --------

        >>> from lmfit.models import GaussianModel
        >>> model = GaussianModel()
        >>> model.set_param_hint('sigma', min=0)

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
        """Print a nicely aligned text-table of parameter hints.

        Parameters
        ----------
        colwidth : int, optional
           Width of each column, except for first and last columns.

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
        """Create a Parameters object for a Model.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print out messages (default is False).
        **kwargs : optional
            Parameter names and initial values.


        Returns
        ---------
        params : Parameters

        Notes
        -----
        1. The parameters may or may not have decent initial values for each
        parameter.

        2. This applies any default values or parameter hints that may have
        been set.

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
                print(' - Adding parameter "%s"' % name)

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
                    print(' - Adding parameter for hint "%s"' % name)
            par._delay_asteval = True
            for item in self._hint_names:
                if item in hint:
                    setattr(par, item, hint[item])
            if basename in kwargs:
                par.value = kwargs[basename]
            # Add the new parameter to self._param_names
            if name not in self._param_names:
                self._param_names.append(name)

        for p in params.values():
            p._delay_asteval = False
        return params

    def guess(self, data, **kws):
        """Guess starting values for the parameters of a model.

        This is not implemented for all models, but is available for many of
        the built-in models.

        Parameters
        ----------
        data : array_like
            Array of data to use to guess parameter values.
        **kws : optional
            Additional keyword arguments, passed to model function.

        Returns
        -------
        params : Parameters

        Notes
        -----
        Should be implemented for each model subclass to run
        self.make_params(), update starting values and return a
        Parameters object.

        Raises
        ------
        NotImplementedError

        """
        cname = self.__class__.__name__
        msg = 'guess() not implemented for %s' % cname
        raise NotImplementedError(msg)

    def _residual(self, params, data, weights, **kwargs):
        """Return the residual.

        Default residual: (data-model)*weights.

        If the model returns complex values, the residual is computed by
        treating the real and imaginary parts separately. In this case,
        if the weights provided are real, they are assumed to apply
        equally to the real and imaginary parts. If the weights are
        complex, the real part of the weights are applied to the real
        part of the residual and the imaginary part is treated
        correspondingly.

        Since the underlying scipy.optimize routines expect numpy.float
        arrays, the only complex type supported is np.complex.

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
        """Handle missing data."""
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
        """Convert parameter values and keywords to function arguments."""
        if params is None:
            params = {}
        if kwargs is None:
            kwargs = {}
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
        """Generate **all** function args for all functions."""
        args = {}
        for key, val in self.make_funcargs(params, kwargs).items():
            args["%s%s" % (self._prefix, key)] = val
        return args

    def eval(self, params=None, **kwargs):
        """Evaluate the model with supplied parameters and keyword arguments.

        Parameters
        -----------
        params : Parameters, optional
            Parameters to use in Model.
        **kwargs : optional
            Additional keyword arguments to pass to model function.

        Returns
        -------
        numpy.ndarray
            Value of model given the parameters and other arguments.

        Notes
        -----
        1. if `params` is None, the values for all parameters are
        expected to be provided as keyword arguments.  If `params` is
        given, and a keyword argument for a parameter value is also given,
        the keyword argument will be used.

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        """
        return self.func(**self.make_funcargs(params, kwargs))

    @property
    def components(self):
        """Return components for composite model."""
        return [self]

    def eval_components(self, params=None, **kwargs):
        """Evaluate the model with the supplied parameters.

        Parameters
        -----------
        params : Parameters, optional
            Parameters to use in Model.
        **kwargs : optional
            Additional keyword arguments to pass to model function.

        Returns
        -------
        OrderedDict
            Keys are prefixes for component model, values are value of each
            component.

        """
        key = self._prefix
        if len(key) < 1:
            key = self._name
        return {key: self.eval(params=params, **kwargs)}

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            **kwargs):
        """Fit the model to the data using the supplied Parameters.

        Parameters
        ----------
        data : array_like
            Array of data to be fit.
        params : Parameters, optional
            Parameters to use in fit (default is None).
        weights : array_like of same size as `data`, optional
            Weights to use for the calculation of the fit residual (default
            is None).
        method : str, optional
            Name of fitting method to use (default is `'leastsq'`).
        iter_cb : callable, optional
             Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
             Whether to automatically scale the covariance matrix when
             calculating uncertainties (default is True, `leastsq` method only).
        verbose: bool, optional
             Whether to print a message when a new parameter is added because
             of a hint (default is True).
        fit_kws: dict, optional
             Options to pass to the minimizer being used.
        **kwargs: optional
             Arguments to pass to the  model function, possibly overriding
             params.

        Returns
        -------
        ModelResult

        Examples
        --------
        Take `t` to be the independent variable and data to be the curve we
        will fit. Use keyword arguments to set initial guesses:

        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        Or, for more control, pass a Parameters object.

        >>> result = my_model.fit(data, params, t=t)

        Keyword arguments override Parameters.

        >>> result = my_model.fit(data, params, tau=5, t=t)

        Notes
        -----
        1. if `params` is None, the values for all parameters are
        expected to be provided as keyword arguments.  If `params` is
        given, and a keyword argument for a parameter value is also given,
        the keyword argument will be used.

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        3. Parameters (however passed in), are copied on input, so the
        original Parameter objects are unchanged, and the updated values
        are in the returned `ModelResult`.

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
        for name in kwargs:
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
                     if p.value is None and p.expr is None]
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
        """+"""
        return CompositeModel(self, other, operator.add)

    def __sub__(self, other):
        """-"""
        return CompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        """*"""
        return CompositeModel(self, other, operator.mul)

    def __div__(self, other):
        """/"""
        return CompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        """/"""
        return CompositeModel(self, other, operator.truediv)


class CompositeModel(Model):
    """Combine two models (`left` and `right`) with a binary operator (`op`)
    into a CompositeModel.

    Normally, one does not have to explicitly create a `CompositeModel`,
    but can use normal Python operators `+`, '-', `*`, and `/` to combine
    components as in::

    >>> fcn1 = lambda x, a: None
    >>> fcn2 = lambda x, b: None
    >>> fcn3 = lambda x, c: None

    >>> mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

    """

    _names_collide = ("\nTwo models have parameters named '{clash}'. "
                      "Use distinct names.")
    _bad_arg = "CompositeModel: argument {arg} is not a Model"
    _bad_op = "CompositeModel: operator {op} is not callable"
    _known_ops = {operator.add: '+', operator.sub: '-',
                  operator.mul: '*', operator.truediv: '/'}

    def __init__(self, left, right, op, **kws):
        """
        Parameters
        ----------
        left : Model
            Left-hand model.
        right : Model
            Right-hand model.
        op : callable binary operator
            Operator to combine `left` and `right` models.
        **kws : optional
            Additional keywords are passed to `Model` when creating this
            new model.

        Notes
        -----
        1. The two models must use the same independent variable.

        """
        if not isinstance(left, Model):
            raise ValueError(self._bad_arg.format(arg=left))
        if not isinstance(right, Model):
            raise ValueError(self._bad_arg.format(arg=right))
        if not callable(op):
            raise ValueError(self._bad_op.format(op=op))

        self.left = left
        self.right = right
        self.op = op

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

        def _tmp(self, *args, **kws):
            pass
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
        """TODO: docstring in public method."""
        return self.op(self.left.eval(params=params, **kwargs),
                       self.right.eval(params=params, **kwargs))

    def eval_components(self, **kwargs):
        """Return OrderedDict of name, results for each component."""
        out = OrderedDict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    @property
    def param_names(self):
        """Return parameter names for composite model."""
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """Return components for composite model."""
        return self.left.components + self.right.components

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out


class ModelResult(Minimizer):
    """Result from the Model fit.

    This has many attributes and methods for viewing and working with
    the results of a fit using Model. It inherits from Minimizer, so
    that it can be used to modify and re-run the fit for the Model.

    """

    def __init__(self, model, params, data=None, weights=None,
                 method='leastsq', fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **fit_kws):
        """
        Parameters
        ----------
        model : Model
            Model to use.
        params : Parameters
            Parameters with initial values for model.
        data : array_like, optional
            Data to be modeled.
        weights : array_like, optional
            Weights to multiply (data-model) for fit residual.
        method : str, optional
            Name of minimization method to use (default is `'leastsq'`).
        fcn_args : sequence, optional
            Positional arguments to send to model function.
        fcn_dict : dict, optional
            Keyword arguments to send to model function.
        iter_cb : callable, optional
            Function to call on each iteration of fit.
        scale_covar : bool, optional
            Whether to scale covariance matrix for uncertainty evaluation.
        **fit_kws : optional
            Keyword arguments to send to minimization routine.
        """
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
        """Re-perform fit for a Model, given data and params.

        Parameters
        ----------
        data : array_like, optional
            Data to be modeled.
        params : Parameters, optional
            Parameters with initial values for model.
        weights : array_like, optional
            Weights to multiply (data-model) for fit residual.
        method : str, optional
            Name of minimization method to use (default is `'leastsq'`).
        **kwargs : optional
            Keyword arguments to send to minimization routine.

        """
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
        self.init_fit = self.model.eval(params=self.params, **self.userkws)

        _ret = self.minimize(method=self.method)

        for attr in dir(_ret):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(_ret, attr))
                except AttributeError:
                    pass

        self.init_values = self.model._make_all_args(self.init_params)
        self.best_values = self.model._make_all_args(_ret.params)
        self.best_fit = self.model.eval(params=_ret.params, **self.userkws)

    def eval(self, params=None, **kwargs):
        """Evaluate model function.

        Parameters
        ----------
        params : Parameters, optional
            Parameters to use.
        **kwargs : optional
            Options to send to Model.eval()

        Returns
        -------
        out : numpy.ndarray
           Array for evaluated model.

        """
        self.userkws.update(kwargs)
        if params is None:
            params = self.params
        return self.model.eval(params=params, **self.userkws)

    def eval_components(self, params=None, **kwargs):
        """Evaluate each component of a composite model function.

        Parameters
        ----------
        params : Parameters, optional
            Parameters, defaults to ModelResult.params
        **kwargs : optional
             Leyword arguments to pass to model function.

        Returns
        -------
        OrderedDict
             Keys are prefixes of component models, and values are
             the estimated model value for each component of the model.

        """
        self.userkws.update(kwargs)
        if params is None:
            params = self.params
        return self.model.eval_components(params=params, **self.userkws)

    def eval_uncertainty(self, params=None, sigma=1, **kwargs):
        """Evaluate the uncertainty of the *model function* from the
        uncertainties for the best-fit parameters.  This can be used to give
        confidence bands for the model.

        Parameters
        ----------
        params : Parameters, optional
             Parameters, defaults to ModelResult.params.
        sigma : float, optional
             Confidence level, i.e. how many sigma (default  is 1).
        **kwargs : optional
             Values of options, independent variables, etcetera.

        Returns
        -------
        numpy.ndarray
           Uncertainty at each value of the model.

        Example
        -------

        >>> out = model.fit(data, params, x=x)
        >>> dely = out.eval_confidence_band(x=x)
        >>> plt.plot(x, data)
        >>> plt.plot(x, out.best_fit)
        >>> plt.fill_between(x, out.best_fit-dely,
        ...                 out.best_fit+dely, color='#888888')

        Notes
        -----
        1. This is based on the excellent and clear example from
           https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals,
           which references the original work of:
           J. Wolberg,Data Analysis Using the Method of Least Squares, 2006, Springer
        2. The value of sigma is number of `sigma` values, and is converted to a
           probability.  Values or 1, 2, or 3 give probalities of 0.6827, 0.9545,
           and 0.9973, respectively. If the sigma value is < 1, it is interpreted
           as the probability itself.  That is, `sigma=1` and `sigma=0.6827` will
           give the same results, within precision errors.

        """
        self.userkws.update(kwargs)
        if params is None:
            params = self.params

        nvarys = self.nvarys
        ndata = self.ndata
        covar = self.covar / self.redchi
        fjac = np.zeros(ndata * nvarys).reshape((nvarys, ndata))
        df2 = np.zeros(ndata)

        # find derivative by hand!
        for i in range(nvarys):
            pname = self.var_names[i]
            pars = self.params
            val0 = pars[pname].value
            dval = pars[pname].stderr / 3.0

            pars[pname].value = val0 + dval
            res1 = self.model.eval(pars, **self.userkws)

            pars[pname].value = val0 - dval
            res2 = self.model.eval(pars, **self.userkws)

            pars[pname].value = val0
            fjac[i] = (res1 - res2) / (2 * dval)

        for i in range(nvarys):
            for j in range(nvarys):
                df2 += fjac[i] * fjac[j] * covar[i, j]

        if sigma < 1.0:
            prob = sigma
        else:
            prob = erf(sigma / np.sqrt(2))
        return np.sqrt(df2 * self.redchi) * t.ppf((prob + 1) / 2.0, ndata - nvarys)

    def conf_interval(self, **kwargs):
        """Calculate the confidence intervals for the variable parameters.

        Confidence intervals are calculated using the
        :func:`confidence.conf_interval()` function and keyword
        arguments (`**kwargs`) are passed to that function. The result
        is stored in the :attr:`ci_out` attribute so that it can be
        accessed without recalculating them.

        """
        if self.ci_out is None:
            self.ci_out = conf_interval(self, self, **kwargs)
        return self.ci_out

    def ci_report(self, with_offset=True, ndigits=5, **kwargs):
        """Return a nicely formatted text report of the confidence intervals.

        Parameters
        ----------
        with_offset : bool, optional
             Whether to subtract best value from all other values (default is True).
        ndigits : int, optional
            Number of significant digits to show (default is 5).
        **kwargs: optional
            Keyword arguments that are passed to the `conf_interval` function.

        Returns
        -------
        str
            Text of formatted report on confidence intervals.

        """
        return ci_report(self.conf_interval(**kwargs),
                         with_offset=with_offset, ndigits=ndigits)

    def fit_report(self, modelpars=None, show_correl=True,
                   min_correl=0.1, sort_pars=False):
        """Return a printable fit report.

        The report contains fit statistics and best-fit values with
        uncertainties and correlations.


        Parameters
        ----------
        modelpars : Parameters, optional
           Known Model Parameters.
        show_correl : bool, optional
           Whether to show list of sorted correlations (default is True).
        min_correl : float, optional
           Smallest correlation in absolute value to show (default is 0.1).
        sort_pars : callable, optional
           Whether to show parameter names sorted in alphanumerical order
           (default is False). If False, then the parameters will be listed in
           the order as they were added to the Parameters dictionary. If callable,
           then this (one argument) function is used to extract a comparison key
           from each list element.

        Returns
        -------
        text : str
           Multi-line text of fit report.

        See Also
        --------
        :func:`fit_report()`

        """
        report = fit_report(self, modelpars=modelpars,
                            show_correl=show_correl,
                            min_correl=min_correl, sort_pars=sort_pars)
        modname = self.model._reprstring(long=True)
        return '[[Model]]\n    %s\n%s\n' % (modname, report)

    @_ensureMatplotlib
    def plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--',
                 xlabel=None, ylabel=None, yerr=None, numpoints=None,
                 data_kws=None, fit_kws=None, init_kws=None, ax_kws=None):
        """Plot the fit results using matplotlib, if available.

        The plot will include the data points, the initial fit curve, and
        the best-fit curve. If the fit  model included weights or if `yerr`
        is specified, errorbars will also be plotted.


        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : str, optional
            Matplotlib format string for data points.
        fitfmt : str, optional
            Matplotlib format string for fitted curve.
        initfmt : str, optional
            Matplotlib format string for initial conditions for the fit.
        xlabel : str, optional
            Matplotlib format string for labeling the x-axis.
        ylabel : str, optional
            Matplotlib format string for labeling the y-axis.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed on to the plot function for the initial
            conditions of the fit.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if there is one being created.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights

        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.

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

        ax.plot(
            x_array_dense,
            self.model.eval(self.init_params,
                            **{independent_var: x_array_dense}),
            initfmt, label='init', **init_kws)
        ax.plot(
            x_array_dense,
            self.model.eval(self.params, **{independent_var: x_array_dense}),
            fitfmt, label='best-fit', **fit_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0 / self.weights
        if yerr is not None:
            ax.errorbar(x_array, self.data, yerr=yerr,
                        fmt=datafmt, label='data', **data_kws)
        else:
            ax.plot(x_array, self.data, datafmt, label='data', **data_kws)

        ax.set_title(self.model.name)
        if xlabel is None:
            ax.set_xlabel(independent_var)
        else:
            ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel('y')
        else:
            ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        return ax

    @_ensureMatplotlib
    def plot_residuals(self, ax=None, datafmt='o', yerr=None, data_kws=None,
                       fit_kws=None, ax_kws=None):
        """Plot the fit residuals using matplotlib, if available.

        If `yerr` is supplied or if the model included weights, errorbars
        will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : str, optional
            Matplotlib format string for data points.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if there is one being created.

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights

        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.

        """
        if data_kws is None:
            data_kws = {}
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
            yerr = 1.0 / self.weights
        if yerr is not None:
            ax.errorbar(x_array, self.eval() - self.data, yerr=yerr,
                        fmt=datafmt, label='residuals', **data_kws)
        else:
            ax.plot(x_array, self.eval() - self.data, datafmt,
                    label='residuals', **data_kws)

        ax.set_title(self.model.name)
        ax.set_ylabel('residuals')
        ax.legend(loc='best')
        return ax

    @_ensureMatplotlib
    def plot(self, datafmt='o', fitfmt='-', initfmt='--', xlabel=None,
             ylabel=None, yerr=None, numpoints=None, fig=None, data_kws=None,
             fit_kws=None, init_kws=None, ax_res_kws=None, ax_fit_kws=None,
             fig_kws=None):
        """Plot the fit results and residuals using matplotlib, if available.

        The method will produce a matplotlib figure with both results of the
        fit and the residuals plotted. If the fit model included weights,
        errorbars will also be plotted.

        Parameters
        ----------
        datafmt : str, optional
            Matplotlib format string for data points.
        fitfmt : str, optional
            Matplotlib format string for fitted curve.
        initfmt : str, optional
            Matplotlib format string for initial conditions for the fit.
        xlabel : str, optional
            Matplotlib format string for labeling the x-axis.
        ylabel : str, optional
            Matplotlib format string for labeling the y-axis.
        yerr : numpy.ndarray, optional
            Array of uncertainties for data array.
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default is None, which means use the
            current pyplot figure or create one if there is none.
        data_kws : dict, optional
            Keyword arguments passed on to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed on to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed on to the plot function for the initial
            conditions of the fit.
        ax_res_kws : dict, optional
            Keyword arguments for the axes for the residuals plot.
        ax_fit_kws : dict, optional
            Keyword arguments for the axes for the fit plot.
        fig_kws : dict, optional
            Keyword arguments for a new figure, if there is one being created.

        Returns
        -------
        A tuple with matplotlib's Figure and GridSpec objects.

        Notes
        -----
        The method combines ModelResult.plot_fit and ModelResult.plot_residuals.

        If `yerr` is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If `yerr` is
        not specified and the fit includes weights, `yerr` set to 1/self.weights

        If `fig` is None then `matplotlib.pyplot.figure(**fig_kws)` is called,
        otherwise `fig_kws` is ignored.

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
        # make a square figure with side equal to the default figure's x-size
        figxsize = plt.rcParams['figure.figsize'][0]
        fig_kws_ = dict(figsize=(figxsize, figxsize))
        if fig_kws is not None:
            fig_kws_.update(fig_kws)

        if len(self.model.independent_vars) != 1:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(fig, plt.Figure):
            fig = plt.figure(**fig_kws_)

        gs = plt.GridSpec(nrows=2, ncols=1, height_ratios=[1, 4])
        ax_res = fig.add_subplot(gs[0], **ax_res_kws)
        ax_fit = fig.add_subplot(gs[1], sharex=ax_res, **ax_fit_kws)

        self.plot_fit(ax=ax_fit, datafmt=datafmt, fitfmt=fitfmt, yerr=yerr,
                      initfmt=initfmt, xlabel=xlabel, ylabel=ylabel,
                      numpoints=numpoints, data_kws=data_kws,
                      fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws)
        self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr,
                            data_kws=data_kws, fit_kws=fit_kws,
                            ax_kws=ax_res_kws)
        plt.setp(ax_res.get_xticklabels(), visible=False)
        ax_fit.set_title('')
        return fig, gs
