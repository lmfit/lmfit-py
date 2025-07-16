"""Implementation of the Model interface."""

from copy import deepcopy
from functools import wraps
import inspect
import json
import operator
import warnings

from asteval import valid_symbol_name
import numpy as np
from scipy.special import erf
from scipy.stats import t

import lmfit

from . import Minimizer, Parameter, Parameters, lineshapes
from .confidence import conf_interval
from .jsonutils import decode4js, encode4js
from .minimizer import MinimizerResult
from .printfuncs import ci_report, fit_report, fitreport_html_table

tiny = 1.e-15

# Use pandas.isnull for aligning missing data if pandas is available.
# otherwise use numpy.isnan
try:
    from pandas import Series, isnull
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
    import matplotlib  # noqa: F401
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


def _ensureMatplotlib(function):
    if _HAS_MATPLOTLIB:
        @wraps(function)
        def wrapper(*args, **kws):
            return function(*args, **kws)
        return wrapper

    def no_op(*args, **kwargs):
        print('matplotlib module is required for plotting the results')
    return no_op


def get_reducer(option):
    """Factory function to build a parser for complex numbers.

    Parameters
    ----------
    option : {'real', 'imag', 'abs', 'angle'}
        Implements the NumPy function with the same name.

    Returns
    -------
    callable
        See docstring for `reducer` below.

    """
    if option not in ['real', 'imag', 'abs', 'angle']:
        raise ValueError(f"Invalid option ('{option}') for function 'propagate_err'.")

    def reducer(array):
        """Convert a complex array to a real array.

        Several conversion methods are available and it does nothing to a
        purely real array.

        Parameters
        ----------
        array : array-like
            Input array. If complex, will be converted to real array via
            one of the following NumPy functions: :numpydoc:`real`,
            :numpydoc:`imag`, :numpydoc:`abs`, or :numpydoc:`angle`.

        Returns
        -------
        numpy.array
            Returned array will be purely real.

        """
        if any(np.iscomplex(array)):
            parsed_array = getattr(np, option)(array)
        else:
            parsed_array = array

        return parsed_array
    return reducer


def propagate_err(z, dz, option):
    """Perform error propagation on a vector of complex uncertainties.

    Required to get values for magnitude (abs) and phase (angle)
    uncertainty.

    Parameters
    ----------
    z : array-like
        Array of complex or real numbers.
    dz : array-like
        Array of uncertainties corresponding to `z`. Must satisfy
        ``numpy.shape(dz) == numpy.shape(z)``.
    option : {'real', 'imag', 'abs', 'angle'}
        How to convert the array `z` to an array with real numbers.

    Returns
    -------
    numpy.array
        Returned array will be purely real.

    Notes
    -----
    Uncertainties are ``1/weights``. If the weights provided are real,
    they are assumed to apply equally to the real and imaginary parts. If
    the weights are complex, the real part of the weights are applied to
    the real part of the residual and the imaginary part is treated
    correspondingly.

    In the case where ``option='angle'`` and ``numpy.abs(z) == 0`` for any
    value of `z` the phase angle uncertainty becomes the entire circle and
    so a value of `math:pi` is returned.

    In the case where ``option='abs'`` and ``numpy.abs(z) == 0`` for any
    value of `z` the magnitude uncertainty is approximated by
    ``numpy.abs(dz)`` for that value.

    """
    if option not in ['real', 'imag', 'abs', 'angle']:
        raise ValueError(f"Invalid option ('{option}') for function 'propagate_err'.")

    if isinstance(dz, np.ndarray) and z.shape != dz.shape:
        raise ValueError(f"shape of z: {z.shape} != shape of dz: {dz.shape}")

    # Check the main vector for complex. Do nothing if real.
    if any(np.iscomplex(z)):
        # if uncertainties are real, apply them equally to
        # real and imaginary parts
        if all(np.isreal(dz)):
            dz = dz+1j*dz

        if option == 'real':
            err = np.real(dz)
        elif option == 'imag':
            err = np.imag(dz)
        elif option in ['abs', 'angle']:
            rz = np.real(z)
            iz = np.imag(z)

            rdz = np.real(dz)
            idz = np.imag(dz)

            # Don't spit out warnings for divide by zero. Will fix these later.
            with np.errstate(divide='ignore', invalid='ignore'):

                if option == 'abs':
                    # Standard error propagation for abs = sqrt(re**2 + im**2)
                    err = np.true_divide(np.sqrt((iz*idz)**2+(rz*rdz)**2),
                                         np.abs(z))

                    # For abs = 0, error is +/- abs(rdz + j idz)
                    err[err == np.inf] = np.abs(dz)[err == np.inf]

                if option == 'angle':
                    # Standard error propagation for angle = arctan(im/re)
                    err = np.true_divide(np.sqrt((rz*idz)**2+(iz*rdz)**2),
                                         np.abs(z)**2)

                    # For abs = 0, error is +/- pi (i.e. the whole circle)
                    err[err == np.inf] = np.pi
    else:
        err = dz

    return err


def coerce_arraylike(x):
    """
    coerce lists, tuples, and pandas Series, hdf5 Groups, etc to an
    ndarray float64 or complex128, but leave other data structures
    and objects unchanged
    """
    if isinstance(x, (list, tuple, Series)) or hasattr(x, '__array__'):
        if np.isrealobj(x):
            return np.asarray(x, dtype=np.float64)
        if np.iscomplexobj(x):
            return np.asarray(x, dtype=np.complex128)
    return x


class Model:
    """Create a model from a user-supplied model function."""

    _forbidden_args = ('data', 'weights', 'params')
    _invalid_ivar = "Invalid independent variable name ('%s') for function %s"
    _invalid_par = "Invalid parameter name ('%s') for function %s"
    _invalid_hint = "unknown parameter hint '%s' for param '%s'"
    _hint_names = ('value', 'vary', 'min', 'max', 'expr')
    valid_forms = ()

    def __init__(self, func, independent_vars=None, param_names=None,
                 nan_policy='raise', prefix='', name=None, **kws):
        """
        The model function will normally take an independent variable
        (generally, the first argument) and a series of arguments that are
        meant to be parameters for the model. It will return an array of
        data to model some data as for a curve-fitting problem.

        Parameters
        ----------
        func : callable
            Function to be wrapped.
        independent_vars : :obj:`list` of :obj:`str`, optional
            Arguments to `func` that are independent variables (default is
            None).
        param_names : :obj:`list` of :obj:`str`, optional
            Names of arguments to `func` that are to be made into
            parameters (default is None).
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            How to handle NaN and missing values in data. See Notes below.
        prefix : str, optional
            Prefix used for the model.
        name : str, optional
            Name for the model. When None (default) the name is the same
            as the model function (`func`).
        **kws : dict, optional
            Additional keyword arguments to pass to model function.

        Notes
        -----
        1. The model function must return an array that will be the same
        size as the data being modeled.

        2. Parameter names are inferred from the function arguments by default,
        and a residual function is automatically constructed.

        3. Specifying `independent_vars` here will explicitly name the
        independent variables for the Model.  in contrast, `param_names` is
        meant to help infer Parameter names for keyword arguments defined with
        ``**kws`` in the Model function.

        4. `nan_policy` sets what to do when a NaN or missing value is
        seen in the data. Should be one of:

           - `'raise'` : raise a `ValueError` (default)
           - `'propagate'` : do nothing
           - `'omit'` : drop missing data

        Examples
        --------
        The model function will normally take an independent variable
        (generally, the first argument) and a series of arguments that are
        meant to be parameters for the model. Thus, a simple peak using a
        Gaussian defined as:

        >>> import numpy as np
        >>> def gaussian(x, amp, cen, wid):
        ...     return amp * np.exp(-(x-cen)**2 / wid)

        can be turned into a Model with:

        >>> gmodel = Model(gaussian)

        this will automatically discover the names of the independent
        variables and parameters:

        >>> print(gmodel.param_names, gmodel.independent_vars)
        ['amp', 'cen', 'wid'], ['x']

        """
        self.func = func
        if not isinstance(prefix, str):
            prefix = ''
        if len(prefix) > 0 and not valid_symbol_name(prefix):
            raise ValueError(f"'{prefix}' is not a valid Model prefix")
        self._prefix = prefix

        self._param_root_names = param_names  # will not include prefixes
        self.independent_vars = independent_vars
        self._func_allargs = []
        self._func_haskeywords = False
        self.nan_policy = nan_policy

        self.opts = kws
        # the following has been changed from OrderedSet for the time being
        self.independent_vars_defvals = {}
        self.param_hints = {}
        self._param_names = []
        self._parse_params()
        if self.independent_vars is None:
            self.independent_vars = []
        if name is None and hasattr(self.func, '__name__'):
            name = self.func.__name__
        self._name = name

    def _reprstring(self, long=True):
        out = self._name
        opts = []
        if len(self._prefix) > 0:
            opts.append(f"prefix='{self._prefix}'")
        if long:
            for k, v in self.opts.items():
                opts.append(f"{k}='{v}'")
        if len(opts) > 0:
            out = f"{out}, {', '.join(opts)}"
        return f"Model({out})"

    def _get_state(self):
        """Save a Model for serialization.

        Note: like the standard-ish '__getstate__' method but not really
        useful with Pickle, and only useful with dill.

        This, and the companion function _buildmodel to use this serialized model
        now supports versions of 'state'.

        State Version History:
          original: up to and including version 1.2.2:
             state is a tuple of length 9:
                (self.func.__name__, funcdef, self._name, self._prefix,
                self.independent_vars, self._param_root_names,
                self.param_hints, self.nan_policy, self.opts)
             with opts used in the version 1.2 sense
          version 1.2.3 and beyond:
             state is a dict with a 'version' keyword and all other
             values from the original state in key/value pairs.
             The initial value for 'version' is '2'.
        """
        funcdef = self.func
        if self.func.__name__ == '_eval':
            funcdef = self.expr
        state = dict(version='2',
                     funcname=self.func.__name__,
                     funcdef=funcdef,
                     name=self._name,
                     prefix=self._prefix,
                     independent_vars=self.independent_vars,
                     param_root_names=self._param_root_names,
                     param_hints=self.param_hints,
                     nan_policy=self.nan_policy,
                     opts=self.opts)
        return (state, None, None)

    def _set_state(self, state, funcdefs=None):
        """Restore Model from serialization.

        Note: like the standard-ish '__setstate__' method but not really
        useful with Pickle.

        Parameters
        ----------
        state
            Serialized state from `_get_state`.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.

        """
        return _buildmodel(state, funcdefs=funcdefs)

    def dumps(self, **kws):
        """Dump serialization of Model as a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.dumps`.

        Returns
        -------
        str
            JSON string representation of Model.

        See Also
        --------
        loads, json.dumps

        """
        return json.dumps(encode4js(self._get_state()), **kws)

    def dump(self, fp, **kws):
        """Dump serialization of Model to a file.

        Parameters
        ----------
        fp : file-like object
            An open and `.write()`-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `json.dumps`.

        Returns
        -------
        int
            Return value from `fp.write()`: the number of characters
            written.

        See Also
        --------
        dumps, load, json.dump

        """
        return fp.write(self.dumps(**kws))

    def loads(self, s, funcdefs=None, **kws):
        """Load Model from a JSON string.

        Parameters
        ----------
        s : str
            Input JSON string containing serialized Model.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.
        **kws : optional
            Keyword arguments that are passed to `json.loads`.

        Returns
        -------
        Model
            Model created from JSON string.

        See Also
        --------
        dump, dumps, load, json.loads

        """
        tmp = decode4js(json.loads(s, **kws))
        return self._set_state(tmp, funcdefs=funcdefs)

    def load(self, fp, funcdefs=None, **kws):
        """Load JSON representation of Model from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and `.read()`-supporting file-like object.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.
        **kws : optional
            Keyword arguments that are passed to `loads`.

        Returns
        -------
        Model
            Model created from `fp`.

        See Also
        --------
        dump, loads, json.load

        """
        return self.loads(fp.read(), funcdefs=funcdefs, **kws)

    @property
    def name(self):
        """Return Model name."""
        return self._reprstring(long=True)

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def prefix(self):
        """Return Model prefix."""
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        """Change Model prefix."""
        self._prefix = value
        self._set_paramhints_prefix()
        self._param_names = []
        self._parse_params()

    def _set_paramhints_prefix(self):
        """Reset parameter hints for prefix: intended to be overwritten."""

    @property
    def param_names(self):
        """Return the parameter names of the Model."""
        return self._param_names

    def __repr__(self):
        """Return representation of Model."""
        return self._reprstring(long=True)

    def copy(self, **kwargs):
        """DOES NOT WORK."""
        raise NotImplementedError("Model.copy does not work. Make a new Model")

    def _parse_params(self):
        """Build parameters from function arguments."""
        if self.func is None:
            return
        kw_args = {}
        keywords_ = None
        indep_vars = []
        default_vals = {}
        # need to fetch the following from the function signature:
        #   pos_args: list of positional argument names
        #   kw_args: dict of keyword arguments with default values
        #   keywords_:  name of **kws argument or None
        # 1. limited support for asteval functions as the model functions:
        if hasattr(self.func, 'argnames') and hasattr(self.func, 'kwargs'):
            pos_args = self.func.argnames[:]
            default_vals = {v: inspect._empty for v in pos_args}
            for name, defval in self.func.kwargs:
                kw_args[name] = defval
                default_vals[name] = defval
        # 2. modern, best-practice approach: use inspect.signature
        else:
            pos_args = []
            sig = inspect.signature(self.func)
            for fnam, fpar in sig.parameters.items():
                if fpar.kind == fpar.VAR_KEYWORD:
                    keywords_ = fnam
                elif fpar.kind in (fpar.KEYWORD_ONLY,
                                   fpar.POSITIONAL_OR_KEYWORD):
                    default_vals[fnam] = fpar.default
                    if (isinstance(fpar.default, (float, int, complex))
                       and not isinstance(fpar.default, bool)):
                        kw_args[fnam] = fpar.default
                        pos_args.append(fnam)
                    elif fpar.default == fpar.empty:
                        pos_args.append(fnam)
                    else:
                        kw_args[fnam] = fpar.default
                        indep_vars.append(fnam)
                elif fpar.kind == fpar.POSITIONAL_ONLY:
                    raise ValueError("positional only arguments with '/' are not supported")
                elif fpar.kind == fpar.VAR_POSITIONAL:
                    raise ValueError(f"varargs '*{fnam}' is not supported")
        # inspection done

        self._func_haskeywords = keywords_ is not None
        self._func_allargs = list(default_vals.keys())
        for key in kw_args:
            if key not in self._func_allargs:
                self._func_allargs.append(key)

        if len(self._func_allargs) == 0 and keywords_ is not None:
            return

        self.independent_vars_defvals = {}
        # default independent_var = 1st argument
        if self.independent_vars is None:
            self.independent_vars = [pos_args.pop(0)]
            # default values for independent variables
            for vnam in indep_vars:
                dval = default_vals[vnam]
                if vnam in self.opts:
                    dval = self.opts[vnam]
                self.independent_vars_defvals[vnam] = dval
                if vnam not in self.independent_vars:
                    self.independent_vars.append(vnam)

        # default param names: all positional args
        # except independent variables
        self.def_vals = {}
        might_be_param = []
        if self._param_root_names is None:
            self._param_root_names = []
            for pname in pos_args:
                if pname not in self.independent_vars:
                    self._param_root_names.append(pname)
            for key, val in kw_args.items():
                if (not isinstance(val, bool) and
                        isinstance(val, (float, int))):
                    if key not in self._param_root_names:
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

        if self._prefix is None:
            self._prefix = ''
        names = [f"{self._prefix}{pname}" for pname in self._param_root_names]

        # check variables names for validity
        # The implicit magic in fit() requires us to disallow some
        fname = self.func.__name__
        for arg in self.independent_vars:
            if arg not in self._func_allargs or arg in self._forbidden_args:
                raise ValueError(self._invalid_ivar % (arg, fname))
        for arg in names:
            if (self._strip_prefix(arg) not in self._func_allargs or
                    arg in self._forbidden_args):
                raise ValueError(self._invalid_par % (arg, fname))
        # the following as been changed from OrderedSet for the time being.
        self._param_names = names[:]

    def set_param_hint(self, name, **kwargs):
        """Set *hints* to use when creating parameters with `make_params()`.

        The given hint can include optional bounds and constraints
        ``(value, vary, min, max, expr)``, which will be used by
        `Model.make_params()` when building default parameters.

        While this can be used to set initial values, `Model.make_params` or
        the function `create_params` should be preferred for creating
        parameters with initial values.

        The intended use here is to control how a Model should create
        parameters, such as setting bounds that are required by the mathematics
        of the model (for example, that a peak width cannot be negative), or to
        define common constrained parameters.

        Parameters
        ----------
        name : str
            Parameter name, can include the models `prefix` or not.
        **kwargs : optional
            Arbitrary keyword arguments, needs to be a Parameter attribute.
            Can be any of the following:

            - value : float, optional
                Numerical Parameter value.
            - vary : bool, optional
                Whether the Parameter is varied during a fit (default is
                True).
            - min : float, optional
                Lower bound for value (default is ``-numpy.inf``, no lower
                bound).
            - max : float, optional
                Upper bound for value (default is ``numpy.inf``, no upper
                bound).
            - expr : str, optional
                Mathematical expression used to constrain the value during
                the fit.

        Example
        --------
        >>> model = GaussianModel()
        >>> model.set_param_hint('sigma', min=0)

        """
        npref = len(self._prefix)
        if npref > 0 and name.startswith(self._prefix):
            name = name[npref:]

        if name not in self.param_hints:
            self.param_hints[name] = {}

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
            Parameter names and initial values or dictionaries of
                 values and attributes.

        Returns
        ---------
        params : Parameters
            Parameters object for the Model.

        Notes
        -----
        1. Parameter values can be numbers (floats or ints) to set the parameter
           value, or dictionaries with any of the following keywords:
           ``value``, ``vary``, ``min``, ``max``, ``expr``, ``brute_step``,
           ``is_init_value`` to set those parameter attributes.

        2. This method will also apply any default values or parameter hints
           that may have been defined for the model.

        Example
        --------
        >>> gmodel = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')
        >>> gmodel.make_params(peak_center=3200, bkg_offset=0, bkg_slope=0,
        ...                    peak_amplitdue=dict(value=100, min=2),
        ...                    peak_sigma=dict(value=25, min=0, max=1000))

        """
        params = Parameters()

        def setpar(par, val):
            # val is expected to be float-like or a dict: must have 'value' or 'expr' key
            if isinstance(val, dict):
                dval = val
            elif np.iscomplex(val) or isinstance(val, complex):
                dval = {'value': val.real}
            else:
                dval = {'value': float(val)}
            if len(dval) < 1 or not ('value' in dval or 'expr' in dval):
                raise TypeError(f'Invalid parameter value for {par}: {val}')

            par.set(**dval)

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
                setpar(par, kwargs[basename])
            if name in kwargs:
                setpar(par, kwargs[name])
            params.add(par)
            if verbose:
                print(f' - Adding parameter "{name}"')

        # next build parameters defined in param_hints
        # note that composites may define their own additional
        # convenience parameters here
        for basename, hint in self.param_hints.items():
            name = f"{self._prefix}{basename}"
            if name in params:
                par = params[name]
            else:
                par = Parameter(name=name)
                params.add(par)
                if verbose:
                    print(f' - Adding parameter for hint "{name}"')
            par._delay_asteval = True
            for item in self._hint_names:
                if item in hint:
                    setattr(par, item, hint[item])
            if basename in kwargs:
                setpar(par, kwargs[basename])
            # Add the new parameter to self._param_names
            if name not in self._param_names:
                self._param_names.append(name)

        # check for parameters that were initially flagged as independent
        # variables because the function signature used "key=None", "key=True",
        # or "key=False": these could actually be variables
        for key, val in kwargs.items():
            if key in params:
                continue
            if key in self.independent_vars:
                dxval = self.independent_vars_defvals.get(key, inspect._empty)
                if dxval is None or isinstance(dxval, bool):
                    name = f"{self._prefix}{key}"
                    par = Parameter(name=name)
                    setpar(par, val)
                    params.add(par)

        for p in params.values():
            p._delay_asteval = False
        return params

    def guess(self, data, x, **kws):
        """Guess starting values for the parameters of a Model.

        This is not implemented for all models, but is available for many
        of the built-in models.

        Parameters
        ----------
        data : array_like
            Array of data (i.e., y-values) to use to guess parameter values.
        x : array_like
            Array of values for the independent variable (i.e., x-values).
        **kws : optional
            Additional keyword arguments, passed to model function.

        Returns
        -------
        Parameters
            Initial, guessed values for the parameters of a Model.

        Raises
        ------
        NotImplementedError
            If the `guess` method is not implemented for a Model.

        Notes
        -----
        Should be implemented for each model subclass to run
        `self.make_params()`, update starting values and return a
        Parameters object.

        .. versionchanged:: 1.0.3
           Argument ``x`` is now explicitly required to estimate starting values.

        """
        cname = self.__class__.__name__
        msg = f'guess() not implemented for {cname}'
        raise NotImplementedError(msg)

    def _residual(self, params, data, weights, **kwargs):
        """Return the residual.

        Default residual: ``(data-model)*weights``.

        If the model returns complex values, the residual is computed by
        treating the real and imaginary parts separately. In this case, if
        the weights provided are real, they are assumed to apply equally
        to the real and imaginary parts. If the weights are complex, the
        real part of the weights are applied to the real part of the
        residual and the imaginary part is treated correspondingly.

        Since the underlying `scipy.optimize` routines expect
        ``numpy.float`` arrays, the only complex type supported is
        ``complex``.

        The "ravels" throughout are necessary to support `pandas.Series`.

        """
        model = self.eval(params, **kwargs)
        if self.nan_policy == 'raise' and not np.all(np.isfinite(model)):
            msg = ('The model function generated NaN values and the fit '
                   'aborted! Please check your model function and/or set '
                   'boundaries on parameters where applicable. In cases like '
                   'this, using "nan_policy=\'omit\'" will probably not work.')
            raise ValueError(msg)

        diff = data - model

        if diff.dtype is complex:
            # data/model are complex
            diff = diff.ravel().view(float)
            if weights is not None:
                if weights.dtype is complex:
                    # weights are complex
                    weights = weights.ravel().view(float)
                else:
                    # real weights but complex data
                    weights = (weights + 1j * weights).ravel().view(float)
        if weights is not None:
            diff *= weights
        return diff

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
        for key, val in self.independent_vars_defvals.items():
            if val is not inspect._empty:
                out[key] = val
        # 0: if a keyword argument is going to overwrite a parameter,
        #    save that value so it can be restored before returning
        saved_values = {}
        for name, val in kwargs.items():
            if name in params:
                saved_values[name] = params[name].value
                params[name].value = val

        if len(saved_values) > 0:
            params.update_constraints()

        # 1. fill in in all parameter values
        for name, par in params.items():
            if strip:
                name = self._strip_prefix(name)
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = par.value

        # 2. for each function argument, use 'prefix+varname' in params,
        # avoiding possible name collisions with unprefixed params
        if len(self._prefix) > 0:
            for fullname in self._param_names:
                if fullname in params:
                    name = self._strip_prefix(fullname)
                    if name in self._func_allargs or self._func_haskeywords:
                        out[name] = params[fullname].value

        # 3. kwargs might directly update function arguments
        validnames = [ivar for ivar in self.independent_vars]
        validnames.extend(self._func_allargs)
        for name, val in kwargs.items():
            if strip:
                name = self._strip_prefix(name)
            if name in validnames or self._func_haskeywords:
                out[name] = val

        # 4. finally, reset any values that have overwritten parameter values
        for name, val in saved_values.items():
            params[name].value = val
        return out

    def post_fit(self, fitresult):
        """function that is called just after fit, can be overloaded by
        subclasses to add non-fitting 'calculated parameters'
        """
        pass

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function args for all functions."""
        args = {}
        for key, val in self.make_funcargs(params, kwargs).items():
            args[f"{self._prefix}{key}"] = val
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
        numpy.ndarray, float, int or complex
            Value of model given the parameters and other arguments.

        Notes
        -----
        1. if `params` is None, the values for all parameters are expected
        to be provided as keyword arguments.

        2. If `params` is given, and a keyword argument for a parameter value
        is also given, the keyword argument will be used in place of the value
        in the value in `params`.

        3. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        4. The return types are generally `numpy.ndarray`, but may depends on
        the model function and input independent variables. That is, return
        values may be Python `float`, `int`, or  `complex` values.

        """
        return coerce_arraylike(self.func(**self.make_funcargs(params, kwargs)))

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
        dict
            Keys are prefixes for component model, values are value of
            each component.

        """
        key = self._prefix
        if len(key) < 1:
            key = self._name
        return {key: self.eval(params=params, **kwargs)}

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            nan_policy=None, calc_covar=True, max_nfev=None,
            coerce_farray=True, **kwargs):
        """Fit the model to the data using the supplied Parameters.

        Parameters
        ----------
        data : array_like
            Array of data to be fit.
        params : Parameters, optional
            Parameters to use in fit (default is None).
        weights : array_like, optional
            Weights to use for the calculation of the fit residual [i.e.,
            `weights*(data-fit)`]. Default is None; must have the same size as
            `data`.
        method : str, optional
            Name of fitting method to use (default is `'leastsq'`).
        iter_cb : callable, optional
            Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
            Whether to automatically scale the covariance matrix when
            calculating uncertainties (default is True).
        verbose : bool, optional
            Whether to print a message when a new parameter is added
            because of a hint (default is True).
        fit_kws : dict, optional
            Options to pass to the minimizer being used.
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than `'leastsq'` and `'least_squares'`.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        coerce_farray : bool, optional
            Whether to coerce data and independent data to be ndarrays
            with dtype of float64 (or complex128).  If set to False, data
            and independent data are not coerced at all, but the output of
            the model function will be. (default is True)
        **kwargs : optional
            Arguments to pass to the model function, possibly overriding
            parameters.

        Returns
        -------
        ModelResult

        Notes
        -----
        1. if `params` is None, the values for all parameters are expected
        to be provided as keyword arguments. Mixing `params` and
        keyword arguments is deprecated (see `Model.eval`).

        2. all non-parameter arguments for the model function, **including
        all the independent variables** will need to be passed in using
        keyword arguments.

        3. Parameters are copied on input, so that the original Parameter objects
        are unchanged, and the updated values are in the returned `ModelResult`.

        Examples
        --------
        Take ``t`` to be the independent variable and data to be the curve
        we will fit. Use keyword arguments to set initial guesses:

        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        Or, for more control, pass a Parameters object.

        >>> result = my_model.fit(data, params, t=t)

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
                warnings.warn(f"The keyword argument {name} does not " +
                              "match any arguments of the model function. " +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any(p not in params.keys() for p in self.param_names)
        blank_param = any((p.value is None and p.expr is None)
                          for p in params.values())
        if missing_param or blank_param:
            msg = ('Assign each parameter an initial value by passing '
                   'Parameters or keyword arguments to fit.\n')
            missing = [p for p in self.param_names if p not in params.keys()]
            blank = [name for name, p in params.items()
                     if p.value is None and p.expr is None]
            msg += f'Missing parameters: {str(missing)}\n'
            msg += f'Non initialized parameters: {str(blank)}'
            raise ValueError(msg)

        # Handle null/missing values.
        if nan_policy is not None:
            self.nan_policy = nan_policy

        mask = None
        if self.nan_policy == 'omit':
            mask = ~isnull(data)
            if mask is not None:
                data = data[mask]
            if weights is not None:
                weights = _align(weights, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            if var not in params and var not in self.opts:
                if var not in kwargs:
                    raise ValueError(f"'Missing independent variable '{var}'")
                if not np.isscalar(kwargs[var]):
                    kwargs[var] = _align(kwargs[var], mask, data)

        if coerce_farray:
            # coerce data and independent variable(s) that are 'array-like' (list,
            # tuples, pandas Series) to float64/complex128.
            data = coerce_arraylike(data)
            for var in self.independent_vars:
                if var not in params and var in kwargs:
                    kwargs[var] = coerce_arraylike(kwargs[var])

        if fit_kws is None:
            fit_kws = {}

        output = ModelResult(self, params, method=method, iter_cb=iter_cb,
                             scale_covar=scale_covar, fcn_kws=kwargs,
                             nan_policy=self.nan_policy, calc_covar=calc_covar,
                             max_nfev=max_nfev, **fit_kws)
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

    def __truediv__(self, other):
        """/"""
        return CompositeModel(self, other, operator.truediv)


class CompositeModel(Model):
    """Combine two models (`left` and `right`) with binary operator (`op`).

    Normally, one does not have to explicitly create a `CompositeModel`,
    but can use normal Python operators ``+``, ``-``, ``*``, and ``/`` to
    combine components as in::

    >>> mod = Model(fcn1) + Model(fcn2) * Model(fcn3)

    """

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
        The two models can use different independent variables.

        """
        if not isinstance(left, Model):
            raise ValueError(f'CompositeModel: argument {left} is not a Model')
        if not isinstance(right, Model):
            raise ValueError(f'CompositeModel: argument {right} is not a Model')
        if not callable(op):
            raise ValueError(f'CompositeModel: operator {op} is not callable')

        self.left = left
        self.right = right
        self.op = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += (f"\nTwo models have parameters named '{collision}'; "
                        "use distinct names.")
            raise NameError(msg)

        # the unique ``independent_vars`` of the left and right model are
        # combined to ``independent_vars`` of the ``CompositeModel``
        if 'independent_vars' not in kws:
            ivars = self.left.independent_vars + self.right.independent_vars
            kws['independent_vars'] = list(np.unique(ivars))
        if 'nan_policy' not in kws:
            kws['nan_policy'] = self.left.nan_policy

        # CompositeModel cannot have a prefix.
        if 'prefix' in kws:
            warnings.warn("CompositeModel ignores `prefix` argument")
            kws['prefix'] = ''

        def _tmp(self, *args, **kws):
            pass
        Model.__init__(self, _tmp, **kws)
        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints[f"{prefix}{basename}"] = hint

    def _parse_params(self):
        self._func_haskeywords = (self.left._func_haskeywords or
                                  self.right._func_haskeywords)
        self._func_allargs = (self.left._func_allargs +
                              self.right._func_allargs)
        self.def_vals = deepcopy(self.right.def_vals)
        self.def_vals.update(self.left.def_vals)
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=True):
        return (f"({self.left._reprstring(long=long)} "
                f"{self._known_ops.get(self.op, self.op)} "
                f"{self.right._reprstring(long=long)})")

    def eval(self, params=None, **kwargs):
        """Evaluate model function for composite model."""
        return self.op(self.left.eval(params=params, **kwargs),
                       self.right.eval(params=params, **kwargs))

    def eval_components(self, **kwargs):
        """Return dictionary of name, results for each component."""
        out = dict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    def post_fit(self, fitresult):
        """function that is called just after fit, can be overloaded by
        subclasses to add non-fitting 'calculated parameters'
        """
        self.left.post_fit(fitresult)
        self.right.post_fit(fitresult)

    @property
    def param_names(self):
        """Return parameter names for composite model."""
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """Return components for composite model."""
        return self.left.components + self.right.components

    def _get_state(self):
        return (self.left._get_state(),
                self.right._get_state(), self.op.__name__)

    def _set_state(self, state, funcdefs=None):
        return _buildmodel(state, funcdefs=funcdefs)

    def _make_all_args(self, params=None, **kwargs):
        """Generate **all** function arguments for all functions."""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out


def save_model(model, fname):
    """Save a Model to a file.

    Parameters
    ----------
    model : Model
        Model to be saved.
    fname : str
        Name of file for saved Model.

    """
    with open(fname, 'w') as fout:
        model.dump(fout)


def load_model(fname, funcdefs=None):
    """Load a saved Model from a file.

    Parameters
    ----------
    fname : str
        Name of file containing saved Model.
    funcdefs : dict, optional
        Dictionary of custom function names and definitions.

    Returns
    -------
    Model
        Model object loaded from file.

    """
    m = Model(lambda x: x)
    with open(fname) as fh:
        model = m.load(fh, funcdefs=funcdefs)
    return model


def _buildmodel(state, funcdefs=None):
    """Build Model from saved state.

    Intended for internal use only.

    """
    if len(state) != 3:
        raise ValueError("Cannot restore Model")
    known_funcs = {}
    for fname in lineshapes.functions:
        fcn = getattr(lineshapes, fname, None)
        if callable(fcn):
            known_funcs[fname] = fcn
    if funcdefs is None:
        funcdefs = {}
    else:
        known_funcs.update(funcdefs)

    left, right, op = state
    if op is None and right is None:
        if isinstance(left, tuple) and len(left) == 9:
            (fname, func, name, prefix, ivars, pnames,
             phints, nan_policy, opts) = left
        elif isinstance(left, dict) and 'version' in left:
            # for future-proofing, we could add "if left['version'] == '2':"
            # here to cover cases when 'version' changes
            fname = left.get('funcname', None)
            func = left.get('funcdef', None)
            name = left.get('name', None)
            prefix = left.get('prefix', None)
            ivars = left.get('independent_vars', None)
            pnames = left.get('param_root_names', None)
            phints = left.get('param_hints', None)
            nan_policy = left.get('nan_policy', None)
            opts = left.get('opts', None)
        else:
            raise ValueError("Cannot restore Model: unrecognized state data")

        # if the function definition was passed in, use that!
        if fname in funcdefs and fname != '_eval':
            func = funcdefs[fname]

        if not callable(func) and fname in known_funcs:
            func = known_funcs[fname]

        if func is None:
            raise ValueError("Cannot restore Model: model function not found")

        if fname == '_eval' and isinstance(func, str):
            from .models import ExpressionModel
            model = ExpressionModel(func, name=name,
                                    independent_vars=ivars,
                                    param_names=pnames,
                                    nan_policy=nan_policy, **opts)

        else:
            model = Model(func, name=name, prefix=prefix,
                          independent_vars=ivars, param_names=pnames,
                          nan_policy=nan_policy, **opts)

        for name, hint in phints.items():
            model.set_param_hint(name, **hint)
        return model
    else:
        lmodel = _buildmodel(left, funcdefs=funcdefs)
        rmodel = _buildmodel(right, funcdefs=funcdefs)
        return CompositeModel(lmodel, rmodel, getattr(operator, op))


def save_modelresult(modelresult, fname):
    """Save a ModelResult to a file.

    Parameters
    ----------
    modelresult : ModelResult
        ModelResult to be saved.
    fname : str
        Name of file for saved ModelResult.

    """
    with open(fname, 'w') as fout:
        modelresult.dump(fout)


def load_modelresult(fname, funcdefs=None):
    """Load a saved ModelResult from a file.

    Parameters
    ----------
    fname : str
        Name of file containing saved ModelResult.
    funcdefs : dict, optional
        Dictionary of custom function names and definitions.

    Returns
    -------
    ModelResult
        ModelResult object loaded from file.

    """
    params = Parameters()
    modres = ModelResult(Model(lambda x: x, None), params)
    with open(fname) as fh:
        mresult = modres.load(fh, funcdefs=funcdefs)
    return mresult


class ModelResult(Minimizer):
    """Result from the Model fit.

    This has many attributes and methods for viewing and working with the
    results of a fit using Model. It inherits from Minimizer, so that it
    can be used to modify and re-run the fit for the Model.

    """

    def __init__(self, model, params, data=None, weights=None,
                 method='leastsq', fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, nan_policy='raise',
                 calc_covar=True, max_nfev=None, **fit_kws):
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
            Weights to multiply ``(data-model)`` for fit residual.
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
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        calc_covar : bool, optional
            Whether to calculate the covariance matrix (default is True)
            for solvers other than `'leastsq'` and `'least_squares'`.
            Requires the ``numdifftools`` package to be installed.
        max_nfev : int or None, optional
            Maximum number of function evaluations (default is None). The
            default value depends on the fitting method.
        **fit_kws : optional
            Keyword arguments to send to minimization routine.

        """
        self.model = model
        self.data = data
        self.weights = weights
        self.method = method
        self.ci_out = None
        self.user_options = None
        self.init_params = deepcopy(params)
        Minimizer.__init__(self, model._residual, params,
                           fcn_args=fcn_args, fcn_kws=fcn_kws,
                           iter_cb=iter_cb, nan_policy=nan_policy,
                           scale_covar=scale_covar, calc_covar=calc_covar,
                           max_nfev=max_nfev, **fit_kws)

    def fit(self, data=None, params=None, weights=None, method=None,
            nan_policy=None, **kwargs):
        """Re-perform fit for a Model, given data and params.

        Parameters
        ----------
        data : array_like, optional
            Data to be modeled.
        params : Parameters, optional
            Parameters with initial values for model.
        weights : array_like, optional
            Weights to multiply ``(data-model)`` for fit residual.
        method : str, optional
            Name of minimization method to use (default is `'leastsq'`).
        nan_policy : {'raise', 'propagate', 'omit'}, optional
            What to do when encountering NaNs when fitting Model.
        **kwargs : optional
            Keyword arguments to send to minimization routine.

        """
        if data is not None:
            self.data = data
        if params is not None:
            self.init_params = deepcopy(params)
        else:
            self.init_params = deepcopy(self.params)

        if weights is not None:
            self.weights = weights
        if method is not None:
            self.method = method
        if nan_policy is not None:
            self.nan_policy = nan_policy

        self.ci_out = None
        self.userargs = (self.data, self.weights)
        self.userkws.update(kwargs)
        self.init_fit = self.model.eval(params=self.init_params, **self.userkws)
        _ret = self.minimize(method=self.method, params=self.init_params)
        self.model.post_fit(_ret)
        _ret.params.create_uvars(covar=_ret.covar)

        for attr in dir(_ret):
            if not attr.startswith('_'):
                try:
                    setattr(self, attr, getattr(_ret, attr))
                except AttributeError:
                    pass

        self.init_values = self.model._make_all_args(self.init_params)
        self.best_values = self.model._make_all_args(_ret.params)
        self.best_fit = self.model.eval(params=_ret.params, **self.userkws)
        if (self.data is not None and len(self.data) > 1
           and isinstance(self.best_fit, np.ndarray)
           and len(self.best_fit) > 1):
            dat = coerce_arraylike(self.data)
            resid = ((dat - self.best_fit)**2).sum()
            sstot = ((dat - dat.mean())**2).sum()
            self.rsquared = 1.0 - resid/max(tiny, sstot)

    def eval(self, params=None, **kwargs):
        """Evaluate model function.

        Parameters
        ----------
        params : Parameters, optional
            Parameters to use.
        **kwargs : optional
            Options to send to Model.eval().

        Returns
        -------
        numpy.ndarray, float, int, or complex
            Array or value for the evaluated model.

        """
        userkws = self.userkws.copy()
        userkws.update(kwargs)
        if params is None:
            params = self.params
        return self.model.eval(params=params, **userkws)

    def eval_components(self, params=None, **kwargs):
        """Evaluate each component of a composite model function.

        Parameters
        ----------
        params : Parameters, optional
            Parameters, defaults to ModelResult.params.
        **kwargs : optional
            Keyword arguments to pass to model function.

        Returns
        -------
        dict
            Keys are prefixes of component models, and values are the
            estimated model value for each component of the model.

        """
        userkws = self.userkws.copy()
        userkws.update(kwargs)
        if params is None:
            params = self.params
        return self.model.eval_components(params=params, **userkws)

    def eval_uncertainty(self, params=None, sigma=1, dscale=0.01, **kwargs):
        """Evaluate the uncertainty of the *model function*.

        This can be used to give confidence bands for the model from the
        uncertainties in the best-fit parameters.

        Parameters
        ----------
        params : Parameters, optional
            Parameters, defaults to ModelResult.params.
        sigma : float, optional
            Confidence level, i.e. how many sigma (default is 1).
        dscale : float, optional
            Scale for derivative steps (default is 0.01).
        **kwargs : optional
            Values of options, independent variables, etcetera.

        Returns
        -------
        numpy.ndarray
            Uncertainty at each value of the model.

        Notes
        -----
        1. This is based on the excellent and clear example from
           https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals,
           which references the original work of:
           J. Wolberg, Data Analysis Using the Method of Least Squares, 2006, Springer
        2. The value of sigma is number of `sigma` values, and is converted
           to a probability. Values of 1, 2, or 3 give probabilities of
           0.6827, 0.9545, and 0.9973, respectively. If the sigma value is
           < 1, it is interpreted as the probability itself. That is,
           ``sigma=1`` and ``sigma=0.6827`` will give the same results,
           within precision errors.
        3. The derivatives are calculated by stepping each Parameter from its best value to
           to +/- stderr*dscale, where `dscale` can be passed in and defaults to 0.01.
        4. Sets attributes of `dely` for the uncertainty of the model
           (which will be the same as the array returned by this method) and
           `dely_comps`, a dictionary of `dely` for each component.
        5. Sets the attribute of `dely_predicted` for the 'predicted interval', the sigma-scaled
           quadrature sum of the uncertainty interval `dely` and reduced chi-square. This should
           give an idea of the expected range in the data.

        Examples
        --------

        >>> out = model.fit(data, params, x=x)
        >>> dely = out.eval_uncertainty(x=x)
        >>> plt.plot(x, data)
        >>> plt.plot(x, out.best_fit)
        >>> plt.fill_between(x, out.best_fit-dely,
        ...                  out.best_fit+dely, color='#888888')

        """
        userkws = self.userkws.copy()
        userkws.update(kwargs)
        if params is None:
            params = self.params

        nvarys = self.nvarys
        # ensure fjac and df2 are correct size if independent var updated by kwargs
        feval = self.model.eval(params, **userkws)
        ndata = np.atleast_1d(feval).view('float64').ravel().size  # allows feval to be complex
        covar = self.covar
        if any(p.stderr is None for p in params.values()):
            return np.zeros(ndata)

        # '0' would be an invalid prefix, here signifying 'Full'
        fjac = {'0': np.zeros((nvarys, ndata), dtype='float64')}
        df2 = {'0': np.zeros(ndata, dtype='float64')}

        for comp in self.model.components:
            label = comp.prefix if len(comp.prefix) > 1 else comp._name
            fjac[label] = np.zeros((nvarys, ndata), dtype='float64')
            df2[label] = np.zeros(ndata, dtype='float64')

        # find derivative by hand!
        pars = params.copy()
        for i in range(nvarys):
            pname = self.var_names[i]
            val0 = params[pname].value
            dval = params[pname].stderr*dscale

            pars[pname].value = val0 + dval
            res1 = {'0': self.model.eval(pars, **userkws)}
            res1.update(self.model.eval_components(params=pars, **userkws))

            pars[pname].value = val0 - dval
            res2 = {'0': self.model.eval(pars, **userkws)}
            res2.update(self.model.eval_components(params=pars, **userkws))

            pars[pname].value = val0
            for key in fjac:
                fjac[key][i] = (np.atleast_1d(res1[key]).view('float64').ravel()
                                - np.atleast_1d(res2[key]).view('float64').ravel()) / (2*dval)

        for i in range(nvarys):
            for j in range(nvarys):
                for key in fjac:
                    df2[key] += fjac[key][i] * fjac[key][j] * covar[i, j]

        if sigma < 1.0:
            prob = sigma
        else:
            prob = erf(sigma/np.sqrt(2))

        scale = t.ppf((prob+1)/2.0, self.ndata-nvarys)

        # for complex data, convert back to real/imag pairs
        if isinstance(feval, float):
            feval = np.float64(feval)
        if feval.dtype in ('complex64', 'complex128'):
            for key in fjac:
                df2[key] = df2[key].view(feval.dtype)

        for key in fjac:
            df2[key] = df2[key].reshape(feval.shape)

        df2_total = df2.pop('0')
        self.dely = scale * np.sqrt(df2_total)
        self.dely_predicted = scale * np.sqrt(df2_total + self.redchi)

        self.dely_comps = {}
        for key in df2:
            self.dely_comps[key] = scale * np.sqrt(df2[key])
        return self.dely

    def conf_interval(self, **kwargs):
        """Calculate the confidence intervals for the variable parameters.

        Confidence intervals are calculated using the
        :func:`confidence.conf_interval` function and keyword arguments
        (`**kwargs`) are passed to that function. The result is stored in
        the :attr:`ci_out` attribute so that it can be accessed without
        recalculating them.

        """
        self.ci_out = conf_interval(self, self, **kwargs)
        return self.ci_out

    def ci_report(self, with_offset=True, ndigits=5, **kwargs):
        """Return a formatted text report of the confidence intervals.

        Parameters
        ----------
        with_offset : bool, optional
            Whether to subtract best value from all other values (default
            is True).
        ndigits : int, optional
            Number of significant digits to show (default is 5).
        **kwargs : optional
            Keyword arguments that are passed to the `conf_interval`
            function.

        Returns
        -------
        str
            Text of formatted report on confidence intervals.

        """
        return ci_report(self.conf_interval(**kwargs),
                         with_offset=with_offset, ndigits=ndigits)

    def fit_report(self, modelpars=None, show_correl=True,
                   min_correl=0.1, sort_pars=False, correl_mode='list'):
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
            (default is False). If False, then the parameters will be
            listed in the order as they were added to the Parameters
            dictionary. If callable, then this (one argument) function is
            used to extract a comparison key from each list element.
        correl_mode : {'list', table'} str, optional
            Mode for how to show correlations. Can be either 'list' (default)
            to show a sorted (if ``sort_pars`` is True) list of correlation
            values, or 'table' to show a complete, formatted table of
            correlations.

        Returns
        -------
        str
            Multi-line text of fit report.

        """
        report = fit_report(self, modelpars=modelpars, show_correl=show_correl,
                            min_correl=min_correl, sort_pars=sort_pars,
                            correl_mode=correl_mode)

        modname = self.model._reprstring(long=True)
        return f'[[Model]]\n    {modname}\n{report}'

    def _repr_html_(self, show_correl=True, min_correl=0.1):
        """Return a HTML representation of parameters data."""
        report = fitreport_html_table(self, show_correl=show_correl,
                                      min_correl=min_correl)
        modname = self.model._reprstring(long=True)
        return f"<h2>Fit Result</h2> <p>Model: {modname}</p> {report}"

    def summary(self):
        """Return a dictionary with statistics and attributes of a ModelResult.

        Returns
        -------
        dict
            Dictionary of statistics and many attributes from a ModelResult.

        Notes
        ------
        1. values for data arrays are not included.

        2. The result summary dictionary will include the following entries:

          ``model``, ``method``, ``ndata``, ``nvarys``, ``nfree``, ``chisqr``,
          ``redchi``, ``aic``, ``bic``, ``rsquared``, ``nfev``, ``max_nfev``,
          ``aborted``, ``errorbars``, ``success``, ``message``,
          ``lmdif_message``, ``ier``, ``nan_policy``, ``scale_covar``,
          ``calc_covar``, ``ci_out``, ``col_deriv``, ``flatchain``,
          ``call_kws``, ``var_names``, ``user_options``, ``kws``,
          ``init_values``, ``best_values``, and ``params``.

        where 'params' is a list of parameter "states": tuples with entries of
        ``(name, value, vary, expr, min, max, brute_step, stderr, correl,
        init_value, user_data)``.

        3. The result will include only plain Python objects, and so should be
        easily serializable with JSON or similar tools.

        """
        summary = {'model': self.model._reprstring(long=True),
                   'method': self.method}

        for attr in ('ndata', 'nvarys', 'nfree', 'chisqr', 'redchi', 'aic',
                     'bic', 'rsquared', 'nfev', 'max_nfev', 'aborted',
                     'errorbars', 'success', 'message', 'lmdif_message', 'ier',
                     'nan_policy', 'scale_covar', 'calc_covar', 'ci_out',
                     'col_deriv', 'flatchain', 'call_kws', 'var_names',
                     'user_options', 'kws', 'init_values', 'best_values'):
            val = getattr(self, attr, None)
            if isinstance(val, np.float64):
                val = float(val)
            elif isinstance(val, (np.int32, np.int64)):
                val = int(val)
            elif isinstance(val, np.bool_):
                val = bool(val)
            elif isinstance(val, bytes):
                val = str(val, encoding='UTF-8')
            summary[attr] = val

        summary['params'] = [par.__getstate__() for par in self.params.values()]
        return summary

    def dumps(self, **kws):
        """Represent ModelResult as a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.dumps`.

        Returns
        -------
        str
            JSON string representation of ModelResult.

        See Also
        --------
        loads, json.dumps

        """
        out = {'__class__': 'lmfit.ModelResult', '__version__': '2',
               'model': encode4js(self.model._get_state())}

        for attr in ('params', 'init_params'):
            out[attr] = getattr(self, attr).dumps()

        for attr in ('aborted', 'aic', 'best_values', 'bic', 'chisqr',
                     'ci_out', 'col_deriv', 'covar', 'errorbars', 'flatchain',
                     'ier', 'init_values', 'lmdif_message', 'message',
                     'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
                     'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
                     'calc_covar', 'success', 'userargs', 'userkws', 'values',
                     'var_names', 'weights', 'user_options'):
            try:
                val = getattr(self, attr)
            except AttributeError:
                continue
            if isinstance(val, np.bool_):
                val = bool(val)
            out[attr] = encode4js(val)

        val = out.get('message', '')
        if isinstance(val, bytes):
            out['message'] = str(val, encoding='ASCII')

        return json.dumps(out, **kws)

    def dump(self, fp, **kws):
        """Dump serialization of ModelResult to a file.

        Parameters
        ----------
        fp : file-like object
            An open and `.write()`-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `json.dumps`.

        Returns
        -------
        int
            Return value from `fp.write()`: the number of characters
            written.

        See Also
        --------
        dumps, load, json.dump

        """
        return fp.write(self.dumps(**kws))

    def loads(self, s, funcdefs=None, **kws):
        """Load ModelResult from a JSON string.

        Parameters
        ----------
        s : str
            String representation of ModelResult, as from `dumps`.
        funcdefs : dict, optional
            Dictionary of custom function names and definitions.
        **kws : optional
            Keyword arguments that are passed to `json.loads`.

        Returns
        -------
        ModelResult
            ModelResult instance from JSON string representation.

        See Also
        --------
        load, dumps, json.dumps

        """
        modres = json.loads(s, **kws)
        if 'modelresult' not in modres['__class__'].lower():
            raise AttributeError('ModelResult.loads() needs saved ModelResult')

        modres = decode4js(modres)
        if 'model' not in modres or 'params' not in modres:
            raise AttributeError('ModelResult.loads() needs valid ModelResult')

        # model
        self.model = _buildmodel(decode4js(modres['model']), funcdefs=funcdefs)

        if funcdefs:
            # Remove model function so as not pass it into the _asteval.symtable
            funcdefs.pop(self.model.func.__name__, None)

        # how params are saved was changed with version 2:
        modres_vers = modres.get('__version__', '1')
        if modres_vers == '1':
            for target in ('params', 'init_params'):
                state = {'unique_symbols': modres['unique_symbols'], 'params': []}
                for parstate in modres['params']:
                    _par = Parameter(name='')
                    _par.__setstate__(parstate)
                    state['params'].append(_par)
                _params = Parameters(usersyms=funcdefs)
                _params.__setstate__(state)
                setattr(self, target, _params)

        elif modres_vers == '2':
            for target in ('params', 'init_params'):
                _pars = Parameters()
                _pars.loads(modres[target])
                if funcdefs:
                    for key, val in funcdefs.items():
                        _pars._asteval.symtable[key] = val
                setattr(self, target, _pars)

        for attr in ('aborted', 'aic', 'best_fit', 'best_values', 'bic',
                     'chisqr', 'ci_out', 'col_deriv', 'covar', 'data',
                     'errorbars', 'fjac', 'flatchain', 'ier', 'init_fit',
                     'init_values', 'kws', 'lmdif_message', 'message',
                     'method', 'nan_policy', 'ndata', 'nfev', 'nfree',
                     'nvarys', 'redchi', 'residual', 'rsquared', 'scale_covar',
                     'calc_covar', 'success', 'userargs', 'userkws',
                     'var_names', 'weights', 'user_options'):
            setattr(self, attr, decode4js(modres.get(attr, None)))

        self.best_fit = self.model.eval(self.params, **self.userkws)
        if len(self.userargs) == 2:
            self.data = self.userargs[0]
            self.weights = self.userargs[1]

        for parname, val in self.init_values.items():
            par = self.init_params.get(parname, None)
            if par is not None:
                par.correl = par.stderr = None
                par.value = par.init_value = self.init_values[parname]

        self.init_fit = self.model.eval(self.init_params, **self.userkws)
        self.result = MinimizerResult()
        self.result.params = self.params

        if self.errorbars and self.covar is not None:
            self.uvars = self.result.params.create_uvars(covar=self.covar)

        self.init_vals = list(self.init_values.items())
        return self

    def load(self, fp, funcdefs=None, **kws):
        """Load JSON representation of ModelResult from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and `.read()`-supporting file-like object.
        funcdefs : dict, optional
            Dictionary of function definitions to use to construct Model.
        **kws : optional
            Keyword arguments that are passed to `loads`.

        Returns
        -------
        ModelResult
            ModelResult created from `fp`.

        See Also
        --------
        dump, loads, json.load

        """
        return self.loads(fp.read(), funcdefs=funcdefs, **kws)

    @_ensureMatplotlib
    def plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--',
                 xlabel=None, ylabel=None, yerr=None, numpoints=None,
                 data_kws=None, fit_kws=None, init_kws=None, ax_kws=None,
                 show_init=False, parse_complex='abs', title=None):
        """Plot the fit results using matplotlib, if available.

        The plot will include the data points, the initial fit curve
        (optional, with ``show_init=True``), and the best-fit curve. If
        the fit model included weights or if `yerr` is specified,
        errorbars will also be plotted.

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
            If provided, the final and initial fit curves are evaluated
            not only at data points, but refined to contain `numpoints`
            points in total.
        data_kws : dict, optional
            Keyword arguments passed to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed to the plot function for the initial
            conditions of the fit.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if a new one is created.
        show_init : bool, optional
            Whether to show the initial conditions for the fit (default is
            False).
        parse_complex : {'abs', 'real', 'imag', 'angle'}, optional
            How to reduce complex data for plotting. Options are one of:
            `'abs'` (default), `'real'`, `'imag'`, or `'angle'`, which
            correspond to the NumPy functions with the same name.
        title : str, optional
            Matplotlib format string for figure title.

        Returns
        -------
        matplotlib.axes.Axes

        See Also
        --------
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.

        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of `matplotlib.axes.Axes.plot`.

        If `yerr` is specified or if the fit model included weights, then
        `matplotlib.axes.Axes.errorbar` is used to plot the data. If
        `yerr` is not specified and the fit includes weights, `yerr` set
        to ``1/self.weights``.

        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.

        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.

        """
        from matplotlib import pyplot as plt
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_kws is None:
            ax_kws = {}

        # The function reduce_complex will convert complex vectors into real vectors
        reduce_complex = get_reducer(parse_complex)

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.axes(**ax_kws)

        x_array = self.userkws[independent_var]

        # make a dense array for x-axis if data is not dense
        if numpoints is not None and len(self.data) < numpoints:
            x_array_dense = np.linspace(min(x_array), max(x_array), numpoints)
        else:
            x_array_dense = x_array

        if show_init:
            y_eval_init = self.model.eval(self.init_params,
                                          **{independent_var: x_array_dense})
            if isinstance(self.model, (lmfit.models.ConstantModel,
                                       lmfit.models.ComplexConstantModel)):
                y_eval_init *= np.ones(x_array_dense.size)

            ax.plot(
                x_array_dense, reduce_complex(y_eval_init), initfmt,
                label='initial fit', **init_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights
        if yerr is not None:
            ax.errorbar(x_array, reduce_complex(self.data),
                        yerr=propagate_err(self.data, yerr, parse_complex),
                        fmt=datafmt, label='data', **data_kws)
        else:
            ax.plot(x_array, reduce_complex(self.data),
                    datafmt, label='data', **data_kws)

        y_eval = self.model.eval(self.params, **{independent_var: x_array_dense})
        if isinstance(self.model, (lmfit.models.ConstantModel,
                                   lmfit.models.ComplexConstantModel)):
            y_eval *= np.ones(x_array_dense.size)

        ax.plot(x_array_dense, reduce_complex(y_eval), fitfmt, label='best fit',
                **fit_kws)

        if title:
            ax.set_title(title)
        elif ax.get_title() == '':
            ax.set_title(self.model.name)
        if xlabel is None:
            ax.set_xlabel(independent_var)
        else:
            ax.set_xlabel(xlabel)
        if ylabel is None:
            ax.set_ylabel('y')
        else:
            ax.set_ylabel(ylabel)
        ax.legend()
        return ax

    @_ensureMatplotlib
    def plot_residuals(self, ax=None, datafmt='o', yerr=None, data_kws=None,
                       fit_kws=None, ax_kws=None, parse_complex='abs',
                       title=None):
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
            Keyword arguments passed to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed to the plot function for fitted curve.
        ax_kws : dict, optional
            Keyword arguments for a new axis, if a new one is created.
        parse_complex : {'abs', 'real', 'imag', 'angle'}, optional
            How to reduce complex data for plotting. Options are one of:
            `'abs'` (default), `'real'`, `'imag'`, or `'angle'`, which
            correspond to the NumPy functions with the same name.
        title : str, optional
            Matplotlib format string for figure title.

        Returns
        -------
        matplotlib.axes.Axes

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.

        Notes
        -----
        For details about plot format strings and keyword arguments see
        documentation of `matplotlib.axes.Axes.plot`.

        If `yerr` is specified or if the fit model included weights, then
        `matplotlib.axes.Axes.errorbar` is used to plot the data. If
        `yerr` is not specified and the fit includes weights, `yerr` set
        to ``1/self.weights``.

        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.

        If `ax` is None then `matplotlib.pyplot.gca(**ax_kws)` is called.

        """
        from matplotlib import pyplot as plt
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if ax_kws is None:
            ax_kws = {}

        # The function reduce_complex will convert complex vectors into real vectors
        reduce_complex = get_reducer(parse_complex)

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.axes(**ax_kws)

        x_array = self.userkws[independent_var]

        ax.axhline(0, **fit_kws, color='k')

        y_eval = self.model.eval(self.params, **{independent_var: x_array})
        if isinstance(self.model, (lmfit.models.ConstantModel,
                                   lmfit.models.ComplexConstantModel)):
            y_eval *= np.ones(x_array.size)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights

        residuals = reduce_complex(self.data) - reduce_complex(self.eval())
        if yerr is not None:
            ax.errorbar(x_array, residuals,
                        yerr=propagate_err(self.data, yerr, parse_complex),
                        fmt=datafmt, **data_kws)
        else:
            ax.plot(x_array, residuals, datafmt, **data_kws)

        if title:
            ax.set_title(title)
        elif ax.get_title() == '':
            ax.set_title(self.model.name)
        ax.set_ylabel('residuals')
        return ax

    @_ensureMatplotlib
    def plot(self, datafmt='o', fitfmt='-', initfmt='--', xlabel=None,
             ylabel=None, yerr=None, numpoints=None, fig=None, data_kws=None,
             fit_kws=None, init_kws=None, ax_res_kws=None, ax_fit_kws=None,
             fig_kws=None, show_init=False, parse_complex='abs', title=None):
        """Plot the fit results and residuals using matplotlib.

        The method will produce a matplotlib figure (if package available)
        with both results of the fit and the residuals plotted. If the fit
        model included weights, errorbars will also be plotted. To show
        the initial conditions for the fit, pass the argument
        ``show_init=True``.

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
            If provided, the final and initial fit curves are evaluated
            not only at data points, but refined to contain `numpoints`
            points in total.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default is None, which means use
            the current pyplot figure or create one if there is none.
        data_kws : dict, optional
            Keyword arguments passed to the plot function for data points.
        fit_kws : dict, optional
            Keyword arguments passed to the plot function for fitted curve.
        init_kws : dict, optional
            Keyword arguments passed to the plot function for the initial
            conditions of the fit.
        ax_res_kws : dict, optional
            Keyword arguments for the axes for the residuals plot.
        ax_fit_kws : dict, optional
            Keyword arguments for the axes for the fit plot.
        fig_kws : dict, optional
            Keyword arguments for a new figure, if a new one is created.
        show_init : bool, optional
            Whether to show the initial conditions for the fit (default is
            False).
        parse_complex : {'abs', 'real', 'imag', 'angle'}, optional
            How to reduce complex data for plotting. Options are one of:
            `'abs'` (default), `'real'`, `'imag'`, or `'angle'`, which
            correspond to the NumPy functions with the same name.
        title : str, optional
            Matplotlib format string for figure title.

        Returns
        -------
        matplotlib.figure.Figure

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.

        Notes
        -----
        The method combines `ModelResult.plot_fit` and
        `ModelResult.plot_residuals`.

        If `yerr` is specified or if the fit model included weights, then
        `matplotlib.axes.Axes.errorbar` is used to plot the data. If
        `yerr` is not specified and the fit includes weights, `yerr` set
        to ``1/self.weights``.

        If model returns complex data, `yerr` is treated the same way that
        weights are in this case.

        If `fig` is None then `matplotlib.pyplot.figure(**fig_kws)` is
        called, otherwise `fig_kws` is ignored.

        """
        from matplotlib import pyplot as plt
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
                      fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws,
                      show_init=show_init, parse_complex=parse_complex,
                      title=title)
        self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr,
                            data_kws=data_kws, fit_kws=fit_kws,
                            ax_kws=ax_res_kws, parse_complex=parse_complex,
                            title=title)
        plt.setp(ax_res.get_xticklabels(), visible=False)
        ax_fit.set_title('')
        return fig
