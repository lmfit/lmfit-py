"""Parameter class."""

from copy import deepcopy
import json
import warnings

from asteval import Interpreter, get_ast_names, valid_symbol_name
from numpy import arcsin, array, cos, inf, isclose, sin, sqrt
from scipy.linalg import LinAlgError
import scipy.special
from uncertainties import correlated_values, ufloat
from uncertainties import wrap as uwrap

from .jsonutils import decode4js, encode4js
from .lineshapes import tiny
from .printfuncs import params_html_table

SCIPY_FUNCTIONS = {'gamfcn': scipy.special.gamma,
                   'loggammafcn': scipy.special.loggamma,
                   'betalnfnc': scipy.special.betaln}
for fnc_name in ('erf', 'erfc', 'wofz'):
    SCIPY_FUNCTIONS[fnc_name] = getattr(scipy.special, fnc_name)


def check_ast_errors(expr_eval):
    """Check for errors derived from asteval."""
    if len(expr_eval.error) > 0:
        expr_eval.raise_exception(None)


class Writer:
    """Replace 'stdout' and 'stderr' for asteval."""
    def __init__(self, **kws):
        self.messages = []
        for k, v in kws.items():
            setattr(self, k, v)

    def write(self, msg):
        """Internal writer."""
        o = msg.strip()
        if len(o) > 0:
            self.messages.append(msg)


def asteval_with_uncertainties(*vals, obj=None, pars=None, names=None, **kwargs):
    """Calculate object value, given values for variables.

    This is used by the uncertainties package to calculate the
    uncertainty in an object even with a complicated expression.
    """
    _asteval = getattr(pars, '_asteval', None)
    if obj is None or pars is None or names is None or _asteval is None:
        return 0
    for val, name in zip(vals, names):
        _asteval.symtable[name] = val

    # re-evaluate constraint parameters topropagate uncertainties
    [p._getval() for p in pars.values()]
    return _asteval.eval(obj._expr_ast)


class Parameters(dict):
    """A dictionary of Parameter objects.

    It should contain all Parameter objects that are required to specify
    a fit model. All minimization and Model fitting routines in lmfit will
    use exactly one Parameters object, typically given as the first
    argument to the objective function.

    All keys of a Parameters() instance must be strings and valid Python
    symbol names, so that the name must match ``[a-z_][a-z0-9_]*`` and
    cannot be a Python reserved word.

    All values of a Parameters() instance must be Parameter objects.

    A Parameters(xs) instance includes an `asteval` Interpreter used for
    evaluation of constrained Parameters.

    Parameters() support copying and pickling, and have methods to convert
    to and from serializations using json strings.

    """

    def __init__(self, usersyms=None):
        """
        Arguments
        ---------
        usersyms : dict, optional
            Dictionary of symbols to add to the
            :class:`asteval.Interpreter` (default is None).

        """
        super().__init__(self)
        self._ast_msgs = Writer()
        self._asteval = Interpreter(writer=self._ast_msgs,
                                    err_writer=self._ast_msgs)

        _syms = {}
        _syms.update(SCIPY_FUNCTIONS)
        if usersyms is not None:
            _syms.update(usersyms)
        for key, val in _syms.items():
            self._asteval.symtable[key] = val

    def _writer(self, msg):
        self._asteval_msgs.append(msg)

    def copy(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def update(self, other):
        """Update values and symbols with another Parameters object."""
        if not isinstance(other, Parameters):
            raise ValueError(f"'{other}' is not a Parameters object")
        self.add_many(*other.values())
        for sym in other._asteval.user_defined_symbols():
            self._asteval.symtable[sym] = other._asteval.symtable[sym]
        return self

    def __copy__(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo):
        """Implementation of Parameters.deepcopy().

        The method needs to make sure that `asteval` is available and that
        all individual Parameter objects are copied.

        """
        _pars = self.__class__()

        # find the symbols that were added by users, not during construction
        unique_symbols = {}
        for key in self._asteval.user_defined_symbols():
            try:
                val = deepcopy(self._asteval.symtable[key])
                unique_symbols[key] = val
            except (TypeError, ValueError):
                unique_symbols[key] = self._asteval.symtable[key]

        _pars._asteval.symtable.update(unique_symbols)

        # we're just about to add a lot of Parameter objects to the newly
        parameter_list = []
        for key, par in self.items():
            if isinstance(par, Parameter):
                param = Parameter(name=par.name,
                                  value=float(par.value),
                                  min=par.min,
                                  max=par.max)
                param.vary = par.vary
                param.brute_step = par.brute_step
                param.stderr = par.stderr
                param.correl = deepcopy(par.correl)
                param.init_value = par.init_value
                param.expr = par.expr
                param.user_data = deepcopy(par.user_data)
                parameter_list.append(param)

        _pars.add_many(*parameter_list)

        return _pars

    def __setitem__(self, key, par):
        """Set items of Parameters object."""
        if key not in self and not valid_symbol_name(key):
            raise KeyError(f"'{key}' is not a valid Parameters name")
        if par is not None and not isinstance(par, Parameter):
            raise ValueError(f"'{par}' is not a Parameter")
        dict.__setitem__(self, key, par)
        par.name = key
        par._expr_eval = self._asteval
        self._asteval.symtable[key] = float(par.value)

    def __add__(self, other):
        """Add Parameters objects."""
        if not isinstance(other, Parameters):
            raise ValueError(f"'{other}' is not a Parameters object")
        out = deepcopy(self)
        out.add_many(*other.values())
        for sym in other._asteval.user_defined_symbols():
            if sym not in out._asteval.symtable:
                out._asteval.symtable[sym] = other._asteval.symtable[sym]
        return out

    def __iadd__(self, other):
        """Add/assign Parameters objects."""
        self.update(other)
        return self

    def __array__(self):
        """Convert Parameters to array."""
        return array([float(k) for k in self.values()])

    def __reduce__(self):
        """Reduce Parameters instance such that it can be pickled."""
        # make a list of all the parameters
        params = [self[k] for k in self]

        # find the symbols from _asteval.symtable, that need to be remembered.
        unique_symbols = {key: deepcopy(self._asteval.symtable[key])
                          for key in self._asteval.user_defined_symbols()}

        return self.__class__, (), {'unique_symbols': unique_symbols,
                                    'params': params}

    def __setstate__(self, state):
        """Unpickle a Parameters instance.

        Parameters
        ----------
        state : dict
            state['unique_symbols'] is a dictionary containing symbols
            that need to be injected into `_asteval.symtable`.
            state['params'] is a list of Parameter instances to be added.

        """
        # first update the Interpreter symbol table. This needs to be done
        # first because Parameter's early in the list may depend on later
        # Parameter's. This leads to problems because add_many eventually leads
        # to a Parameter value being retrieved with _getval, which, if the
        # dependent value hasn't already been added to the symtable, leads to
        # an Error. Another way of doing this would be to remove all the expr
        # from the Parameter instances before they get added, then to restore
        # them.

        symtab = self._asteval.symtable
        for key, val in state['unique_symbols'].items():
            if key not in symtab:
                symtab[key] = val

        # then add all the parameters
        self.add_many(*state['params'])

    def __repr__(self):
        """__repr__ from OrderedDict."""
        if not self:
            return f'{self.__class__.__name__}()'
        return f'{self.__class__.__name__}({list(self.items())!r})'

    def eval(self, expr):
        """Evaluate a statement using the `asteval` Interpreter.

        Parameters
        ----------
        expr : str
            An expression containing parameter names and other symbols
            recognizable by the `asteval` Interpreter.

        Returns
        -------
        float
            The result of evaluating the expression.

        """
        return self._asteval.eval(expr)

    def update_constraints(self):
        """Update all constrained parameters.

        This method ensures that dependencies are evaluated as needed.

        """
        requires_update = {name for name, par in self.items() if par._expr is
                           not None}
        updated_tracker = set(requires_update)

        def _update_param(name):
            """Update a parameter value, including setting bounds.

            For a constrained parameter (one with an `expr` defined), this
            first updates (recursively) all parameters on which the
            parameter depends (using the 'deps' field).

            """
            par = self.__getitem__(name)
            if par._expr_eval is None:
                par._expr_eval = self._asteval
            for dep in par._expr_deps:
                if dep in updated_tracker:
                    _update_param(dep)
            self._asteval.symtable[name] = float(par.value)
            updated_tracker.discard(name)

        for name in requires_update:
            _update_param(name)

    def pretty_repr(self, oneline=False):
        """Return a pretty representation of a Parameters class.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation (default
            is False).

        Returns
        -------
        s: str
            Parameters representation.

        """
        if oneline:
            return self.__repr__()
        s = "Parameters({\n"
        for key in self.keys():
            s += f"    '{key}': {self[key]}, \n"
        s += "    })\n"
        return s

    def pretty_print(self, oneline=False, colwidth=8, precision=4, fmt='g',
                     columns=['value', 'min', 'max', 'stderr', 'vary', 'expr',
                              'brute_step']):
        """Pretty-print of parameters data.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation [False]
        colwidth : int, optional
            Column width for all columns specified in `columns` [8]
        precision : int, optional
            Number of digits to be printed after floating point [4]
        fmt : {'g', 'e', 'f'}, optional
            Single-character numeric formatter. Valid values are: `'g'`
            floating point and exponential (default), `'e'` exponential,
            or `'f'` floating point.
        columns : :obj:`list` of :obj:`str`, optional
            List of :class:`Parameter` attribute names to print (default
            is to show all attributes).

        """
        if oneline:
            print(self.pretty_repr(oneline=oneline))
            return

        name_len = max(len(s) for s in self)
        allcols = ['name'] + columns
        title = '{:{name_len}} ' + len(columns) * ' {:>{n}}'
        print(title.format(*allcols, name_len=name_len, n=colwidth).title())
        numstyle = '{%s:>{n}.{p}{f}}'  # format for numeric columns
        otherstyles = dict(name='{name:<{name_len}} ', stderr='{stderr!s:>{n}}',
                           vary='{vary!s:>{n}}', expr='{expr!s:>{n}}',
                           brute_step='{brute_step!s:>{n}}')
        line = ' '.join(otherstyles.get(k, numstyle % k) for k in allcols)
        for name, values in sorted(self.items()):
            pvalues = {k: getattr(values, k) for k in columns}
            pvalues['name'] = name
            # stderr is a special case: it is either numeric or None (i.e. str)
            if 'stderr' in columns and pvalues['stderr'] is not None:
                pvalues['stderr'] = (numstyle % '').format(
                    pvalues['stderr'], n=colwidth, p=precision, f=fmt)
            elif 'brute_step' in columns and pvalues['brute_step'] is not None:
                pvalues['brute_step'] = (numstyle % '').format(
                    pvalues['brute_step'], n=colwidth, p=precision, f=fmt)
            print(line.format(name_len=name_len, n=colwidth, p=precision,
                              f=fmt, **pvalues))

    def _repr_html_(self):
        """Return a HTML representation of parameters data."""
        return params_html_table(self)

    def set(self, **kws):
        """Set Parameter values and other attributes.

        Parameters
        ----------
        **kws : optional
            Parameter names and initial values or dictionaries of
                 values and attributes.

        Returns
        -------
        None

        Notes
        -----
        1. keyword arguments will be used to create parameter names.
        2. values can either be numbers (floats or integers) to set the
           parameter value, or can be dictionaries with any of the following
           keywords: ``value``, ``vary``, ``min``, ``max``, ``expr``,
           ``brute_step``, or ``is_init_value`` to set those parameter attributes.
        3. for each parameter,  ``is_init_value`` controls whether to set
           ``init_value`` when setting ``value``, and defaults to True.

        Examples
        --------
        >>> params = Parameters()
        >>> params.add('xvar', value=0.50, min=0, max=1)
        >>> params.add('yvar', expr='1.0 - xvar')
        >>> params.set(xvar=0.80, zvar={'value':3, 'min':0})

        """
        for name, val in kws.items():
            if name not in self:
                self.__setitem__(name, Parameter(value=-inf, name=name,
                                                 vary=True, min=-inf, max=inf,
                                                 expr=None, brute_step=None))
            par = self.__getitem__(name)
            if isinstance(val, (float, int)):
                val = {'value': val}
            if 'is_init_value' not in val:
                val['is_init_value'] = True
            par.set(**val)

    def add(self, name, value=None, vary=None, min=-inf, max=inf, expr=None,
            brute_step=None, user_data=None):
        """Add a Parameter.

        Parameters
        ----------
        name : str or Parameter
            If ``name`` refers to a Parameter object it will be added directly
            to the Parameters instance, otherwise a new Parameter object with name
            ``string`` is created before adding it. In both cases, ``name`` must
            match ``[a-z_][a-z0-9_]*`` and cannot be a Python reserved word.
        value : float, optional
            Numerical Parameter value, typically the *initial value*.
        vary : bool, optional
            Whether the Parameter is varied during a fit (default is True).
        min : float, optional
            Lower bound for value (default is ``-numpy.inf``, no lower
            bound).
        max : float, optional
            Upper bound for value (default is ``numpy.inf``, no upper
            bound).
        expr : str, optional
            Mathematical expression used to constrain the value during the
            fit (default is None).
        brute_step : float, optional
            Step size for grid points in the `brute` method (default is
            None).

        Examples
        --------
        >>> params = Parameters()
        >>> params.add('xvar', value=0.50, min=0, max=1)
        >>> params.add('yvar', expr='1.0 - xvar')

        which is equivalent to:

        >>> params = Parameters()
        >>> params['xvar'] = Parameter(name='xvar', value=0.50, min=0, max=1)
        >>> params['yvar'] = Parameter(name='yvar', expr='1.0 - xvar')

        """
        if isinstance(name, Parameter):
            if name.value is None:
                name.value = float(value)
            if name.expr is None:
                name.expr = expr
            if name.min in (None, -inf):
                name.min = min
            if name.max in (None, inf):
                name.max = max
            if name.brute_step is None:
                name.brute_step = brute_step
            if name.user_data is None:
                name.user_data = user_data
            if vary is not None:
                name.vary = vary
            self.__setitem__(name.name, name)
        elif isinstance(value, Parameter):
            value = deepcopy(value)
            value.name = name
            if value.expr is None:
                value.expr = expr
            if value.min in (None, -inf):
                value.min = min
            if value.max in (None, inf):
                value.max = max
            if value.brute_step is None:
                value.brute_step = brute_step
            if value.user_data is None:
                value.user_data = user_data
            if vary is not None:
                value.vary = vary
            self.__setitem__(name, value)
        else:
            if vary is None:
                vary = True
            self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                             min=min, max=max, expr=expr,
                                             brute_step=brute_step, user_data=user_data))
        if len(self._asteval.error) > 0:
            err = self._asteval.error[0]
            raise err.exc(err.msg)

    def add_many(self, *parlist):
        """Add many parameters, using a sequence of tuples.

        Parameters
        ----------
        *parlist : :obj:`sequence` of :obj:`tuple` or Parameter
            A sequence of tuples, or a sequence of `Parameter` instances.
            If it is a sequence of tuples, then each tuple must contain at
            least a `name`. The order in each tuple must be
            ``(name, value, vary, min, max, expr, brute_step)``.

        Examples
        --------
        >>>  params = Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        >>> params.add_many(('amp', 10, True, None, None, None, None),
        ...                 ('cen', 4, True, 0.0, None, None, None),
        ...                 ('wid', 1, False, None, None, None, None),
        ...                 ('frac', 0.5))
        # add a sequence of Parameters
        >>> f = Parameter('par_f', 100)
        >>> g = Parameter('par_g', 2.)
        >>> params.add_many(f, g)

        """
        __params = []
        for par in parlist:
            if not isinstance(par, Parameter):
                par = Parameter(*par)
            __params.append(par)
            par._delay_asteval = True
            self.__setitem__(par.name, par)

        for para in __params:
            para._delay_asteval = False

    def valuesdict(self):
        """Return an ordered dictionary of parameter values.

        Returns
        -------
        dict
            A dictionary of :attr:`name`::attr:`value` pairs for each
            Parameter.

        """
        return {p.name: float(p.value) for p in self.values()}

    def create_uvars(self, covar=None):
        """Return a dict of uncertainties ufloats from the current Parameter
        values and stderr, and an optionally-supplied covariance matrix.
        Uncertainties in Parameters with constraint expressions will be
        calculated, propagating uncertaintes (and including correlations)

        Parameters
        ----------
        covar : optional
              Nvar x Nvar covariance matrix from fit

        Returns
        -------
        dict with keys of Parameter names and values of uncertainties.ufloats.

        Notes
        -----
        1.  if covar is provide, it must correspond to the existing *variable*
            Parameters.  If covar is given, the returned uncertainties ufloats
            will take the correlations into account when combining values.
        2.  See the uncertainties package documentation
            (https://pythonhosted.org/uncertainties) for more details.
        """
        uvars = {}
        has_expr = False
        vnames, vbest, vindex = [], [], -1
        savevals = self.valuesdict()
        for par in self.values():
            has_expr = has_expr or par.expr is not None
            if par.vary:
                vindex += 1
                vnames.append(par.name)
                vbest.append(float(par.value))
                if getattr(par, 'stderr', None) is None and covar is not None:
                    par.stderr = sqrt(covar[vindex, vindex])
            stderr = getattr(par, 'stderr', 0.0)
            if stderr is None:
                stderr = 0.00
            if stderr < tiny*max(tiny, abs(par.value)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    uvars[par.name] = ufloat(par.value, stderr)
            else:
                uvars[par.name] = ufloat(par.value, stderr)

        corr_uvars = None
        if covar is not None:
            try:
                corr_uvars = correlated_values(vbest, covar)
                for name, cuv in zip(vnames, corr_uvars):
                    uvars[name] = cuv
            except (LinAlgError, ValueError):
                pass

        if has_expr and corr_uvars is not None:
            # for uncertainties on constrained parameters, use the calculated
            # correlated values, evaluate the uncertainties on the constrained
            # parameters and reset the Parameters to best-fit value
            wrap_ueval = uwrap(asteval_with_uncertainties)
            for par in self.values():
                if getattr(par, '_expr_ast', None) is not None:
                    try:
                        uval = wrap_ueval(*corr_uvars, obj=par,
                                          pars=self, names=vnames)
                        par.stderr = uval.std_dev
                        uvars[par.name] = uval
                    except Exception:
                        par.stderr = 0
        # restore all param values to saved best values
        for parname, param in self.items():
            param.value = savevals[parname]
        return uvars

    def dumps(self, **kws):
        """Represent Parameters as a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.dumps`.

        Returns
        -------
        str
            JSON string representation of Parameters.

        See Also
        --------
        dump, loads, load, json.dumps

        """
        params = [p.__getstate__() for p in self.values()]
        unique_symbols = {key: encode4js(deepcopy(self._asteval.symtable[key]))
                          for key in self._asteval.user_defined_symbols()}
        return json.dumps({'unique_symbols': unique_symbols,
                           'params': params}, **kws)

    def loads(self, s, **kws):
        """Load Parameters from a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.loads`.

        Returns
        -------
        Parameters
            Updated Parameters from the JSON string.

        Notes
        -----
        Current Parameters will be cleared before loading the data from
        the JSON string.

        See Also
        --------
        dump, dumps, load, json.loads

        """
        self.clear()

        tmp = json.loads(s, **kws)
        unique_symbols = {key: decode4js(tmp['unique_symbols'][key]) for key
                          in tmp['unique_symbols']}

        state = {'unique_symbols': unique_symbols, 'params': []}
        for parstate in tmp['params']:
            _par = Parameter(name='')
            _par.__setstate__(parstate)
            state['params'].append(_par)
        self.__setstate__(state)
        return self

    def dump(self, fp, **kws):
        """Write JSON representation of Parameters to a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and `.write()`-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `dumps`.

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

    def load(self, fp, **kws):
        """Load JSON representation of Parameters from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and `.read()`-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `loads`.

        Returns
        -------
        Parameters
            Updated Parameters loaded from `fp`.

        See Also
        --------
        dump, loads, json.load

        """
        return self.loads(fp.read(), **kws)


class Parameter:
    """A Parameter is an object that can be varied in a fit.

    It is a central component of lmfit, and all minimization and modeling
    methods use Parameter objects.

    A Parameter has a `name` attribute, and a scalar floating point
    `value`. It also has a `vary` attribute that describes whether the
    value should be varied during the minimization. Finite bounds can be
    placed on the Parameter's value by setting its `min` and/or `max`
    attributes. A Parameter can also have its value determined by a
    mathematical expression of other Parameter values held in the `expr`
    attribute. Additional attributes include `brute_step` used as the step
    size in a brute-force minimization, and `user_data` reserved
    exclusively for user's need.

    After a minimization, a Parameter may also gain other attributes,
    including `stderr` holding the estimated standard error in the
    Parameter's value, and `correl`, a dictionary of correlation values
    with other Parameters used in the minimization.

    """

    def __init__(self, name, value=None, vary=True, min=-inf, max=inf,
                 expr=None, brute_step=None, user_data=None):
        """
        Parameters
        ----------
        name : str
            Name of the Parameter.
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is varied during a fit (default is True).
        min : float, optional
            Lower bound for value (default is ``-numpy.inf``, no lower
            bound).
        max : float, optional
            Upper bound for value (default is ``numpy.inf``, no upper
            bound).
        expr : str, optional
            Mathematical expression used to constrain the value during the
            fit (default is None).
        brute_step : float, optional
            Step size for grid points in the `brute` method (default is
            None).
        user_data : optional
            User-definable extra attribute used for a Parameter (default
            is None).

        Attributes
        ----------
        stderr : float
            The estimated standard error for the best-fit value.
        correl : dict
            A dictionary of the correlation with the other fitted
            Parameters of the form::

            {'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}

        """
        self.name = name
        self.user_data = user_data
        self.init_value = value
        self.min = min
        self.max = max
        self.brute_step = brute_step
        self._vary = vary
        self._expr = expr
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self._delay_asteval = False
        self.stderr = None
        self.correl = None
        self.from_internal = lambda val: float(val)
        self._val = value
        self._init_bounds()

    def set(self, value=None, vary=None, min=None, max=None, expr=None,
            brute_step=None, is_init_value=True):
        """Set or update Parameter attributes.

        Parameters
        ----------
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is varied during a fit.
        min : float, optional
            Lower bound for value. To remove a lower bound you must use
            ``-numpy.inf``.
        max : float, optional
            Upper bound for value. To remove an upper bound you must use
            ``numpy.inf``.
        expr : str, optional
            Mathematical expression used to constrain the value during the
            fit. To remove a constraint you must supply an empty string.
        brute_step : float, optional
            Step size for grid points in the `brute` method. To remove the
            step size you must use ``0``.
        is_init_value: bool, optional
            Whether to set value as `init_value`, when setting value.

        Notes
        -----
        Each argument to `set()` has a default value of None, which will
        leave the current value for the attribute unchanged. Thus, to lift
        a lower or upper bound, passing in None will not work. Instead,
        you must set these to ``-numpy.inf`` or ``numpy.inf``, as with::

            par.set(min=None)        # leaves lower bound unchanged
            par.set(min=-numpy.inf)  # removes lower bound

        Similarly, to clear an expression, pass a blank string, (not
        None!) as with::

            par.set(expr=None)  # leaves expression unchanged
            par.set(expr='')    # removes expression

        Explicitly setting a value or setting ``vary=True`` will also
        clear the expression.

        Finally, to clear the brute_step size, pass ``0``, not None::

            par.set(brute_step=None)  # leaves brute_step unchanged
            par.set(brute_step=0)     # removes brute_step

        """
        if vary is not None:
            self._vary = vary
            if vary:
                self.__set_expression('')

        if min is not None:
            self.min = min

        if max is not None:
            self.max = max

        # need to set this after min and max, so that it will use new
        # bounds in the setter for value
        if value is not None:
            is_init_value = is_init_value or self.value in (None, -inf, inf)
            self.value = value
            if is_init_value:
                self.init_value = value
            self.__set_expression("")

        if expr is not None:
            self.__set_expression(expr)

        if brute_step is not None:
            if brute_step == 0.0:
                self.brute_step = None
            else:
                self.brute_step = brute_step

    def _init_bounds(self):
        """Make sure initial bounds are self-consistent."""
        # _val is None means - infinity.
        if self.max is None:
            self.max = inf
        if self.min is None:
            self.min = -inf
        if self._val is None:
            self._val = -inf
        if self.min > self.max:
            self.min, self.max = self.max, self.min
        if isclose(self.min, self.max, atol=1e-13, rtol=1e-13):
            raise ValueError(f"Parameter '{self.name}' has min == max")
        if self._val > self.max:
            self._val = self.max
        if self._val < self.min:
            self._val = self.min
        self.setup_bounds()

    def __getstate__(self):
        """Get state for pickle."""
        return (self.name, self.value, self._vary, self.expr, self.min,
                self.max, self.brute_step, self.stderr, self.correl,
                self.init_value, self.user_data)

    def __setstate__(self, state):
        """Set state for pickle."""
        (self.name, _value, self._vary, self.expr, self.min, self.max,
         self.brute_step, self.stderr, self.correl, self.init_value,
         self.user_data) = state
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self._delay_asteval = False
        self._val = _value
        self._init_bounds()
        self.value = _value

    def __repr__(self):
        """Return printable representation of a Parameter object."""
        s = []
        sval = f"value={repr(self._getval())}"
        if not self._vary and self._expr is None:
            sval += " (fixed)"
        elif self.stderr is not None:
            sval += f" +/- {self.stderr:.3g}"
        s.append(sval)
        s.append(f"bounds=[{repr(self.min)}:{repr(self.max)}]")
        if self._expr is not None:
            s.append(f"expr='{self.expr}'")
        if self.brute_step is not None:
            s.append(f"brute_step={self.brute_step}")
        return f"<Parameter '{self.name}', {', '.join(s)}>"

    def setup_bounds(self):
        """Set up Minuit-style internal/external parameter transformation
        of min/max bounds.

        As a side-effect, this also defines the self.from_internal method
        used to re-calculate self.value from the internal value, applying
        the inverse Minuit-style transformation. This method should be
        called prior to passing a Parameter to the user-defined objective
        function.

        This code borrows heavily from JJ Helmus' leastsqbound.py

        Returns
        -------
        _val : float
            The internal value for parameter from `self.value` (which holds
            the external, user-expected value). This internal value should
            actually be used in a fit.

        """
        if self.min is None:
            self.min = -inf
        if self.max is None:
            self.max = inf
        if self.min == -inf and self.max == inf:
            self.from_internal = lambda val: float(val)
            _val = self._val
        elif self.max == inf:
            self.from_internal = lambda val: self.min - 1.0 + sqrt(val*val + 1)
            _val = sqrt((self._val - self.min + 1.0)**2 - 1)
        elif self.min == -inf:
            self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
            _val = sqrt((self.max - self._val + 1.0)**2 - 1)
        else:
            self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                                 (self.max - self.min) / 2.0
            _val = arcsin(2*(self._val - self.min)/(self.max - self.min) - 1)
        if abs(_val) < tiny:
            _val = 0.0
        return float(_val)

    def scale_gradient(self, val):
        """Return scaling factor for gradient.

        Parameters
        ----------
        val : float
            Numerical Parameter value.

        Returns
        -------
        float
            Scaling factor for gradient the according to Minuit-style
            transformation.

        """
        if self.min == -inf and self.max == inf:
            return 1.0
        if self.max == inf:
            return val / sqrt(val*val + 1)
        if self.min == -inf:
            return -val / sqrt(val*val + 1)
        return cos(val) * (self.max - self.min) / 2.0

    def _getval(self):
        """Get value, with bounds applied."""
        # Note assignment to self._val has been changed to self.value
        # The self.value property setter makes sure that the
        # _expr_eval.symtable is kept up-to-date.
        # If you just assign to self._val then _expr_eval.symtable[self.name]
        # becomes stale if parameter.expr is not None.
        if self._expr is not None:
            if self._expr_ast is None:
                self.__set_expression(self._expr)
            if self._expr_eval is not None and not self._delay_asteval:
                self.value = self._expr_eval(self._expr_ast)
                check_ast_errors(self._expr_eval)
        return self._val

    @property
    def value(self):
        """Return the numerical Parameter value, with bounds applied."""
        return self._getval()

    @value.setter
    def value(self, val):
        """Set the numerical Parameter value."""
        self._val = val
        if self._val is not None:
            if self._val > self.max:
                self._val = self.max
            elif self._val < self.min:
                self._val = self.min
        if not hasattr(self, '_expr_eval'):
            self._expr_eval = None
        if self._expr_eval is not None:
            self._expr_eval.symtable[self.name] = self._val

    @property
    def vary(self):
        """Return whether the parameter is variable"""
        return self._vary

    @vary.setter
    def vary(self, val):
        """Set whether a parameter is varied"""
        self._vary = val
        if val:
            self.__set_expression('')

    @property
    def expr(self):
        """Return the mathematical expression used to constrain the value in fit."""
        return self._expr

    @expr.setter
    def expr(self, val):
        """Set the mathematical expression used to constrain the value in fit.

        To remove a constraint you must supply an empty string.

        """
        self.__set_expression(val)

    def __set_expression(self, val):
        if val == '':
            val = None
        self._expr = val
        if val is not None:
            self._vary = False
        if not hasattr(self, '_expr_eval'):
            self._expr_eval = None
        if val is None:
            self._expr_ast = None
        if val is not None and self._expr_eval is not None:
            self._expr_eval.error = []
            self._expr_eval.error_msg = None
            self._expr_ast = self._expr_eval.parse(val)
            check_ast_errors(self._expr_eval)
            self._expr_deps = get_ast_names(self._expr_ast)

    def __array__(self):
        """array"""
        return array(float(self._getval()))

    def __str__(self):
        """string"""
        return self.__repr__()

    def __abs__(self):
        """abs"""
        return abs(self._getval())

    def __neg__(self):
        """neg"""
        return -self._getval()

    def __pos__(self):
        """positive"""
        return +self._getval()

    def __bool__(self):
        """bool"""
        return self._getval() != 0

    def __int__(self):
        """int"""
        return int(self._getval())

    def __float__(self):
        """float"""
        return float(self._getval())

    def __trunc__(self):
        """trunc"""
        return self._getval().__trunc__()

    def __add__(self, other):
        """+"""
        return self._getval() + other

    def __sub__(self, other):
        """-"""
        return self._getval() - other

    def __truediv__(self, other):
        """/"""
        return self._getval() / other

    def __floordiv__(self, other):
        """//"""
        return self._getval() // other

    def __divmod__(self, other):
        """divmod"""
        return divmod(self._getval(), other)

    def __mod__(self, other):
        """%"""
        return self._getval() % other

    def __mul__(self, other):
        """*"""
        return self._getval() * other

    def __pow__(self, other):
        """**"""
        return self._getval() ** other

    def __gt__(self, other):
        """>"""
        return self._getval() > other

    def __ge__(self, other):
        """>="""
        return self._getval() >= other

    def __le__(self, other):
        """<="""
        return self._getval() <= other

    def __lt__(self, other):
        """<"""
        return self._getval() < other

    def __eq__(self, other):
        """=="""
        return self._getval() == other

    def __ne__(self, other):
        """!="""
        return self._getval() != other

    def __radd__(self, other):
        """+ (right)"""
        return other + self._getval()

    def __rtruediv__(self, other):
        """/ (right)"""
        return other / self._getval()

    def __rdivmod__(self, other):
        """divmod (right)"""
        return divmod(other, self._getval())

    def __rfloordiv__(self, other):
        """// (right)"""
        return other // self._getval()

    def __rmod__(self, other):
        """% (right)"""
        return other % self._getval()

    def __rmul__(self, other):
        """* (right)"""
        return other * self._getval()

    def __rpow__(self, other):
        """** (right)"""
        return other ** self._getval()

    def __rsub__(self, other):
        """- (right)"""
        return other - self._getval()


def create_params(**kws):
    """Create lmfit.Parameters instance and set initial values and attributes.

    Parameters
    ----------
    **kws
        keywords are parameter names, value are dictionaries of Parameter
        values and attributes.

    Returns
    -------
    Parameters instance

    Notes
    -----
    1. keyword arguments will be used to create parameter names.
    2. values can either be numbers (floats or integers) to set the parameter
       value, or can be dictionaries with any of the following keywords:
       ``value``, ``vary``, ``min``, ``max``, ``expr``, ``brute_step``, or
       ``is_init_value`` to set those parameter attributes.
    3. for each parameter,  ``is_init_value`` controls whether to set
       ``init_value`` when setting ``value``, and defaults to True.

    Examples
    --------
    >>> params = create_params(amplitude=2, center=200,
                               sigma={'value': 3, 'min':0},
                               fwhm={'expr': '2.0*sigma'})
    """
    params = Parameters()
    params.set(**kws)
    return params
