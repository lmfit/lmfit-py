"""Parameter class."""
from __future__ import division

from collections import OrderedDict
from copy import deepcopy
import json
import importlib

from asteval import Interpreter, get_ast_names, valid_symbol_name
from numpy import arcsin, array, cos, inf, isclose, nan, sin, sqrt
import scipy.special
import uncertainties

from .jsonutils import decode4js, encode4js
from .printfuncs import params_html_table

SCIPY_FUNCTIONS = {'gamfcn': scipy.special.gamma}
for name in ('erf', 'erfc', 'wofz'):
    SCIPY_FUNCTIONS[name] = getattr(scipy.special, name)


def check_ast_errors(expr_eval):
    """Check for errors derived from asteval."""
    if len(expr_eval.error) > 0:
        expr_eval.raise_exception(None)


class Parameters(OrderedDict):
    """An ordered dictionary of all the Parameter objects required to
    specify a fit model. All minimization and Model fitting routines in
    lmfit will use exactly one Parameters object, typically given as the
    first argument to the objective function.

    All keys of a Parameters() instance must be strings and valid Python
    symbol names, so that the name must match ``[a-z_][a-z0-9_]*`` and
    cannot be a Python reserved word.

    All values of a Parameters() instance must be Parameter objects.

    A Parameters() instance includes an asteval interpreter used for
    evaluation of constrained Parameters.

    Parameters() support copying and pickling, and have methods to convert
    to and from serializations using json strings.

    """

    def __init__(self, asteval=None, usersyms=None, *args, **kwds):
        """
        Arguments
        ---------
        asteval : :class:`asteval.Interpreter`, optional
            Instance of the asteval Interpreter to use for constraint
            expressions. If None, a new interpreter will be created.
            Warning: *deprecated, use usersyms if possible*
        usersyms : dictionary of symbols to add to the
            :class:`asteval.Interpreter`.
        *args : optional
            Arguments.
        **kwds : optional
            Keyword arguments.

        """
        super(Parameters, self).__init__(self)

        self._asteval = asteval
        if self._asteval is None:
            self._asteval = Interpreter()

        _syms = {}
        _syms.update(SCIPY_FUNCTIONS)
        if usersyms is not None:
            _syms.update(usersyms)
        for key, val in _syms.items():
            self._asteval.symtable[key] = val

        self.update(*args, **kwds)

    def copy(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def __copy__(self):
        """Parameters.copy() should always be a deepcopy."""
        return self.__deepcopy__(None)

    def __deepcopy__(self, memo):
        """Implementation of Parameters.deepcopy().

        The method needs to make sure that asteval is available and that all
        individual Parameter objects are copied.

        """
        _pars = Parameters(asteval=None)

        # find the symbols that were added by users, not during construction
        unique_symbols = {key: self._asteval.symtable[key]
                          for key in self._asteval.user_defined_symbols()}
        _pars._asteval.symtable.update(unique_symbols)

        # we're just about to add a lot of Parameter objects to the newly
        parameter_list = []
        for key, par in self.items():
            if isinstance(par, Parameter):
                param = Parameter(name=par.name,
                                  value=par.value,
                                  min=par.min,
                                  max=par.max)
                param.vary = par.vary
                param.brute_step = par.brute_step
                param.stderr = par.stderr
                param.correl = par.correl
                param.init_value = par.init_value
                param.expr = par.expr
                param.user_data = par.user_data
                parameter_list.append(param)

        _pars.add_many(*parameter_list)

        return _pars

    def __setitem__(self, key, par):
        """TODO: add magic method docstring."""
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if par is not None and not isinstance(par, Parameter):
            raise ValueError("'%s' is not a Parameter" % par)
        OrderedDict.__setitem__(self, key, par)
        par.name = key
        par._expr_eval = self._asteval
        self._asteval.symtable[key] = par.value

    def __add__(self, other):
        """Add Parameters objects."""
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        out = deepcopy(self)
        params = other.values()
        out.add_many(*params)
        return out

    def __iadd__(self, other):
        """Add/assign Parameters objects."""
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        params = other.values()
        self.add_many(*params)
        return self

    def __array__(self):
        """Convert Parameters to array."""
        return array([float(k) for k in self.values()])

    def __reduce__(self):
        """Reduce Parameters instance such that it can be pickled."""
        # make a list of all the parameters
        params = [self[k] for k in self]

        # find the symbols from _asteval.symtable, that need to be remembered.
        sym_unique = self._asteval.user_defined_symbols()
        unique_symbols = {key: deepcopy(self._asteval.symtable[key])
                          for key in sym_unique}

        return self.__class__, (), {'unique_symbols': unique_symbols,
                                    'params': params}

    def __setstate__(self, state):
        """Unpickle a Parameters instance.

        Parameters
        ----------
        state : dict
            state['unique_symbols'] is a dictionary containing symbols that
            need to be injected into _asteval.symtable
            state['params'] is a list of Parameter instances to be added

        """
        # first update the Interpreter symbol table. This needs to be done
        # first because Parameter's early in the list may depend on later
        # Parameter's. This leads to problems because add_many eventually leads
        # to a Parameter value being retrieved with _getval, which, if the
        # dependent value hasn't already been added to the symtable, leads to
        # an Error. Another way of doing this would be to remove all the expr
        # from the Parameter instances before they get added, then to restore
        # them.
        # self._asteval.symtable.update(state['unique_symbols'])

        symtab = self._asteval.symtable
        for key, val in state['unique_symbols'].items():
            if key not in symtab:
                if isinstance(val, dict):
                    value = val.get('__name__', None)
                    symname = val.get('__name__', None)
                    importer = val.get('importer', None)
                    if value is None and symname is not None and importer is not None:
                        _mod = importlib.import_module(importer)
                        value = getattr(_mod, symname, None)
                    if value is not None:
                        symtab[key] = value
                else:
                    symtab[key] = val
        # then add all the parameters
        self.add_many(*state['params'])

    def eval(self, expr):
        """Evaluate a statement using the asteval Interpreter.

        Parameters
        ----------
        expr : string
            An expression containing parameter names and other symbols
            recognizable by the asteval Interpreter.

        Returns
        -------
           The result of the expression.

        """
        return self._asteval.eval(expr)

    def update_constraints(self):
        """Update all constrained parameters, checking that dependencies are
        evaluated as needed."""
        requires_update = {name for name, par in self.items() if par._expr is
                           not None}
        updated_tracker = set(requires_update)

        def _update_param(name):
            """Update a parameter value, including setting bounds.

            For a constrained parameter (one with an `expr` defined),
            this first updates (recursively) all parameters on which the
            parameter depends (using the 'deps' field).

            """
            par = self.__getitem__(name)
            if par._expr_eval is None:
                par._expr_eval = self._asteval
            for dep in par._expr_deps:
                if dep in updated_tracker:
                    _update_param(dep)
            self._asteval.symtable[name] = par.value
            updated_tracker.discard(name)

        for name in requires_update:
            _update_param(name)

    def pretty_repr(self, oneline=False):
        """Return a pretty representation of a Parameters class.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation (default is
            False).

        Returns
        -------
        s: str
           Parameters representation.

        """
        if oneline:
            return super(Parameters, self).__repr__()
        s = "Parameters({\n"
        for key in self.keys():
            s += "    '%s': %s, \n" % (key, self[key])
        s += "    })\n"
        return s

    def pretty_print(self, oneline=False, colwidth=8, precision=4, fmt='g',
                     columns=['value', 'min', 'max', 'stderr', 'vary', 'expr',
                              'brute_step']):
        """Pretty-print of parameters data.

        Parameters
        ----------
        oneline : bool, optional
            If True prints a one-line parameters representation (default is
            False).
        colwidth : int, optional
            Column width for all columns specified in :attr:`columns`.
        precision : int, optional
            Number of digits to be printed after floating point.
        fmt : {'g', 'e', 'f'}, optional
            Single-character numeric formatter. Valid values are: 'f' floating
            point, 'g' floating point and exponential, or 'e' exponential.
        columns : :obj:`list` of :obj:`str`, optional
            List of :class:`Parameter` attribute names to print.

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
        line = ' '.join([otherstyles.get(k, numstyle % k) for k in allcols])
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
        """Returns a HTML representation of parameters data."""
        return params_html_table(self)

    def add(self, name, value=None, vary=True, min=-inf, max=inf, expr=None,
            brute_step=None):
        """Add a Parameter.

        Parameters
        ----------
        name : str
            Name of parameter.  Must match ``[a-z_][a-z0-9_]*`` and cannot be
            a Python reserved word.
        value : float, optional
            Numerical Parameter value, typically the *initial value*.
        vary : bool, optional
            Whether the Parameter is varied during a fit (default is True).
        min : float, optional
            Lower bound for value (default is `-numpy.inf`, no lower bound).
        max : float, optional
            Upper bound for value (default is `numpy.inf`, no upper bound).
        expr : str, optional
            Mathematical expression used to constrain the value during the fit.
        brute_step : float, optional
            Step size for grid points in the `brute` method.

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
            self.__setitem__(name.name, name)
        else:
            self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                             min=min, max=max, expr=expr,
                                             brute_step=brute_step))

    def add_many(self, *parlist):
        """Add many parameters, using a sequence of tuples.

        Parameters
        ----------
        parlist : :obj:`sequence` of :obj:`tuple` or :class:`Parameter`
            A sequence of tuples, or a sequence of `Parameter` instances. If
            it is a sequence of tuples, then each tuple must contain at least
            the name. The order in each tuple must be `(name, value, vary,
            min, max, expr, brute_step)`.

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
        OrderedDict
           An ordered dictionary of :attr:`name`::attr:`value` pairs for each
           Parameter.

        """
        return OrderedDict((p.name, p.value) for p in self.values())

    def dumps(self, **kws):
        """Represent Parameters as a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.dumps()`.

        Returns
        -------
        str
           JSON string representation of Parameters.

        See Also
        --------
        dump(), loads(), load(), json.dumps()

        """
        params = [p.__getstate__() for p in self.values()]
        sym_unique = self._asteval.user_defined_symbols()
        unique_symbols = {key: encode4js(deepcopy(self._asteval.symtable[key]))
                          for key in sym_unique}
        return json.dumps({'unique_symbols': unique_symbols,
                           'params': params}, **kws)

    def loads(self, s, **kws):
        """Load Parameters from a JSON string.

        Parameters
        ----------
        **kws : optional
            Keyword arguments that are passed to `json.loads()`.

        Returns
        -------
        :class:`Parameters`
           Updated Parameters from the JSON string.

        Notes
        -----
        Current Parameters will be cleared before loading the data from the
        JSON string.

        See Also
        --------
        dump(), dumps(), load(), json.loads()

        """
        self.clear()

        tmp = decode4js(json.loads(s, **kws))
        state = {'unique_symbols': tmp['unique_symbols'],
                 'params': []}
        for parstate in tmp['params']:
            _par = Parameter()
            _par.__setstate__(parstate)
            state['params'].append(_par)
        self.__setstate__(state)
        return self

    def dump(self, fp, **kws):
        """Write JSON representation of Parameters to a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and ``.write()``-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `dumps()`.

        Returns
        -------
        None or int
            Return value from `fp.write()`. None for Python 2.7 and the
            number of characters written in Python 3.

        See Also
        --------
        dump(), load(), json.dump()

        """
        return fp.write(self.dumps(**kws))

    def load(self, fp, **kws):
        """Load JSON representation of Parameters from a file-like object.

        Parameters
        ----------
        fp : file-like object
            An open and ``.read()``-supporting file-like object.
        **kws : optional
            Keyword arguments that are passed to `loads()`.

        Returns
        -------
        :class:`Parameters`
           Updated Parameters loaded from `fp`.

        See Also
        --------
        dump(), loads(), json.load()

        """
        return self.loads(fp.read(), **kws)


class Parameter(object):
    """A Parameter is an object that can be varied in a fit, or one of the
    controlling variables in a model. It is a central component of lmfit,
    and all minimization and modeling methods use Parameter objects.

    A Parameter has a `name` attribute, and a scalar floating point
    `value`.  It also has a `vary` attribute that describes whether the
    value should be varied during the minimization.  Finite bounds can be
    placed on the Parameter's value by setting its `min` and/or `max`
    attributes.  A Parameter can also have its value determined by a
    mathematical expression of other Parameter values held in the `expr`
    attrribute.  Additional attributes include `brute_step` used as the step
    size in a brute-force minimization, and `user_data` reserved
    exclusively for user's need.

    After a minimization, a Parameter may also gain other attributes,
    including `stderr` holding the estimated standard error in the
    Parameter's value, and `correl`, a dictionary of correlation values
    with other Parameters used in the minimization.

    """

    def __init__(self, name=None, value=None, vary=True, min=-inf, max=inf,
                 expr=None, brute_step=None, user_data=None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the Parameter.
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is varied during a fit (default is True).
        min : float, optional
            Lower bound for value (default is `-numpy.inf`, no lower bound).
        max : float, optional
            Upper bound for value (default is `numpy.inf`, no upper bound).
        expr : str, optional
            Mathematical expression used to constrain the value during the fit.
        brute_step : float, optional
            Step size for grid points in the `brute` method.
        user_data : optional
            User-definable extra attribute used for a Parameter.

        Attributes
        ----------
        stderr : float
            The estimated standard error for the best-fit value.
        correl : dict
            A dictionary of the correlation with the other fitted Parameters
            of the form::

            `{'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}`

        """
        self.name = name
        self._val = value
        self.user_data = user_data
        self.init_value = value
        self.min = min
        self.max = max
        self.brute_step = brute_step
        self.vary = vary
        self._expr = expr
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self._delay_asteval = False
        self.stderr = None
        self.correl = None
        self.from_internal = lambda val: val
        self._init_bounds()

    def set(self, value=None, vary=None, min=None, max=None, expr=None,
            brute_step=None):
        """Set or update Parameter attributes.

        Parameters
        ----------
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is varied during a fit.
        min : float, optional
            Lower bound for value. To remove a lower bound you must use
            `-numpy.inf`.
        max : float, optional
            Upper bound for value. To remove an upper bound you must use
            `numpy.inf`.
        expr : str, optional
            Mathematical expression used to constrain the value during the fit.
            To remove a constraint you must supply an empty string.
        brute_step : float, optional
            Step size for grid points in the `brute` method. To remove the
            step size you must use ``0``.

        Notes
        -----
        Each argument to `set()` has a default value of `None`, which will
        leave the current value for the attribute unchanged.  Thus, to lift a
        lower or upper bound, passing in `None` will not work.  Instead,
        you must set these to `-numpy.inf` or `numpy.inf`, as with::

            par.set(min=None)        # leaves lower bound unchanged
            par.set(min=-numpy.inf)  # removes lower bound

        Similarly, to clear an expression, pass a blank string, (not
        ``None``!) as with::

            par.set(expr=None)  # leaves expression unchanged
            par.set(expr='')    # removes expression

        Explicitly setting a value or setting `vary=True` will also
        clear the expression.

        Finally, to clear the brute_step size, pass ``0``, not ``None``::

            par.set(brute_step=None)  # leaves brute_step unchanged
            par.set(brute_step=0)     # removes brute_step

        """
        if value is not None:
            self.value = value
            self.__set_expression('')

        if vary is not None:
            self.vary = vary
            if vary:
                self.__set_expression('')

        if min is not None:
            self.min = min

        if max is not None:
            self.max = max

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
            raise ValueError("Parameter '%s' has min == max" % self.name)
        if self._val > self.max:
            self._val = self.max
        if self._val < self.min:
            self._val = self.min
        self.setup_bounds()

    def __getstate__(self):
        """Get state for pickle."""
        return (self.name, self.value, self.vary, self.expr, self.min,
                self.max, self.brute_step, self.stderr, self.correl,
                self.init_value, self.user_data)

    def __setstate__(self, state):
        """Set state for pickle."""
        (self.name, self.value, self.vary, self.expr, self.min, self.max,
         self.brute_step, self.stderr, self.correl, self.init_value,
         self.user_data) = state
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self._delay_asteval = False
        self._init_bounds()

    def __repr__(self):
        """Return printable representation of a Parameter object."""
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        sval = repr(self._getval())
        if not self.vary and self._expr is None:
            sval = "value=%s (fixed)" % sval
        elif self.stderr is not None:
            sval = "value=%s +/- %.3g" % (sval, self.stderr)
        s.append(sval)
        s.append("bounds=[%s:%s]" % (repr(self.min), repr(self.max)))
        if self._expr is not None:
            s.append("expr='%s'" % self.expr)
        if self.brute_step is not None:
            s.append("brute_step=%s" % (self.brute_step))
        return "<Parameter %s>" % ', '.join(s)

    def setup_bounds(self):
        """Set up Minuit-style internal/external parameter transformation of
        min/max bounds.

        As a side-effect, this also defines the self.from_internal method
        used to re-calculate self.value from the internal value, applying
        the inverse Minuit-style transformation. This method should be
        called prior to passing a Parameter to the user-defined objective
        function.

        This code borrows heavily from JJ Helmus' leastsqbound.py

        Returns
        -------
        _val : float
            The internal value for parameter from self.value (which holds
            the external, user-expected value). This internal value should
            actually be used in a fit.

        """
        if self.min is None:
            self.min = -inf
        if self.max is None:
            self.max = inf
        if self.min == -inf and self.max == inf:
            self.from_internal = lambda val: val
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
        return _val

    def scale_gradient(self, val):
        """Return scaling factor for gradient.

        Parameters
        ----------
        val: float
            Numerical Parameter value.

        Returns
        -------
        float
            Scaling factor for gradient the according to Minuit-style
            transformation.

        """
        if self.min == -inf and self.max == inf:
            return 1.0
        elif self.max == inf:
            return val / sqrt(val*val + 1)
        elif self.min == -inf:
            return -val / sqrt(val*val + 1)
        return cos(val) * (self.max - self.min) / 2.0

    def _getval(self):
        """Get value, with bounds applied."""
        # Note assignment to self._val has been changed to self.value
        # The self.value property setter makes sure that the
        # _expr_eval.symtable is kept updated.
        # If you just assign to self._val then
        # _expr_eval.symtable[self.name]
        # becomes stale if parameter.expr is not None.
        if (isinstance(self._val, uncertainties.core.Variable) and
                self._val is not nan):

            try:
                self.value = self._val.nominal_value
            except AttributeError:
                pass
        if not self.vary and self._expr is None:
            return self._val

        if self._expr is not None:
            if self._expr_ast is None:
                self.__set_expression(self._expr)

            if self._expr_eval is not None:
                if not self._delay_asteval:
                    self.value = self._expr_eval(self._expr_ast)
                    check_ast_errors(self._expr_eval)

        if self._val is not None:
            if self._val > self.max:
                self._val = self.max
            elif self._val < self.min:
                self._val = self.min
        if self._expr_eval is not None:
            self._expr_eval.symtable[self.name] = self._val
        return self._val

    def set_expr_eval(self, evaluator):
        """Set expression evaluator instance."""
        self._expr_eval = evaluator

    @property
    def value(self):
        """Return the numerical value of the Parameter, with bounds applied."""
        return self._getval()

    @value.setter
    def value(self, val):
        """Set the numerical Parameter value."""
        self._val = val
        if not hasattr(self, '_expr_eval'):
            self._expr_eval = None
        if self._expr_eval is not None:
            self._expr_eval.symtable[self.name] = val

    @property
    def expr(self):
        """Return the mathematical expression used to constrain the value
        during the fit."""
        return self._expr

    @expr.setter
    def expr(self, val):
        """Set the mathematical expression used to constrain the value during
        the fit.

        To remove a constraint you must supply an empty string.

        """
        self.__set_expression(val)

    def __set_expression(self, val):
        if val == '':
            val = None
        self._expr = val
        if val is not None:
            self.vary = False
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

    def __nonzero__(self):
        """not zero"""
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

    def __div__(self, other):
        """/"""
        return self._getval() / other
    __truediv__ = __div__

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

    def __rdiv__(self, other):
        """/ (right)"""
        return other / self._getval()
    __rtruediv__ = __rdiv__

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


def isParameter(x):
    """Test for Parameter-ness."""
    return (isinstance(x, Parameter) or
            x.__class__.__name__ == 'Parameter')
