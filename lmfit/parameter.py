"""
Parameter class
"""
from __future__ import division
from numpy import arcsin, cos, sin, sqrt, inf, nan
import json
from copy import deepcopy
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from . import uncertainties

from .asteval import Interpreter
from .astutils import get_ast_names, valid_symbol_name

def check_ast_errors(expr_eval):
    """check for errors derived from asteval"""
    if len(expr_eval.error) > 0:
        expr_eval.raise_exception(None)

class Parameters(OrderedDict):
    """
    A dictionary of all the Parameters required to specify a fit model.

    All keys must be strings, and valid Python symbol names, and all values
    must be Parameters.

    Custom methods:
    ---------------

    add()
    add_many()
    dumps() / dump()
    loads() / load()
    """
    def __init__(self, asteval=None, *args, **kwds):
        super(Parameters, self).__init__(self)
        self._asteval = asteval
        if asteval is None:
            self._asteval = Interpreter()
        self.update(*args, **kwds)

    def __deepcopy__(self, memo):
        _pars = Parameters()
        for key, val in self._asteval.symtable.items():
            if key not in self._asteval.no_deepcopy:
                _pars._asteval.symtable[key] = deepcopy(val, memo)
        for key, par in self.items():
            if isinstance(par, Parameter):
                name = par.name
                _pars.add(name, value=par.value, min=par.min, max=par.max)
                _pars[name].vary = par.vary
                _pars[name].stderr = par.stderr
                _pars[name].correl = par.correl
                _pars[name].init_value = par.init_value
                _pars[name].expr = par.expr
                _pars._asteval.symtable[name] = par.value
        return _pars

    def __setitem__(self, key, par):
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if par is not None and not isinstance(par, Parameter):
            raise ValueError("'%s' is not a Parameter" % par)
        OrderedDict.__setitem__(self, key, par)
        par.name = key
        par._expr_eval = self._asteval
        self._asteval.symtable[key] = par.value
        self.update_constraints()

    def __add__(self, other):
        "add Parameters objects"
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        out = deepcopy(self)
        out.update(other)
        return out

    def __iadd__(self, other):
        "add/assign Parameters objects"
        if not isinstance(other, Parameters):
            raise ValueError("'%s' is not a Parameters object" % other)
        self.update(other)
        return self

    def update_constraints(self):
        """update all constrained parameters, checking that
        dependencies are evaluated as needed.
        """
        _updated = dict([(name, False) for name in self.keys()])
        def _update_param(name):
            """update a parameter value, including setting bounds.
            For a constrained parameter (one with an expr defined),
            this first updates (recursively) all parameters on which
            the parameter depends (using the 'deps' field).
            """
            # Has this param already been updated?
            # if this an expression dependency, it may have been
            if _updated[name]:
                return
            par = self.__getitem__(name)
            if par._expr_eval is None:
                par._expr_eval = self._asteval
            if par._expr is not None:
                par.expr = par._expr
            if par._expr_ast is not None:
                for dep in par._expr_deps:
                    if dep in self.keys():
                        _update_param(dep)
            self._asteval.symtable[name] = par.value
            _updated[name] = True

        for name in self.keys():
            _update_param(name)

    def pretty_repr(self, oneline=False):
        if oneline:
            return super(Parameters, self).__repr__()
        s = "Parameters({\n"
        for key in self.keys():
            s += "    '%s': %s, \n" % (key, self[key])
        s += "    })\n"
        return s

    def pretty_print(self, oneline=False):
        print(self.pretty_repr(oneline=oneline))

    def add(self, name, value=None, vary=True, min=None, max=None, expr=None):
        """
        Convenience function for adding a Parameter:

        Example
        -------
        p = Parameters()
        p.add(name, value=XX, ...)

        is equivalent to:
        p[name] = Parameter(name=name, value=XX, ....
        """
        self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                         min=min, max=max, expr=expr))

    def add_many(self, *parlist):
        """
        Convenience function for adding a list of Parameters.

        Parameters
        ----------
        parlist : sequence
        A sequence of tuples, each containing at least the name. The order in
        each tuple is the following:
            name, value, vary, min, max, expr

        Example
        -------
        p = Parameters()
        p.add_many( (name1, val1, True, None, None, None),
                    (name2, val2, True,  0.0, None, None),
                    (name3, val3, False, None, None, None),
                    (name4, val4))

        """
        for para in parlist:
            self.add(*para)

    def valuesdict(self):
        """
        Returns
        -------
        An ordered dictionary of name:value pairs for each Parameter.
        This is distinct from the Parameters itself, as it has values of
        the Parameter values, not the full Parameter object.
        """

        return OrderedDict(((p.name, p.value) for p in self.values()))

    def dumps(self, **kws):
        """represent Parameters as a JSON string.

        all keyword arguments are passed to `json.dumps()`

        Returns
        -------
        json string representation of Parameters

        See Also
        --------
        dump(), loads(), load(), json.dumps()
        """
        out = [p.__getstate__() for p in self.values()]
        return json.dumps(out, **kws)

    def loads(self, s, **kws):
        """load Parameters from a JSON string.

        current Parameters will be cleared before loading.

        all keyword arguments are passed to `json.loads()`

        Returns
        -------
        None.   Parameters are updated as a side-effect

        See Also
        --------
        dump(), dumps(), load(), json.loads()

        """
        self.clear()
        for parstate in json.loads(s, **kws):
            _par = Parameter()
            _par.__setstate__(parstate)
            self.__setitem__(parstate[0], _par)

    def dump(self, fp, **kws):
        """write JSON representation of Parameters to a file
        or file-like object (must have a `write()` method).

        Arguments
        ---------
        fp         open file-like object with `write()` method.

        all keyword arguments are passed to `dumps()`

        Returns
        -------
        return value from `fp.write()`

        See Also
        --------
        dump(), load(), json.dump()
        """
        return fp.write(self.dumps(**kws))

    def load(self, fp, **kws):
        """load JSON representation of Parameters from a file
        or file-like object (must have a `read()` method).

        Arguments
        ---------
        fp         open file-like object with `read()` method.

        all keyword arguments are passed to `loads()`

        Returns
        -------
        None.   Parameters are updated as a side-effect

        See Also
        --------
        dump(), loads(), json.load()
        """
        return self.loads(fp.read(), **kws)

class Parameter(object):
    """
    A Parameter is an object used to define a Fit Model.
    Attributes
    ----------
    name : str
        Parameter name.
    value : float
        The numerical value of the Parameter.
    vary : bool
        Whether the Parameter is fixed during a fit.
    min : float
        Lower bound for value (None = no lower bound).
    max : float
        Upper bound for value (None = no upper bound).
    expr : str
        An expression specifying constraints for the parameter.
    stderr : float
        The estimated standard error for the best-fit value.
    correl : dict
        Specifies correlation with the other fitted Parameter after a fit.
        Of the form `{'decay': 0.404, 'phase': -0.020, 'frequency': 0.102}`
    """
    def __init__(self, name=None, value=None, vary=True,
                 min=None, max=None, expr=None):
        """
        Parameters
        ----------
        name : str, optional
            Name of the parameter.
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is fixed during a fit.
        min : float, optional
            Lower bound for value (None = no lower bound).
        max : float, optional
            Upper bound for value (None = no upper bound).
        expr : str, optional
            Mathematical expression used to constrain the value during the fit.
        """
        self.name = name
        self._val = value
        self.user_value = value
        self.init_value = value
        self.min = min
        self.max = max
        self.vary = vary
        self._expr = expr
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self.stderr = None
        self.correl = None
        self.from_internal = lambda val: val
        self._init_bounds()

    def set(self, value=None, vary=None, min=None, max=None, expr=None):
        """
        Set or update Parameter attributes.

        Parameters
        ----------
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is fixed during a fit.
        min : float, optional
            Lower bound for value. To remove a lower bound you must use -np.inf
        max : float, optional
            Upper bound for value. To remove an upper bound you must use np.inf
        expr : str, optional
            Mathematical expression used to constrain the value during the fit.
            To remove a constraint you must supply an empty string.
        """

        self.__set_expression(expr)
        if value is not None:
            self._val = value
        if vary is not None:
            self.vary = vary
        if min is not None:
            self.min = min
        if max is not None:
            self.max = max

    def _init_bounds(self):
        """make sure initial bounds are self-consistent"""
        #_val is None means - infinity.
        if self._val is not None:
            if self.max is not None and self._val > self.max:
                self._val = self.max
            if self.min is not None and self._val < self.min:
                self._val = self.min
        elif self.min is not None and self._expr is None:
            self._val = self.min
        elif self.max is not None and self._expr is None:
            self._val = self.max
        self.setup_bounds()

    def __getstate__(self):
        """get state for pickle"""
        return (self.name, self.value, self.vary, self.expr, self.min,
                self.max, self.stderr, self.correl, self.init_value)

    def __setstate__(self, state):
        """set state for pickle"""
        (self.name, self.value, self.vary, self.expr, self.min,
         self.max, self.stderr, self.correl, self.init_value) = state
        self._expr_ast = None
        self._expr_eval = None
        self._expr_deps = []
        self._init_bounds()

    def __repr__(self):
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        sval = repr(self._getval())
        if not self.vary and self._expr is None:
            sval = "value=%s (fixed)" % (sval)
        elif self.stderr is not None:
            sval = "value=%s +/- %.3g" % (sval, self.stderr)
        s.append(sval)
        s.append("bounds=[%s:%s]" % (repr(self.min), repr(self.max)))
        if self._expr is not None:
            s.append("expr='%s'" % (self.expr))
        return "<Parameter %s>" % ', '.join(s)

    def setup_bounds(self):
        """
        Set up Minuit-style internal/external parameter transformation
        of min/max bounds.

        As a side-effect, this also defines the self.from_internal method
        used to re-calculate self.value from the internal value, applying
        the inverse Minuit-style transformation.  This method should be
        called prior to passing a Parameter to the user-defined objective
        function.

        This code borrows heavily from JJ Helmus' leastsqbound.py

        Returns
        -------
        The internal value for parameter from self.value (which holds
        the external, user-expected value).   This internal value should
        actually be used in a fit.
        """
        if self.min in (None, -inf) and self.max in (None, inf):
            self.from_internal = lambda val: val
            _val  = self._val
        elif self.max in (None, inf):
            self.from_internal = lambda val: self.min - 1.0 + sqrt(val*val + 1)
            _val  = sqrt((self._val - self.min + 1.0)**2 - 1)
        elif self.min in (None, -inf):
            self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
            _val  = sqrt((self.max - self._val + 1.0)**2 - 1)
        else:
            self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                                 (self.max - self.min) / 2.0
            _val  = arcsin(2*(self._val - self.min)/(self.max - self.min) - 1)
        return _val

    def scale_gradient(self, val):
        """
        Returns
        -------
        scaling factor for gradient the according to Minuit-style
        transformation.
        """
        if self.min in (None, -inf) and self.max in (None, inf):
            return 1.0
        elif self.max in (None, inf):
            return val / sqrt(val*val + 1)
        elif self.min in (None, -inf):
            return -val / sqrt(val*val + 1)
        else:
            return cos(val) * (self.max - self.min) / 2.0


    def _getval(self):
        """get value, with bounds applied"""
        if (self._val is not nan and
            isinstance(self._val, uncertainties.Variable)):
            try:
                self._val = self._val.nominal_value
            except AttributeError:
                pass
        if not self.vary and self._expr is None:
            return self._val
        if not hasattr(self, '_expr_eval'):
            self._expr_eval = None
        if not hasattr(self, '_expr_ast'):
            self._expr_ast = None
        if self._expr_ast is not None and self._expr_eval is not None:
            self._val = self._expr_eval(self._expr_ast)
            check_ast_errors(self._expr_eval)

        if self.min is None:
            self.min = -inf
        if self.max is None:
            self.max =  inf
        if self.max < self.min:
            self.max, self.min = self.min, self.max
        if (abs((1.0*self.max - self.min)/
                max(abs(self.max), abs(self.min), 1.e-13)) < 1.e-13):
            raise ValueError("Parameter '%s' has min == max" % self.name)
        try:
            if self.min > -inf:
                self._val = max(self.min, self._val)
            if self.max < inf:
                self._val = min(self.max, self._val)
        except(TypeError, ValueError):
            self._val = nan
        return self._val

    def set_expr_eval(self, evaluator):
        "set expression evaluator instance"
        self._expr_eval = evaluator

    @property
    def value(self):
        "The numerical value of the Parameter, with bounds applied"
        return self._getval()

    @value.setter
    def value(self, val):
        "Set the numerical Parameter value."
        self._val = val
        if not hasattr(self, '_expr_eval'):  self._expr_eval = None
        if self._expr_eval is not None:
            self._expr_eval.symtable[self.name] = val

    @property
    def expr(self):
        """
        The mathematical expression used to constrain the value during the fit.
        """
        return self._expr

    @expr.setter
    def expr(self, val):
        """
        The mathematical expression used to constrain the value during the fit.
        To remove a constraint you must supply an empty string.
        """
        self.__set_expression(val)

    def __set_expression(self, val):
        if val == '':
            val = None
        self._expr = val
        if val is not None:
            self.vary = False
        if not hasattr(self, '_expr_eval'):  self._expr_eval = None
        if val is None: self._expr_ast = None
        if val is not None and self._expr_eval is not None:
            self._expr_ast = self._expr_eval.parse(val)
            check_ast_errors(self._expr_eval)
            self._expr_deps = get_ast_names(self._expr_ast)

    def __str__(self):
        "string"
        return self.__repr__()

    def __abs__(self):
        "abs"
        return abs(self._getval())

    def __neg__(self):
        "neg"
        return -self._getval()

    def __pos__(self):
        "positive"
        return +self._getval()

    def __nonzero__(self):
        "not zero"
        return self._getval() != 0

    def __int__(self):
        "int"
        return int(self._getval())

    def __long__(self):
        "long"
        return long(self._getval())

    def __float__(self):
        "float"
        return float(self._getval())

    def __trunc__(self):
        "trunc"
        return self._getval().__trunc__()

    def __add__(self, other):
        "+"
        return self._getval() + other

    def __sub__(self, other):
        "-"
        return self._getval() - other

    def __div__(self, other):
        "/"
        return self._getval() / other
    __truediv__ = __div__

    def __floordiv__(self, other):
        "//"
        return self._getval() // other

    def __divmod__(self, other):
        "divmod"
        return divmod(self._getval(), other)

    def __mod__(self, other):
        "%"
        return self._getval() % other

    def __mul__(self, other):
        "*"
        return self._getval() * other

    def __pow__(self, other):
        "**"
        return self._getval() ** other

    def __gt__(self, other):
        ">"
        return self._getval() > other

    def __ge__(self, other):
        ">="
        return self._getval() >= other

    def __le__(self, other):
        "<="
        return self._getval() <= other

    def __lt__(self, other):
        "<"
        return self._getval() < other

    def __eq__(self, other):
        "=="
        return self._getval() == other
    def __ne__(self, other):
        "!="
        return self._getval() != other

    def __radd__(self, other):
        "+ (right)"
        return other + self._getval()

    def __rdiv__(self, other):
        "/ (right)"
        return other / self._getval()
    __rtruediv__ = __rdiv__

    def __rdivmod__(self, other):
        "divmod (right)"
        return divmod(other, self._getval())

    def __rfloordiv__(self, other):
        "// (right)"
        return other // self._getval()

    def __rmod__(self, other):
        "% (right)"
        return other % self._getval()

    def __rmul__(self, other):
        "* (right)"
        return other * self._getval()

    def __rpow__(self, other):
        "** (right)"
        return other ** self._getval()

    def __rsub__(self, other):
        "- (right)"
        return other - self._getval()

def isParameter(x):
    "test for Parameter-ness"
    return (isinstance(x, Parameter) or
            x.__class__.__name__ == 'Parameter')
