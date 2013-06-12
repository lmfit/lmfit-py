"""
Parameter class
"""
from numpy import arcsin, cos, sin, sqrt, inf, nan

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import re
from . import uncertainties


RESERVED_WORDS = ('and', 'as', 'assert', 'break', 'class', 'continue',
                  'def', 'del', 'elif', 'else', 'except', 'exec',
                  'finally', 'for', 'from', 'global', 'if', 'import', 'in',
                  'is', 'lambda', 'not', 'or', 'pass', 'print', 'raise',
                  'return', 'try', 'while', 'with', 'True', 'False',
                  'None', 'eval', 'execfile', '__import__', '__package__')

NAME_MATCH = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*$").match

def valid_symbol_name(name):
    "input is a valid name"
    if name in RESERVED_WORDS:
        return False
    return NAME_MATCH(name) is not None


class Parameters(OrderedDict):
    """a custom dictionary of Parameters.  All keys must be
    strings, and valid Python symbol names, and all values
    must be Parameters.

    Custom methods:
    ---------------

    add()
    add_many()
    """
    def __init__(self, *args, **kwds):
        OrderedDict.__init__(self)
        self.update(*args, **kwds)

    def __setitem__(self, key, value):
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if value is not None and not isinstance(value, Parameter):
            raise ValueError("'%s' is not a Parameter" % value)
        OrderedDict.__setitem__(self, key, value)
        value.name = key

    def add(self, name, value=None, vary=True, min=None, max=None, expr=None):
        """convenience function for adding a Parameter:
        with   p = Parameters()
        p.add(name, value=XX, ....)

        is equivalent to
        p[name] = Parameter(name=name, value=XX, ....
        """
        self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                         min=min, max=max, expr=expr))

    def add_many(self, *parlist):
        """convenience function for adding a list of Parameters:
        Here, you must provide a sequence of tuples, each containing
        at least the name. The order in each tuple is the following:
            name, value, vary, min, max, expr
        with   p = Parameters()
        p.add_many( (name1, val1, True, None, None, None),
                    (name2, val2, True,  0.0, None, None),
                    (name3, val3, False, None, None, None),
                    (name4, val4))

        """
        for para in parlist:
            self.add(*para)

class Parameter(object):
    """A Parameter is the basic Parameter going
    into Fit Model.  The Parameter holds many attributes:
    value, vary, max_value, min_value, constraint expression.
    The value and min/max values will be be set to floats.
    """
    def __init__(self, name=None, value=None, vary=True,
                 min=None, max=None, expr=None):
        self.name = name
        self._val = value
        self.user_value = value
        self.init_value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.deps   = None
        self.stderr = None
        self.correl = None
        if self.max is not None and value > self.max:
            self._val = self.max
        if self.min is not None and value < self.min:
            self._val = self.min
        self.from_internal = lambda val: val

    def __repr__(self):
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        sval = repr(self._val)
        if self.stderr is not None:
            sval = "value=%s +/- %.3g" % (sval, self.stderr)
        if not self.vary and self.expr is None:
            sval = "value=%s (fixed)" % (sval)
        s.append(sval)
        s.append("bounds=[%s:%s]" % (repr(self.min), repr(self.max)))
        if self.expr is not None:
            s.append("expr='%s'" % (self.expr))
        return "<Parameter %s>" % ', '.join(s)

    def setup_bounds(self):
        """set up Minuit-style internal/external parameter transformation
        of min/max bounds.

        returns internal value for parameter from self.value (which holds
        the external, user-expected value).   This internal values should
        actually be used in a fit....

        As a side-effect, this also defines the self.from_internal method
        used to re-calculate self.value from the internal value, applying
        the inverse Minuit-style transformation.  This method should be
        called prior to passing a Parameter to the user-defined objective
        function.

        This code borrows heavily from JJ Helmus' leastsqbound.py
        """
        if self.min in (None, -inf) and self.max in (None, inf):
            self.from_internal = lambda val: val
            _val  = self._val
        elif self.max in (None, inf):
            self.from_internal = lambda val: self.min - 1 + sqrt(val*val + 1)
            _val  = sqrt((self._val - self.min + 1)**2 - 1)
        elif self.min in (None, -inf):
            self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
            _val  = sqrt((self.max - self._val + 1)**2 - 1)
        else:
            self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                                 (self.max - self.min) / 2
            _val  = arcsin(2*(self._val - self.min)/(self.max - self.min) - 1)
        return _val

    def scale_gradient(self, val):
        """returns scaling factor for gradient the according to Minuit-style
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
            self._val = self._val.nominal_value

        if self.min is None:
            self.min = -inf
        if self.max is None:
            self.max =  inf
        if self.max < self.min:
            self.max, self.min = self.min, self.max

        try:
            if self.min > -inf:
                self._val = max(self.min, self._val)
            if self.max < inf:
                self._val = min(self.max, self._val)
        except(TypeError, ValueError):
            self._val = nan
        return self._val

    @property
    def value(self):
        "get value"
        return self._getval()

    @value.setter
    def value(self, val):
        "set value"
        self._val = val
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

