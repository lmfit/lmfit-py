from numpy import arcsin, cos, sin, sqrt

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from .astutils import valid_symbol_name

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

    def add(self, name, value=None, vary=True, expr=None,
            min=None, max=None):
        """convenience function for adding a Parameter:
        with   p = Parameters()
        p.add(name, value=XX, ....)

        is equivalent to
        p[name] = Parameter(name=name, value=XX, ....
        """
        self.__setitem__(name, Parameter(value=value, name=name, vary=vary,
                                         expr=expr, min=min, max=max))

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
                 min=None, max=None, expr=None, **kws):
        self.name = name
        self.value = value
        self.user_value = value
        self.init_value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.deps   = None
        self.stderr = None
        self.correl = None
        if self.max is not None and value > self.max: self.value = self.max
        if self.min is not None and value < self.min: self.value = self.min
        self.from_internal = lambda val: val

    def __repr__(self):
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        sval = repr(self.value)
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
        if self.min is None and self.max is None:
            self.from_internal = lambda val: val
            _val  = self.value
        elif self.max is None:
            self.from_internal = lambda val: self.min - 1 + sqrt(val*val + 1)
            _val  = sqrt((self.value - self.min + 1)**2 - 1)
        elif self.min is None:
            self.from_internal = lambda val: self.max + 1 - sqrt(val*val + 1)
            _val  = sqrt((self.max - self.value + 1)**2 - 1)
        else:
            self.from_internal = lambda val: self.min + (sin(val) + 1) * \
                                 (self.max - self.min) / 2
            _val  = arcsin(2*(self.value - self.min)/(self.max - self.min) - 1)
        return _val

    def scale_gradient(self, val):
        """returns scaling factor for gradient the according to Minuit-style
        transformation.
        """
        if self.min is None and self.max is None:
            return 1.0
        elif self.max is None:
            return val / sqrt(val*val + 1)
        elif self.min is None:
            return -val / sqrt(val*val + 1)
        else:
            return cos(val) * (self.max - self.min) / 2.0


def isParameter(x):
    return (isinstance(x, Parameter) or
            x.__class__.__name__ == 'Parameter')

