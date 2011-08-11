"""
Fit Parameter for making Fitting Models
"""

from astutils import valid_symbol_name

class Parameters(dict):
    """a dictionay of Parameters.  all keys must be strings
    that are valid symbol Python names, and all values must
    be Parameters"""
    def __init__(self, *args, **kwds):
        self.update(*args, **kwds)

    def __setitem__(self, key, value):
        if key not in self:
            if not valid_symbol_name(key):
                raise KeyError("'%s' is not a valid Parameters name" % key)
        if value is not None and not isinstance(value, Parameter):
            raise ValueError("'%s' is not a Parameter" % value)
        dict.__setitem__(self, key, value)
        value.name = key


class Parameter(object):
    """A Parameter is the basic Parameter going
into Fit Model.  The Parameter holds many attributes:
  value, vary, max_value, min_value, constraint expression.
    """
    def __init__(self, value=None, vary=True, name=None,
                 min=None, max=None, expr=None, **kws):
        self.value = value
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr
        self.name = None
        self.stderr = None
        self.correl = None

    def __repr__(self):
        s = []
        if self.name is not None:
            s.append("'%s'" % self.name)
        val = repr(self.value)
        if self.vary and self.stderr is not None:
            val = "value=%s +/- %.3g" % (repr(self.value), self.stderr)
        elif not self.vary:
            val = "value=%s (fixed)" % (repr(self.value))
        s.append(val)
        s.append("bounds=[%s:%s]" % (repr(self.min),repr(self.max)))
        if self.expr is not None:
            s.append("expr='%s'" % (self.expr))

        return "<Parameter %s>" % ', '.join(s)
