"""
Fit Parameter for making Fitting Models
"""

class Parameter(object):
    """A Parameter is the basic Parameter going
into Fit Model.  The Parameter holds many attributes:
  value, vary, max_value, min_value, constraint expression.
    """
    def __init__(self, value=None, vary=True,
                 min=None, max=None, expr=None, **kws):
        self.value = value
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr
        self.stderr = 0.0
        self.correl = None

    def __repr__(self):
        s = []
        vstr = 'varied'
        if not self.vary:
            vstr = 'fixed'
        s.append("value=%s (%s)" % (repr(self.value), vstr))
        s.append("bounds=[%s:%s]" % (repr(self.min),repr(self.max)))
        if self.expr is not None:
            s.append("expr='%s'" % (self.expr))

        return "<Parameter %s>" % ', '.join(s)
