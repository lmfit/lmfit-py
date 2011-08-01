"""
Fit Parameter for making Fitting Models
"""

class Parameter(object):
    """A FitParameter is the basic Parameter going
into Fit Model.  The FitParameter holds many attributes
for the Parameter:
  value, vary, max_value, min_value, constraint expression.
    """
    def __init__(self, value=None, vary=True,
                 min=None, max=None, expr=None, **kws):
        self.value = value
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr
        self.uncertainty = 0.0
        self.correlation = None

    def __repr__(self):
        s = []
        vstr = 'fixed'
        if self.vary:
            vstr = 'varied'
        s.append("value=%s (%s)" % (repr(self.value), vstr))
        s.append("bounds=[%s:%s]" % (repr(self.min),repr(self.max)))
        if self.expr is not None:
            s.append("expr='%s'" % (self.expr))

        return "<Parameter %s>" % ', '.join(s)


