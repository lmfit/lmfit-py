"""
Fit Parameter for making Fitting Models
"""

class FitParameter(object):
    """A FitParameter is the basic Parameter going
into Fit Model.  The FitParameter holds many attributes
for the Parameter:
  name, value, max, min value, constraint expression.
    """
    def __init__(self, name, value=None, float=True,
                 min=None, max=None, expr=None, **kws):
        self.name = name
        self.value = value
        self.float = float
        self.min = min
        self.max = max
        self.expr = expr
        self.uncertainty = 0.0
        self.correlation = None

    def __repr__(self):
        sout = "'%s', value=%s" % (self.name, repr(self.value))
        sout = "%s, bounds=[%s:%s]" % (sout, repr(self.min),
                                        repr(self.max))
        if self.expr is not None:
            sout = "%s, expr='%s'" % (sout, self.expr)

        return "<FitParameter %s>" % sout
