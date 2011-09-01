from lmfit import Parameters, minimize

from numpy import linspace, zeros, sin, exp, random, sqrt, pi, sign
from scipy.optimize import leastsq

try:
    import pylab
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


from testutils import report_errors

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.33)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.010)

def residual(pars, x, data=None):
    amp = pars['amp'].value
    per = pars['period'].value
    shift = pars['shift'].value
    decay = pars['decay'].value

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp*sin(shift + x/per) * exp(-x*x*decay*decay)
    if data is None:
        return model
    return (model - data)

n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.7215, size=n)
x     = linspace(xmin, xmax, n)
data  = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=13.0)
fit_params.add('period', value=2)
fit_params.add('shift', value=0.0)
fit_params.add('decay', value=0.02)

out = minimize(residual, fit_params, args=(x,), kws={'data':data})

fit = residual(fit_params, x)

print ' N fev = ', out.nfev
print out.chisqr, out.redchi, out.nfree

report_errors(fit_params)


if HASPYLAB:
    pylab.plot(x, data, 'ro')
    pylab.plot(x, fit, 'b')
    pylab.show()





