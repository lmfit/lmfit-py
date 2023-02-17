# <examples/doc_fitting_withreport.py>
from numpy import exp, linspace, pi, random, sign, sin

from lmfit import create_params, fit_report, minimize

p_true = create_params(amp=14.0, period=5.46, shift=0.123, decay=0.032)


def residual(pars, x, data=None):
    """Model a decaying sine wave and subtract data."""
    vals = pars.valuesdict()
    amp = vals['amp']
    per = vals['period']
    shift = vals['shift']
    decay = vals['decay']

    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = amp * sin(shift + x/per) * exp(-x*x*decay*decay)
    if data is None:
        return model
    return model - data


random.seed(0)
x = linspace(0.0, 250., 1001)
noise = random.normal(scale=0.7215, size=x.size)
data = residual(p_true, x) + noise

fit_params = create_params(amp=13, period=2, shift=0, decay=0.02)

out = minimize(residual, fit_params, args=(x,), kws={'data': data})

print(fit_report(out))
# <end examples/doc_fitting_withreport.py>
