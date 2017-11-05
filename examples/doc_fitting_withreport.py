#!/usr/bin/env python

# <examples/doc_fitting_withreport.py>
from numpy import exp, linspace, pi, random, sign, sin

from lmfit import Parameters, fit_report, minimize

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.46)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.032)


def residual(pars, x, data=None):
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


n = 1001
xmin = 0.
xmax = 250.0

random.seed(0)

noise = random.normal(scale=0.7215, size=n)
x = linspace(xmin, xmax, n)
data = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=13.0)
fit_params.add('period', value=2)
fit_params.add('shift', value=0.0)
fit_params.add('decay', value=0.02)

out = minimize(residual, fit_params, args=(x,), kws={'data': data})

print(fit_report(out))
# <end of examples/doc_fitting_withreport.py>
