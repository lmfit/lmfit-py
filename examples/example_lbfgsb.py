#!/usr/bin/env python

from numpy import exp, linspace, pi, random, sign, sin

from lmfit import Parameters, minimize, report_fit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.33)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.010)


def residual(pars, x, data=None):
    argu = (x * pars['decay'])**2
    shift = pars['shift']
    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = pars['amp'] * sin(shift + x/pars['period']) * exp(-argu)
    if data is None:
        return model
    return model - data


n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.7215, size=n)
x = linspace(xmin, xmax, n)
data = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=11.0, min=5, max=20)
fit_params.add('period', value=5., min=1., max=7)
fit_params.add('shift', value=.10,  min=0.0, max=0.2)
fit_params.add('decay', value=6.e-3, min=0, max=0.1)

init = residual(fit_params, x)

out = minimize(residual, fit_params, method='lbfgsb', args=(x,),
               kws={'data': data})

fit = residual(out.params, x)

for name, par in out.params.items():
    nout = "%s:%s" % (name, ' '*(20-len(name)))
    print("%s: %s (true=%s) " % (nout, par.value, p_true[name].value))

report_fit(fit_params)

if HASPYLAB:
    plt.plot(x, data, 'r--')
    plt.plot(x, init, 'k')
    plt.plot(x, fit, 'b')
    plt.show()
