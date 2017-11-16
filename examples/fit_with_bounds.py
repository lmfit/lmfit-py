#!/usr/bin/env python

from numpy import exp, linspace, pi, random, sign, sin

from lmfit import Parameters, minimize
from lmfit.printfuncs import report_fit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.4321)
p_true.add('shift', value=0.12345)
p_true.add('decay', value=0.01000)


def residual(pars, x, data=None):
    argu = (x * pars['decay'])**2
    shift = pars['shift']
    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = pars['amp'] * sin(shift + x/pars['period']) * exp(-argu)
    if data is None:
        return model
    return model - data


n = 1500
xmin = 0.
xmax = 250.0
random.seed(0)
noise = random.normal(scale=2.80, size=n)
x = linspace(xmin, xmax, n)
data = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=13.0, max=20, min=0.0)
fit_params.add('period', value=2, max=10)
fit_params.add('shift', value=0.0, max=pi/2., min=-pi/2.)
fit_params.add('decay', value=0.02, max=0.10, min=0.00)

out = minimize(residual, fit_params, args=(x,), kws={'data': data})

fit = residual(out.params, x)

report_fit(out, show_correl=True, modelpars=p_true)

print('\n\nRaw (unordered, unscaled) Covariance Matrix:')
print(out.covar)

if HASPYLAB:
    plt.plot(x, data, 'ro')
    plt.plot(x, fit, 'b')
    plt.show()
