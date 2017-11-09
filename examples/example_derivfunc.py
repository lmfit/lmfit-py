#!/usr/bin/env python

import numpy as np

from lmfit import Minimizer, Parameters

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def func(pars, x, data=None):
    a, b, c = pars['a'], pars['b'], pars['c']
    model = a * np.exp(-b*x) + c
    if data is None:
        return model
    return model - data


def dfunc(pars, x, data=None):
    a, b, c = pars['a'], pars['b'], pars['c']
    v = np.exp(-b*x)
    return np.array([v, -a*x*v, np.ones(len(x))])


def f(var, x):
    return var[0] * np.exp(-var[1]*x) + var[2]


params1 = Parameters()
params1.add('a', value=10)
params1.add('b', value=10)
params1.add('c', value=10)

params2 = Parameters()
params2.add('a', value=10)
params2.add('b', value=10)
params2.add('c', value=10)

a, b, c = 2.5, 1.3, 0.8
x = np.linspace(0, 4, 50)
y = f([a, b, c], x)
data = y + 0.15*np.random.normal(size=len(x))

# fit without analytic derivative
min1 = Minimizer(func, params1, fcn_args=(x,), fcn_kws={'data': data})
out1 = min1.leastsq()
fit1 = func(out1.params, x)

# fit with analytic derivative
min2 = Minimizer(func, params2, fcn_args=(x,), fcn_kws={'data': data})
out2 = min2.leastsq(Dfun=dfunc, col_deriv=1)
fit2 = func(out2.params, x)

print('''Comparison of fit to exponential decay
with and without analytic derivatives, to
   model = a*exp(-b*x) + c
for a = %.2f, b = %.2f, c = %.2f
==============================================
Statistic/Parameter|   Without   | With      |
----------------------------------------------
N Function Calls   |   %3i       |   %3i     |
Chi-square         |   %.4f    |   %.4f  |
   a               |   %.4f    |   %.4f  |
   b               |   %.4f    |   %.4f  |
   c               |   %.4f    |   %.4f  |
----------------------------------------------
''' % (a, b, c,
       out1.nfev, out2.nfev,
       out1.chisqr, out2.chisqr,
       out1.params['a'], out2.params['a'],
       out1.params['b'], out2.params['b'],
       out1.params['c'], out2.params['c']))


if HASPYLAB:
    plt.plot(x, data, 'ro')
    plt.plot(x, fit1, 'b')
    plt.plot(x, fit2, 'k')
    plt.show()
