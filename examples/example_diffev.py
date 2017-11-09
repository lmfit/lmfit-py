#!/usr/bin/env python

"""Example comparing leastsq with differential_evolution
on a fairly simple problem.
"""
import numpy as np

import lmfit

try:
    import matplotlib.pyplot as plt
    HAS_PYLAB = True
except ImportError:
    HAS_PYLAB = False

np.random.seed(2)
x = np.linspace(0, 10, 101)

# Setup example
decay = 5
offset = 1.0
amp = 2.0
omega = 4.0

y = offset + amp*np.sin(omega*x) * np.exp(-x/decay)
yn = y + np.random.normal(size=len(y), scale=0.450)


def resid(params, x, ydata):
    decay = params['decay'].value
    offset = params['offset'].value
    omega = params['omega'].value
    amp = params['amp'].value

    y_model = offset + amp * np.sin(x*omega) * np.exp(-x/decay)
    return y_model - ydata


params = lmfit.Parameters()
params.add('offset', 2.0, min=0, max=10.0)
params.add('omega', 3.3, min=0, max=10.0)
params.add('amp', 2.5, min=0, max=10.0)
params.add('decay', 1.0, min=0, max=10.0)

o1 = lmfit.minimize(resid, params, args=(x, yn), method='leastsq')
print("\n\n# Fit using leastsq:")
lmfit.report_fit(o1)

o2 = lmfit.minimize(resid, params, args=(x, yn), method='differential_evolution')
print("\n\n# Fit using differential_evolution:")
lmfit.report_fit(o2)

if HAS_PYLAB:
    plt.plot(x, yn, 'ko', lw=2)
    plt.plot(x, yn+o1.residual, 'r-', lw=2)
    plt.plot(x, yn+o2.residual, 'b--', lw=2)
    plt.legend(['data', 'leastsq', 'diffev'], loc='upper left')
    plt.show()
