#!/usr/bin/env python

"""Example using the Student's t log-likelihood
for robust fitting of data with outliers.
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

y = offset + amp * np.sin(omega*x) * np.exp(-x/decay)

yn = y + np.random.normal(size=len(y), scale=0.250)

outliers = np.random.random_integers(int(len(x)/3.0), len(x)-1, int(len(x)/12))
yn[outliers] += 5*np.random.random(len(outliers))


def resid(params, x, ydata):
    decay = params['decay'].value
    offset = params['offset'].value
    omega = params['omega'].value
    amp = params['amp'].value

    y_model = offset + amp * np.sin(x*omega) * np.exp(-x/decay)
    return y_model - ydata


params = lmfit.Parameters()

params.add('offset', 2.0)
params.add('omega', 3.3)
params.add('amp', 2.5)
params.add('decay', 1.0, min=0)

method = 'L-BFGS-B'

o1 = lmfit.minimize(resid, params, args=(x, yn), method=method)
print("# Fit using sum of squares:")
lmfit.report_fit(o1)

o2 = lmfit.minimize(resid, params, args=(x, yn), method=method, reduce_fcn='neglogcauchy')
print("# Robust Fit, using log-likelihood with Cauchy PDF:")
lmfit.report_fit(o2)

if HAS_PYLAB:
    plt.plot(x, y, 'ko', lw=2)
    plt.plot(x, yn, 'k--*', lw=1)
    plt.plot(x, yn+o1.residual, 'r-', lw=2)
    plt.plot(x, yn+o2.residual, 'b-', lw=2)
    plt.legend(['True function',
                'with noise+outliers',
                'sum of squares fit',
                'robust fit'], loc='upper left')
    plt.show()
