"""
Fit Using differential_evolution Algorithm
==========================================

This example compares the ``leastsq`` and ``differential_evolution`` algorithms
on a fairly simple problem.

"""
import matplotlib.pyplot as plt
import numpy as np

import lmfit


def resid(params, x, ydata):
    decay = params['decay'].value
    offset = params['offset'].value
    omega = params['omega'].value
    amp = params['amp'].value

    y_model = offset + amp * np.sin(x*omega) * np.exp(-x/decay)
    return y_model - ydata


###############################################################################
# Generate synthetic data and set-up Parameters with initial values/boundaries:
decay = 5
offset = 1.0
amp = 2.0
omega = 4.0

np.random.seed(2)
x = np.linspace(0, 10, 101)
y = offset + amp*np.sin(omega*x) * np.exp(-x/decay)
yn = y + np.random.normal(size=y.size, scale=0.450)

params = lmfit.Parameters()
params.add('offset', 2.0, min=0, max=10.0)
params.add('omega', 3.3, min=0, max=10.0)
params.add('amp', 2.5, min=0, max=10.0)
params.add('decay', 1.0, min=0, max=10.0)

###############################################################################
# Perform the fits and show fitting results and plot:
o1 = lmfit.minimize(resid, params, args=(x, yn), method='leastsq')
print("# Fit using leastsq:")
lmfit.report_fit(o1)

###############################################################################
o2 = lmfit.minimize(resid, params, args=(x, yn), method='differential_evolution')
print("\n\n# Fit using differential_evolution:")
lmfit.report_fit(o2)

###############################################################################
plt.plot(x, yn, 'o', label='data')
plt.plot(x, yn+o1.residual, '-', label='leastsq')
plt.plot(x, yn+o2.residual, '--', label='diffev')
plt.legend()
