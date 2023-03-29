"""
Fit Specifying Different Reduce Function
========================================

The ``reduce_fcn`` specifies how to convert a residual array to a scalar value
for the scalar minimizers. The default value is None (i.e., "sum of squares of
residual") - alternatives are: ``negentropy``, ``neglogcauchy``, or a
user-specified ``callable``. For more information please refer to:
https://lmfit.github.io/lmfit-py/fitting.html#using-the-minimizer-class

Here, we use as an example the Student's t log-likelihood for robust fitting
of data with outliers.

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
# Generate synthetic data with noise/outliers and initialize fitting Parameters:
decay = 5
offset = 1.0
amp = 2.0
omega = 4.0

np.random.seed(2)
x = np.linspace(0, 10, 101)
y = offset + amp * np.sin(omega*x) * np.exp(-x/decay)
yn = y + np.random.normal(size=y.size, scale=0.250)

outliers = np.random.randint(int(len(x)/3.0), len(x), int(len(x)/12))
yn[outliers] += 5*np.random.random(len(outliers))

params = lmfit.create_params(offset=2.0, omega=3.3, amp=2.5,
                             decay=dict(value=1, min=0))

###############################################################################
# Perform fits using the ``L-BFGS-B`` method with different ``reduce_fcn``:
method = 'L-BFGS-B'
o1 = lmfit.minimize(resid, params, args=(x, yn), method=method)
print("# Fit using sum of squares:\n")
lmfit.report_fit(o1)

###############################################################################
o2 = lmfit.minimize(resid, params, args=(x, yn), method=method,
                    reduce_fcn='neglogcauchy')
print("\n\n# Robust Fit, using log-likelihood with Cauchy PDF:\n")
lmfit.report_fit(o2)

###############################################################################
plt.plot(x, y, 'o', label='true function')
plt.plot(x, yn, '--*', label='with noise+outliers')
plt.plot(x, yn+o1.residual, '-', label='sum of squares fit')
plt.plot(x, yn+o2.residual, '-', label='robust fit')
plt.legend()
plt.show()
