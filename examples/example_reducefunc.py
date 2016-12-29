#!/usr/bin/env python
"""
Example using the Student's t log-likelihood
for robust fitting of data with outliers.
"""
import numpy as np
import scipy.special as sp
import scipy.stats as stats
import lmfit

try:
    import matplotlib.pyplot as plt
    HAS_PYLAB = True
except ImportError:
    HAS_PYLAB = False


np.random.seed(2)
x = np.linspace(-5, 5, 31)

# Setup example
w = 0.5
c = 1.5
amp_sin = 2
omega = 3
amp_erf = 5

y  = amp_erf*sp.erf(x/w)+2*np.sin(x*omega)+c
yn = y + np.random.randn(*y.shape)*0.3
outliers = np.random.random_integers(0, x.size-1, x.size/5)
yn[outliers] += np.random.randn(outliers.size)*5.0


def resid(params, x, ydata):
    amp_erf = params['amp_erf'].value
    w       = params['w'].value
    c       = params['c'].value
    omega   = params['omega'].value
    amp_sin = params['amp_sin'].value

    y_model = amp_erf*sp.erf(x/w) + amp_sin* np.sin(x*omega) + c
    return y_model - ydata

params = lmfit.Parameters()
params.add('w', 0.25)
params.add('c', 0.75)
params.add('omega', 2.5, min=2, max=4)
params.add('amp_sin', 3.6)
params.add('amp_erf', 2.7)


def logln(x):
    "Returning the t-log-likehood of x with df=2"
    return -stats.t.logpdf(x, df=2).sum()

o1 = lmfit.minimize(resid, params, args=(x, yn), method='L-BFGS-B')
print("# Sum of Squares Fit:")
lmfit.report_fit(o1)

o2 = lmfit.minimize(resid, params, args=(x, yn), method='L-BFGS-B',
                    reducefunc=logln)

print("# Robust Fit, using logpdf():")
lmfit.report_fit(o2)


if HAS_PYLAB:
    plt.plot(x, y,  'ko', lw=2)
    plt.plot(x, yn, 'k--*', lw=1)
    plt.plot(x, yn+o1.residual, 'r-', lw=2)
    plt.plot(x, yn+o2.residual, 'b-', lw=2)
    plt.legend(['True function',
                'with noise+outliers',
                'sum of squares fit',
                'robust fit'], loc='upper left')
    plt.show()
