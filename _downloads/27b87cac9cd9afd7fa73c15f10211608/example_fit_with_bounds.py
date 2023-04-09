"""
Fit Using Bounds
================

A major advantage of using lmfit is that one can specify boundaries on fitting
parameters, even if the underlying algorithm in SciPy does not support this.
For more information on how this is implemented, please refer to:
https://lmfit.github.io/lmfit-py/bounds.html

The example below shows how to set boundaries using the ``min`` and ``max``
attributes to fitting parameters.

"""
import matplotlib.pyplot as plt
from numpy import exp, linspace, pi, random, sign, sin

from lmfit import create_params, minimize
from lmfit.printfuncs import report_fit

###############################################################################
# create the 'true' Parameter values and residual function:
p_true = create_params(amp=14.0, period=5.4321, shift=0.12345, decay=0.010)


def residual(pars, x, data=None):
    argu = (x * pars['decay'])**2
    shift = pars['shift']
    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = pars['amp'] * sin(shift + x/pars['period']) * exp(-argu)
    if data is None:
        return model
    return model - data


###############################################################################
# Generate synthetic data and initialize fitting Parameters:
random.seed(0)
x = linspace(0, 250, 1500)
noise = random.normal(scale=2.8, size=x.size)
data = residual(p_true, x) + noise

fit_params = create_params(amp=dict(value=13, max=20, min=0),
                           period=dict(value=2, max=10),
                           shift=dict(value=0, max=pi/2., min=-pi/2.),
                           decay=dict(value=0.02, max=0.1, min=0))

###############################################################################
# Perform the fit and show the results:
out = minimize(residual, fit_params, args=(x,), kws={'data': data})
fit = residual(out.params, x)

###############################################################################
report_fit(out, modelpars=p_true, correl_mode='table')

###############################################################################
plt.plot(x, data, 'o', label='data')
plt.plot(x, fit, label='best fit')
plt.legend()
plt.show()
