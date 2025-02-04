#!/usr/bin/env python
"""
Fitting data with uncertainties in x and y
==============================================

This examples shows a general way of fitting a model
to y(x) data which has uncertainties in both y and x.

For more in-depth discussion, see
    https://dx.doi.org/10.1021/acs.analchem.0c02178
"""

import matplotlib.pyplot as plt
import numpy as np

from lmfit import Minimizer, Parameters, report_fit
from lmfit.lineshapes import gaussian

# create data to be fitted
np.random.seed(17)
xtrue = np.linspace(0, 50, 101)
xstep = xtrue[1] - xtrue[0]
amp, cen, sig, offset, slope = 39, 28.2, 4.4, -13, 0.012

ytrue = (gaussian(xtrue, amplitude=amp, center=cen, sigma=sig)
         + offset + slope * xtrue)

ydat = ytrue + np.random.normal(size=xtrue.size, scale=0.1)

# we add errors to x after y has been created, as if there is
# an ideal y(x) and we have noise in both x and y.
# we force the uncertainty away from 'normal', forcing
# it to be smaller than the step size.
xerr = np.random.normal(size=xtrue.size, scale=0.1*xstep)
max_xerr = 0.8*xstep
xerr[np.where(xerr > max_xerr)] = max_xerr
xerr[np.where(xerr < -max_xerr)] = -max_xerr
xdat = xtrue + xerr

# now we assert that we know the uncertaintits in y and x
#   we'll pick values that are reesonable but not exactly
#   what we used to make the noise
yerr = 0.06
xerr = xstep


def peak_model(params, x):
    """Model a peak with a linear background."""
    amp = params['amp'].value
    cen = params['cen'].value
    sig = params['sig'].value
    offset = params['offset'].value
    slope = params['slope'].value
    return offset + slope * x + gaussian(x, amplitude=amp, center=cen, sigma=sig)


# objective without xerr
def objective_no_xerr(params, x, y, yerr):
    model = peak_model(params, x)
    return (model - y) / abs(yerr)


# objective with xerr
def objective_with_xerr(params, x, y, yerr, xerr):
    model = peak_model(params, x)
    dmodel_dx = np.gradient(model) / np.gradient(x)
    dmodel = np.sqrt(yerr**2 + (xerr*dmodel_dx)**2)
    return (model - y) / dmodel


# create a set of Parameters
params = Parameters()
params.add('amp', value=50, min=0)
params.add('cen', value=25)
params.add('sig', value=10)
params.add('slope', value=1.e-4)
params.add('offset', value=-5)

# first fit without xerr
mini1 = Minimizer(objective_no_xerr, params, fcn_args=(xdat, ydat, yerr))
result1 = mini1.minimize()
bestfit1 = peak_model(result1.params, xdat)


mini2 = Minimizer(objective_with_xerr, params, fcn_args=(xdat, ydat, yerr, xerr))
result2 = mini2.minimize()

bestfit2 = peak_model(result2.params, xdat)


print("### not including uncertainty in x:")
print(report_fit(result1))
print("### including uncertainty in x:")
print(report_fit(result2))

print(xdat[:4])

plt.plot(xdat, ydat, 'o', label='data with noise in x and y')
plt.plot(xtrue, ytrue, '-+', label='true data')
plt.plot(xdat, bestfit1, label='fit, no x error')
plt.plot(xdat, bestfit2, label='fit, with x error')
plt.legend()
plt.show()

# # <end examples/doc_uncertainties_in_x_and_y.py>
