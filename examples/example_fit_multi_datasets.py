"""
Fit Multiple Data Sets
======================

Fitting multiple (simulated) Gaussian data sets simultaneously.

All minimizers require the residual array to be one-dimensional. Therefore, in
the ``objective`` function we need to ``flatten`` the array before returning it.

TODO: this could/should be using the Model interface / built-in models!

"""
import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters, minimize, report_fit


def gauss(x, amp, cen, sigma):
    """Gaussian lineshape."""
    return amp * np.exp(-(x-cen)**2 / (2.*sigma**2))


def gauss_dataset(params, i, x):
    """Calculate Gaussian lineshape from parameters for data set."""
    amp = params[f'amp_{i+1}']
    cen = params[f'cen_{i+1}']
    sig = params[f'sig_{i+1}']
    return gauss(x, amp, cen, sig)


def objective(params, x, data):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]

    # make residual per data set
    for i in range(ndata):
        resid[i, :] = data[i, :] - gauss_dataset(params, i, x)

    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


###############################################################################
# Create five simulated Gaussian data sets
np.random.seed(2021)
x = np.linspace(-1, 2, 151)
data = []
for _ in np.arange(5):
    amp = 0.60 + 9.50*np.random.rand()
    cen = -0.20 + 1.20*np.random.rand()
    sig = 0.25 + 0.03*np.random.rand()
    dat = gauss(x, amp, cen, sig) + np.random.normal(size=x.size, scale=0.1)
    data.append(dat)
data = np.array(data)

###############################################################################
# Create five sets of fitting parameters, one per data set
fit_params = Parameters()
for iy, y in enumerate(data):
    fit_params.add(f'amp_{iy+1}', value=0.5, min=0.0, max=200)
    fit_params.add(f'cen_{iy+1}', value=0.4, min=-2.0, max=2.0)
    fit_params.add(f'sig_{iy+1}', value=0.3, min=0.01, max=3.0)

###############################################################################
# Constrain the values of sigma to be the same for all peaks by assigning
# sig_2, ..., sig_5 to be equal to sig_1.
for iy in (2, 3, 4, 5):
    fit_params[f'sig_{iy}'].expr = 'sig_1'

###############################################################################
# Run the global fit and show the fitting result
out = minimize(objective, fit_params, args=(x, data))
report_fit(out.params)

###############################################################################
# Plot the data sets and fits
plt.figure()
for i in range(5):
    y_fit = gauss_dataset(out.params, i, x)
    plt.plot(x, data[i, :], 'o', x, y_fit, '-')
