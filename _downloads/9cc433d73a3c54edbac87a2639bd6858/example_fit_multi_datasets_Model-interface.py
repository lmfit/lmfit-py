"""
Fit Multiple Data Sets Using Model Interface
============================================

Fitting multiple (simulated) Gaussian data sets simultaneously, using the
Model interface.

All minimizers require the residual array to be one-dimensional. Therefore,
in the ``objective`` function we need to ``flatten`` the array before
returning it.

"""
import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters, minimize, report_fit
from lmfit.models import GaussianModel

##############################################################################
# Create N simulated Gaussian data sets
N = 5
np.random.seed(2021)
x = np.linspace(-1, 2, 151)
data = []
for _ in np.arange(N):
    params = Parameters()
    params.add('amplitude', value=0.60 + 9.50*np.random.rand())
    params.add('center', value=-0.20 + 1.20*np.random.rand())
    params.add('sigma', value=0.25 + 0.03*np.random.rand())
    dat = (GaussianModel().eval(x=x, params=params) +
           np.random.normal(size=x.size, scale=0.1))
    data.append(dat)
data = np.array(data)


##############################################################################
# The objective function will extract and evaluate a Gaussian from the
# compound model
def objective(params, x, data, model):
    """Calculate total residual for fits of Gaussians to several data sets."""
    ndata, _ = data.shape
    resid = 0.0*data[:]

    # make residual per data set
    for i in range(ndata):
        components = model.components[i].eval(params=params, x=x)
        resid[i, :] = data[i, :] - components

    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


##############################################################################
# Create a composite model by adding Gaussians
model_arr = [GaussianModel(prefix=f'n{i+1}_') for i, _ in enumerate(data)]
model = sum(model_arr[1:], start=model_arr[0])

##############################################################################
# Prepare the fitting parameters and constrain n2_sigma, ..., nN_sigma to be
# equal to n1_sigma
fit_params = model.make_params()
for iy, y in enumerate(data):
    fit_params.add(f'n{iy+1}_amplitude', value=0.5, min=0.0, max=200)
    fit_params.add(f'n{iy+1}_center', value=0.4, min=-2.0, max=2.0)
    fit_params.add(f'n{iy+1}_sigma', value=0.3, min=0.01, max=3.0)

    if iy > 0:
        fit_params[f'n{iy+1}_sigma'].expr = 'n1_sigma'

##############################################################################
# Run the global fit and show the fitting result
out = minimize(objective, fit_params, args=(x, data, model))
report_fit(out.params)

##############################################################################
# Plot the data sets and fits
plt.figure()
for i, y in enumerate(data):
    components = model.eval_components(params=out.params, x=x)
    plt.plot(x, y, 'o', x, components[f'n{i+1}_'], '-')

plt.show()
