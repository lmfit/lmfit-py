"""
Fit with Algebraic Constraint
=============================

"""
###############################################################################
# Example on how to use algebraic constraints using the ``expr`` attribute.
import matplotlib.pyplot as plt
from numpy import linspace, random

from lmfit.lineshapes import gaussian, lorentzian
from lmfit.models import GaussianModel, LinearModel, LorentzianModel

random.seed(0)
x = linspace(0.0, 20.0, 601)

data = (gaussian(x, amplitude=21, center=8.1, sigma=1.2) +
        lorentzian(x, amplitude=10, center=9.6, sigma=2.4) +
        0.01 + x*0.05 + random.normal(scale=0.23, size=x.size))


model = GaussianModel(prefix='g_') + LorentzianModel(prefix='l_') + LinearModel(prefix='line_')

params = model.make_params(g_amplitude=10, g_center=9, g_sigma=1,
                           line_slope=0, line_intercept=0)

params.add(name='total_amplitude', value=20)
params.set(l_amplitude=dict(expr='total_amplitude - g_amplitude'))
params.set(l_center=dict(expr='1.5+g_center'))
params.set(l_sigma=dict(expr='2*g_sigma'))


data_uncertainty = 0.021  # estimate of data error (for all data points)

init = model.eval(params, x=x)
result = model.fit(data, params, x=x, weights=1.0/data_uncertainty)

print(result.fit_report())

plt.plot(x, data, '+')
plt.plot(x, init, '--', label='initial fit')
plt.plot(x, result.best_fit, '-', label='best fit')
plt.legend()
plt.show()
