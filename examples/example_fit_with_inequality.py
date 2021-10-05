"""
Fit Using Inequality Constraint
===============================

Sometimes specifying boundaries using ``min`` and ``max`` are not sufficient,
and more complicated (inequality) constraints are needed. In the example below
the center of the Lorentzian peak is constrained to be between 0-5 away from
the center of the Gaussian peak.

See also: https://lmfit.github.io/lmfit-py/constraints.html#using-inequality-constraints
"""
import matplotlib.pyplot as plt
import numpy as np

from lmfit import Minimizer, Parameters, report_fit
from lmfit.lineshapes import gaussian, lorentzian


def residual(pars, x, data):
    model = (gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g']) +
             lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l']))
    return model - data


###############################################################################
# Generate the simulated data using a Gaussian and Lorentzian lineshape:
np.random.seed(0)
x = np.linspace(0, 20.0, 601)

data = (gaussian(x, 21, 6.1, 1.2) + lorentzian(x, 10, 9.6, 1.3) +
        np.random.normal(scale=0.1, size=x.size))

###############################################################################
# Create the fitting parameters and set an inequality constraint for ``cen_l``.
# First, we add a new fitting  parameter ``peak_split``, which can take values
# between 0 and 5. Afterwards, we constrain the value for ``cen_l`` using the
# expression to be ``'peak_split+cen_g'``:
pfit = Parameters()
pfit.add(name='amp_g', value=10)
pfit.add(name='amp_l', value=10)
pfit.add(name='cen_g', value=5)
pfit.add(name='peak_split', value=2.5, min=0, max=5, vary=True)
pfit.add(name='cen_l', expr='peak_split+cen_g')
pfit.add(name='wid_g', value=1)
pfit.add(name='wid_l', expr='wid_g')

mini = Minimizer(residual, pfit, fcn_args=(x, data))
out = mini.leastsq()
best_fit = data + out.residual

###############################################################################
# Performing a fit, here using the ``leastsq`` algorithm, gives the following
# fitting results:
report_fit(out.params)

###############################################################################
# and figure:
plt.plot(x, data, 'o')
plt.plot(x, best_fit, '--', label='best fit')
plt.legend()
