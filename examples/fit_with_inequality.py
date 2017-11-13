#!/usr/bin/env python

from numpy import linspace, random

from lmfit import Minimizer, Parameters, report_fit
from lmfit.lineshapes import gaussian, lorentzian

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def residual(pars, x, data):
    model = (gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g']) +
             lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l']))
    return model - data


n = 601
random.seed(0)
x = linspace(0, 20.0, n)

data = (gaussian(x, 21, 6.1, 1.2) +
        lorentzian(x, 10, 9.6, 1.3) +
        random.normal(scale=0.1, size=n))

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

report_fit(out.params)

best_fit = data + out.residual

if HASPYLAB:
    plt.plot(x, data, 'bo')
    plt.plot(x, best_fit, 'r--')
    plt.show()
