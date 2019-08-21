"""
Fit with Algebraic Constraint
=============================


"""
import matplotlib.pyplot as plt
from numpy import linspace, random

from lmfit import Minimizer, Parameters
from lmfit.lineshapes import gaussian, lorentzian
from lmfit.printfuncs import report_fit


def residual(pars, x, sigma=None, data=None):
    yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])
    yl = lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l'])

    slope = pars['line_slope']
    offset = pars['line_off']
    model = yg + yl + offset + x*slope

    if data is None:
        return model
    if sigma is None:
        return model - data
    return (model - data) / sigma


random.seed(0)
x = linspace(0.0, 20.0, 601)

data = (gaussian(x, 21, 8.1, 1.2) +
        lorentzian(x, 10, 9.6, 2.4) +
        random.normal(scale=0.23, size=x.size) +
        x*0.5)


pfit = Parameters()
pfit.add(name='amp_g', value=10)
pfit.add(name='cen_g', value=9)
pfit.add(name='wid_g', value=1)
pfit.add(name='amp_tot', value=20)
pfit.add(name='amp_l', expr='amp_tot - amp_g')
pfit.add(name='cen_l', expr='1.5+cen_g')
pfit.add(name='wid_l', expr='2*wid_g')
pfit.add(name='line_slope', value=0.0)
pfit.add(name='line_off', value=0.0)

sigma = 0.021  # estimate of data error (for all data points)

myfit = Minimizer(residual, pfit,
                  fcn_args=(x,), fcn_kws={'sigma': sigma, 'data': data},
                  scale_covar=True)

result = myfit.leastsq()
init = residual(pfit, x)
fit = residual(result.params, x)

report_fit(result)

plt.plot(x, data, 'r+')
plt.plot(x, init, 'b--', label='initial fit')
plt.plot(x, fit, 'k-', label='best fit')
plt.legend(loc='best')
plt.show()
