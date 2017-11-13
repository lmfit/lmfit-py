#!/usr/bin/env python

from numpy import linspace, random

from lmfit import Minimizer, Parameters, report_fit
from lmfit.lineshapes import gaussian

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False


def residual(pars, x, sigma=None, data=None):
    yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])

    model = yg + pars['line_off'] + x * pars['line_slope']
    if data is None:
        return model
    if sigma is None:
        return model - data
    return (model-data) / sigma


n = 201
xmin = 0.
xmax = 20.0
x = linspace(xmin, xmax, n)

p_true = Parameters()
p_true.add('amp_g', value=21.0)
p_true.add('cen_g', value=8.1)
p_true.add('wid_g', value=1.6)
p_true.add('line_off', value=-1.023)
p_true.add('line_slope', value=0.62)

data = (gaussian(x, p_true['amp_g'], p_true['cen_g'], p_true['wid_g']) +
        random.normal(scale=0.23,  size=n) +
        x*p_true['line_slope'] + p_true['line_off'])

if HASPYLAB:
    plt.plot(x, data, 'r+')

p_fit = Parameters()
p_fit.add('amp_g', value=10.0)
p_fit.add('cen_g', value=9)
p_fit.add('wid_g', value=1)
p_fit.add('line_slope', value=0.0)
p_fit.add('line_off', value=0.0)

myfit = Minimizer(residual, p_fit,
                  fcn_args=(x,),
                  fcn_kws={'sigma': 0.2, 'data': data})

myfit.prepare_fit()

for sigma in (0.1, 0.2, 0.22, 0.5):
    for scale_covar in (True, False):
        myfit.scale_covar = scale_covar
        print('====  scale_covar = %s, sigma=%.2f ===' % (myfit.scale_covar,
                                                          sigma))
        myfit.userkws['sigma'] = sigma

        p_fit['amp_g'].value = 10
        p_fit['cen_g'].value = 9
        p_fit['wid_g'].value = 1
        p_fit['line_slope'].value = 0.0
        p_fit['line_off'].value = 0.0

        out = myfit.leastsq()
        print('chi-square, reduced chi-square = %.3f, %.3f ' % (out.chisqr,
                                                                out.redchi))

        report_fit(out.params, modelpars=p_true, show_correl=False)
        print('====')
