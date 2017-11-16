#!/usr/bin/env python

"""
Created on Sun Apr 15 19:47:45 2012

@author: Tillsten
"""
from numpy import exp, linspace, pi, random, sign, sin

from lmfit import Minimizer, Parameters, conf_interval, report_ci, report_fit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

p_true = Parameters()
p_true.add('amp', value=14.0)
p_true.add('period', value=5.33)
p_true.add('shift', value=0.123)
p_true.add('decay', value=0.010)


def residual(pars, x, data=None):
    argu = (x*pars['decay'])**2
    shift = pars['shift']
    if abs(shift) > pi/2:
        shift = shift - sign(shift)*pi
    model = pars['amp']*sin(shift + x/pars['period']) * exp(-argu)
    if data is None:
        return model
    return model - data


n = 2500
xmin = 0.
xmax = 250.0
noise = random.normal(scale=0.7215, size=n)
x = linspace(xmin, xmax, n)
data = residual(p_true, x) + noise

fit_params = Parameters()
fit_params.add('amp', value=13.0)
fit_params.add('period', value=2)
fit_params.add('shift', value=0.0)
fit_params.add('decay', value=0.02)

mini = Minimizer(residual, fit_params, fcn_args=(x,),
                 fcn_kws={'data': data})
out = mini.leastsq()

fit = residual(out.params, x)
report_fit(out)

ci, tr = conf_interval(mini, out, trace=True)
report_ci(ci)

if HASPYLAB:
    names = out.params.keys()
    i = 0
    gs = plt.GridSpec(4, 4)
    sx = {}
    sy = {}
    for fixed in names:
        j = 0
        for free in names:
            if j in sx and i in sy:
                ax = plt.subplot(gs[i, j], sharex=sx[j], sharey=sy[i])
            elif i in sy:
                ax = plt.subplot(gs[i, j], sharey=sy[i])
                sx[j] = ax
            elif j in sx:
                ax = plt.subplot(gs[i, j], sharex=sx[j])
                sy[i] = ax
            else:
                ax = plt.subplot(gs[i, j])
                sy[i] = ax
                sx[j] = ax
            if i < 3:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(free)

            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(fixed)

            res = tr[fixed]
            prob = res['prob']
            f = prob < 0.96

            x, y = res[free], res[fixed]
            ax.scatter(x[f], y[f], c=1-prob[f], s=200*(1-prob[f]+0.5))
            ax.autoscale(1, 1)
            j += 1
        i += 1
    plt.show()
