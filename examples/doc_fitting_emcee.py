#!/usr/bin/env python

# <examples/doc_fitting_emcee.py>
import numpy as np

import lmfit

try:
    import matplotlib.pyplot as plt
    HASPYLAB = True
except ImportError:
    HASPYLAB = False

try:
    import corner
    HASCORNER = True
except ImportError:
    HASCORNER = False

x = np.linspace(1, 10, 250)
np.random.seed(0)
y = (3.0*np.exp(-x/2) - 5.0*np.exp(-(x-0.1) / 10.) +
     0.1*np.random.randn(len(x)))
if HASPYLAB:
    plt.plot(x, y, 'b')
    # plt.savefig('../doc/_images/emcee_dbl_exp.png')
    plt.show()

p = lmfit.Parameters()
p.add_many(('a1', 4), ('a2', 4), ('t1', 3), ('t2', 3., True))


def residual(p):
    v = p.valuesdict()
    return v['a1']*np.exp(-x/v['t1']) + v['a2']*np.exp(-(x-0.1) / v['t2']) - y


mi = lmfit.minimize(residual, p, method='Nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
if HASPYLAB:
    plt.figure()
    plt.plot(x, y, 'b')
    plt.plot(x, residual(mi.params) + y, 'r')
    # plt.savefig('../doc/_images/emcee_dbl_exp2.png')
    plt.show()

# Place bounds on the ln(sigma) parameter that emcee will automatically add
# to estimate the true uncertainty in the data since is_weighted=False
mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300,
                     steps=1000, thin=20, params=mi.params, is_weighted=False)

if HASPYLAB and HASCORNER:
    emcee_corner = corner.corner(res.flatchain, labels=res.var_names,
                                 truths=list(res.params.valuesdict().values()))
    # emcee_corner.savefig('../doc/_images/emcee_corner.png')
    plt.show()

print("\nmedian of posterior probability distribution")
print('--------------------------------------------')
lmfit.report_fit(res.params)

# find the maximum likelihood solution
highest_prob = np.argmax(res.lnprob)
hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
mle_soln = res.chain[hp_loc]
for i, par in enumerate(p):
    p[par].value = mle_soln[i]
print("\nMaximum likelihood Estimation")
print('-----------------------------')
print(p)

if HASPYLAB:
    plt.figure()
    plt.plot(x, y, 'b')
    plt.plot(x, residual(mi.params) + y, 'r', label='Nelder-Mead')
    plt.plot(x, residual(res.params) + y, 'k--', label='emcee')
    plt.legend()
    plt.show()

quantiles = np.percentile(res.flatchain['t1'], [2.28, 15.9, 50, 84.2, 97.7])
print("1 sigma spread", 0.5 * (quantiles[3] - quantiles[1]))
print("2 sigma spread", 0.5 * (quantiles[4] - quantiles[0]))
# <end of examples/doc_fitting_emcee.py>
