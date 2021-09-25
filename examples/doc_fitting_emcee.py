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
     0.1*np.random.randn(x.size))

p = lmfit.Parameters()
p.add_many(('a1', 4), ('a2', 4), ('t1', 3), ('t2', 3., True))


def residual(p):
    v = p.valuesdict()
    return v['a1']*np.exp(-x/v['t1']) + v['a2']*np.exp(-(x-0.1) / v['t2']) - y


mi = lmfit.minimize(residual, p, method='nelder', nan_policy='omit')
lmfit.printfuncs.report_fit(mi.params, min_correl=0.5)
if HASPYLAB:
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, residual(mi.params) + y, label='best fit')
    plt.legend()
    plt.show()

# Place bounds on the ln(sigma) parameter that emcee will automatically add
# to estimate the true uncertainty in the data since is_weighted=False
mi.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))

res = lmfit.minimize(residual, method='emcee', nan_policy='omit', burn=300,
                     steps=1000, thin=20, params=mi.params, is_weighted=False,
                     progress=False)

if HASPYLAB and HASCORNER:
    emcee_corner = corner.corner(res.flatchain, labels=res.var_names,
                                 truths=list(res.params.valuesdict().values()))
    plt.show()

if HASPYLAB:
    plt.plot(res.acceptance_fraction, 'o')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    plt.show()

if hasattr(res, "acor"):
    print("Autocorrelation time for the parameters:")
    print("----------------------------------------")
    for i, par in enumerate(p):
        print(par, res.acor[i])

print("\nmedian of posterior probability distribution")
print('--------------------------------------------')
lmfit.report_fit(res.params)


# find the maximum likelihood solution
highest_prob = np.argmax(res.lnprob)
hp_loc = np.unravel_index(highest_prob, res.lnprob.shape)
mle_soln = res.chain[hp_loc]
for i, par in enumerate(p):
    p[par].value = mle_soln[i]

print('\nMaximum Likelihood Estimation from emcee       ')
print('-------------------------------------------------')
print('Parameter  MLE Value   Median Value   Uncertainty')
fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
for name, param in p.items():
    print(fmt(name, param.value, res.params[name].value,
              res.params[name].stderr))

if HASPYLAB:
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, residual(mi.params) + y, label='Nelder-Mead')
    plt.plot(x, residual(res.params) + y, '--', label='emcee')
    plt.legend()
    plt.show()

print('\nError Estimates from emcee    ')
print('------------------------------------------------------')
print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma ')

for name in p.keys():
    quantiles = np.percentile(res.flatchain[name],
                              [2.275, 15.865, 50, 84.135, 97.275])
    median = quantiles[2]
    err_m2 = quantiles[0] - median
    err_m1 = quantiles[1] - median
    err_p1 = quantiles[3] - median
    err_p2 = quantiles[4] - median
    fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
    print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))
