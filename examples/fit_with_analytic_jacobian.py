#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from lmfit.lineshapes import gaussian, lorentzian, s2pi
from lmfit.models import GaussianModel, LorentzianModel


def dfunc_gaussian(params, *ys, **xs):
    x = xs['x']
    var = params.valuesdict()
    A, s, mu = var['amplitude'], var['sigma'], var['center']
    fac = np.exp(-(x-mu)**2/(2.*s**2))

    da = fac / (s2pi * s)
    ds = A * fac * ((x-mu)**2-s**2) / (s2pi*s**4)
    dmu = A * fac * (x-mu) / (s2pi*s**3)
    return np.array([ds, dmu, da])


def dfunc_lorentzian(params, *ys, **xs):
    xx = xs['x']
    var = params.valuesdict()
    A, s, mu = var['amplitude'], var['sigma'], var['center']
    fac = ((xx-mu)**2+s**2)

    ds = (A*((xx-mu)**2-s**2)) / (np.pi * fac**2)
    da = s / (np.pi*fac)
    dmu = (2. * A * (xx-mu)*s)/(np.pi * fac**2)
    return np.array([ds, dmu, da])


if __name__ == '__main__':
    xs = np.linspace(-4, 4, 100)

    print('**********************************')
    print('***** Test Gaussian **************')
    print('**********************************')
    ys = gaussian(xs, 2.5, 0, 0.5)
    yn = ys + 0.1*np.random.normal(size=len(xs))

    mod = GaussianModel()
    pars = mod.guess(yn, xs)
    out = mod.fit(yn, pars, x=xs)
    out2 = mod.fit(yn, pars,  x=xs, fit_kws={'Dfun': dfunc_gaussian, 'col_deriv': 1})
    print('lmfit without dfunc **************')
    print('number of function calls: ', out.nfev)
    print('params', out.best_values)
    print('lmfit with dfunc *****************')
    print('number of function calls: ', out2.nfev)
    print('params', out2.best_values)
    print('\n \n')
    out2.plot(datafmt='.')

    print('**********************************')
    print('***** Test Lorentzian ************')
    print('**********************************')
    ys = lorentzian(xs, 2.5, 0, 0.5)
    yn = ys + 0.1*np.random.normal(size=len(xs))

    mod = LorentzianModel()
    pars = mod.guess(yn, xs)
    out = mod.fit(yn, pars, x=xs)
    out2 = mod.fit(yn, pars, x=xs, fit_kws={'Dfun': dfunc_lorentzian, 'col_deriv': 1})
    print('lmfit without dfunc **************')
    print('number of function calls: ', out.nfev)
    print('params', out.best_values)
    print('lmfit with dfunc *****************')
    print('number of function calls: ', out2.nfev)
    print('params', out2.best_values)
    print('\n \n')
    out2.plot(datafmt='.')

    plt.show()
