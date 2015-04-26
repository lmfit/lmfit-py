# -*- coding: utf-8 -*-
"""
Example using the t-likelihood for stable fitting.
"""
import numpy as np
import scipy.special as sp
import scipy.stats as stats
import lmfit
try:
    import matplotlib.pyplot as plt
    HAS_PYLAB = True
except ImportError:
    HAS_PYLAB = False

np.random.seed(1)
x = np.linspace(-5, 5, 30)

# Setup example
amp_erf = 5
w = 0.3
c = 1
amp_sin = 2
omega = 3

y = amp_erf*sp.erf(x/w)+2*np.sin(x*omega)+c
yn = y + np.random.randn(*y.shape)*1.5
outliers = np.random.random_integers(0, x.size-1, x.size/5)
yn[outliers] += np.random.randn(outliers.size)*10

if HAS_PYLAB:
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, lw=3)
    plt.plot(x, yn, 'o')

def func(x, amp_erf, w, c, omega, amp_sin):
    y_model = amp_erf*sp.erf(x/w) + amp_sin* np.sin(x*omega) + c
    return y_model

mod = lmfit.model.Model(func, ['x'])
res = mod.fit(yn, x=x, w=1, c=1, omega=3, amp_sin=2, amp_erf=3)

def logln(x):
    "Returning the t-log-likehood of x with df=3"
    return -stats.t.logpdf(x, df=2).sum()

o = res.scalar_minimize(method='L-BFGS-B', likelihood_fun=logln)

if HAS_PYLAB:
    plt.plot(x, res.best_fit, 'k', lw=3)
    plt.plot(x, res.eval(), 'r', lw=3)
    plt.legend(['True function', 'w. noise+outliers',
                'Fit', 'Robust Fit'], loc='upper left')

