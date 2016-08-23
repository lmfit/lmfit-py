#!/usr/bin/env python
#<examples/doc_model4.py>
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit import  Model

import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

def gaussian(x, amp, cen, wid):
    "1-d gaussian: gaussian(x, amp, cen, wid)"
    return (amp/(sqrt(2*pi)*wid)) * exp(-(x-cen)**2 /(2*wid**2))

gmod = Model(gaussian)
result = gmod.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())
dely = result.eval_uncertainty(sigma=3)

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")
plt.show()
#<end examples/doc_model4.py>
