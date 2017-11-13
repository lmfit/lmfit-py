#!/usr/bin/env python

import numpy as np

from lmfit.models import ExpressionModel

try:
    import matplotlib.pyplot as plt
    HAS_PYLAB = True
except ImportError:
    HAS_PYLAB = False

x = np.linspace(-10, 10, 201)

amp, cen, wid = 3.4, 1.8, 0.5

y = amp * np.exp(-(x-cen)**2 / (2*wid**2)) / (np.sqrt(2*np.pi)*wid)
y = y + np.random.normal(size=len(x), scale=0.01)

gmod = ExpressionModel("amp * exp(-(x-cen)**2 /(2*wid**2))/(sqrt(2*pi)*wid)")

result = gmod.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())

if HAS_PYLAB:
    plt.plot(x, y, 'bo')
    plt.plot(x, result.init_fit, 'k--')
    plt.plot(x, result.best_fit, 'r-')
    plt.show()
