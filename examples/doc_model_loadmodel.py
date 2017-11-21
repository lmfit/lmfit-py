#!/usr/bin/env python

# <examples/doc_model_loadmodel.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.model import load_model


def mysine(x, amp, freq, shift):
    return amp * np.sin(x*freq + shift)


data = np.loadtxt('sinedata.dat')
x = data[:, 0]
y = data[:, 1]

model = load_model('sinemodel.sav', funcdefs={'mysine': mysine})
params = model.make_params(amp=3, freq=0.52, shift=0)
params['shift'].max = 1
params['shift'].min = -1
params['amp'].min = 0.0

result = model.fit(y, params, x=x)
print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.best_fit, 'r-')
plt.show()
# <end examples/doc_model_loadmodel.py>
