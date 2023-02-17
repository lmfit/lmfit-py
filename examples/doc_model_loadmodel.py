# <examples/doc_model_loadmodel.py>
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from lmfit.model import load_model

if not os.path.exists('sinemodel.sav'):
    os.system(f"{sys.executable} doc_model_savemodel.py")


def mysine(x, amp, freq, shift):
    return amp * np.sin(x*freq + shift)


data = np.loadtxt('sinedata.dat')
x = data[:, 0]
y = data[:, 1]

model = load_model('sinemodel.sav', funcdefs={'mysine': mysine})
params = model.make_params(amp=dict(value=3, min=0),
                           freq=0.52,
                           shift=dict(value=0, min=-1, max=1))

result = model.fit(y, params, x=x)
print(result.fit_report())

plt.plot(x, y, 'o')
plt.plot(x, result.best_fit, '-')
plt.show()
# <end examples/doc_model_loadmodel.py>
