#!/usr/bin/env python
#<examples/doc_load_model.py>
import numpy as np
import matplotlib.pyplot as plt

from lmfit.model import load_model

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]


model = load_model('tmp_save_model.sav')

result = model.fit(y, x=x, amplitude=5, center=5, sigma=1)

print(result.fit_report())


plt.plot(x, y,         'bo')
plt.plot(x, result.best_fit, 'r-')
plt.show()


#<end examples/doc_load_model.py>
