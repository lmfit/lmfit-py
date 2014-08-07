#!/usr/bin/env python
#<examples/models_doc1.py>
from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

model = GaussianModel()

model.guess_starting_values(y, x=x)

result = model.fit(y, x=x)

print fit_report(result.params, min_correl=0.25)

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()

#<end examples/models_doc1.py>
