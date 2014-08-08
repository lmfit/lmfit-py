#!/usr/bin/env python
#<examples/models_doc2.py>
from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import GaussianModel, LinearModel
import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1] + x  * 0.1 - 3.0

gauss = GaussianModel()
gauss.guess_starting_values(y, x=x)

line = LinearModel()
line.params['slope'].value  = 0
line.params['intercept'].value  = -1.0

total = gauss + line

result = total.fit(y, x=x)

print fit_report(result.params, min_correl=0.25)

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()

#<end examples/models_doc2.py>
