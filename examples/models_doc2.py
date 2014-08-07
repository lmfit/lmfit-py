#!/usr/bin/env python
#<model_docs2.py>
from numpy import loadtxt
from lmfit import fit_report
from lmfit.models import (VoigtModel, GaussianModel,
                          LorentzianModel, PseudoVoigtModel, LinearModel
import matplotlib.pyplot as plt

data = loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1] + x  * 0.1 - 3.0

model = PseudoVoigtModel()
model.guess_starting_values(y, x=x)

line = LinearModel()
line.params['slope'].value  = 0
line.params['intercept'].value  = -1.0

model = model + line

result = model.fit(y, x=x)

print fit_report(result.params, min_correl=0.25)

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()

#<end model_docs2.py>
