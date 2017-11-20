#!/usr/bin/env python

# <examples/doc_model_savemodelresult.py>
import matplotlib.pyplot as plt
import numpy as np

from lmfit.model import save_modelresult
from lmfit.models import GaussianModel

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

gmodel = GaussianModel()
result = gmodel.fit(y, x=x, amplitude=5, center=5, sigma=1)

save_modelresult(result, 'gauss_modelresult.sav')

print(result.fit_report())

plt.plot(x, y, 'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()
# <end examples/doc_model_savemodelresult.py>
