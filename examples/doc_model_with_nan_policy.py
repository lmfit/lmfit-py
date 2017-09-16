#!/usr/bin/env python
#<examples/doc_model_with_nan_policy.py>
import numpy as np

from lmfit.models import  GaussianModel

import matplotlib.pyplot as plt

data = np.loadtxt('model1d_gauss.dat')
x = data[:, 0]
y = data[:, 1]

y[44] = np.nan
y[65] = np.nan

# nan_policy = 'raise'
# nan_policy = 'propagate'
nan_policy = 'omit'

gmodel = GaussianModel()
result = gmodel.fit(y, x=x, amplitude=5, center=6, sigma=1,
                    nan_policy=nan_policy)

print(result.fit_report())

plt.plot(x, y,         'bo')
plt.plot(x, result.init_fit, 'k--')
plt.plot(x, result.best_fit, 'r-')
plt.show()
#<end examples/doc_model_with_nan_policy.py>
