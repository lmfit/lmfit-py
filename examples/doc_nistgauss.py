#!/usr/bin/env python
#<examples/doc_nistgauss.py>
import numpy as np
from lmfit.models import GaussianModel, ExponentialModel

import matplotlib.pyplot as plt

dat = np.loadtxt('NIST_Gauss2.dat')
x = dat[:, 1]
y = dat[:, 0]

exp_mod = ExponentialModel(prefix='exp_')
exp_mod.guess_starting_values(y, x=x)

gauss1  = GaussianModel(prefix='g1_')
gauss2  = GaussianModel(prefix='g2_')

gauss1.set_param('center',    105, min=75, max=125)
gauss1.set_param('sigma',      15, min=3)
gauss1.set_param('amplitude', 2000, min=10)

gauss2.set_param('center',    155, min=125, max=175)
gauss2.set_param('sigma',      15, min=3)
gauss2.set_param('amplitude', 2000, min=10)

mod = gauss1 + gauss2 + exp_mod

out = mod.fit(y, x=x)

print(mod.fit_report(min_correl=0.5))

plt.plot(x, y)
plt.plot(x, out.init_fit, 'k--')
plt.plot(x, out.best_fit, 'r-')
plt.show()
#<end examples/doc_nistgauss.py>
